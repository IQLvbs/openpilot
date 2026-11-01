#!/bin/bash
# USB mass-storage gadget exposing a snapshot of /data/media/0/realdata (dashcam clips + logs)
# over the same configfs gadget mechanism as /usr/comma/set_adb.sh. openpilot keeps running; the
# export is a read-only snapshot built at enable time, not a live view of realdata.
#
# The device only has one physical USB controller (UDC), so ADB and USB storage must live in the
# SAME composite gadget (/config/usb_gadget/g1) rather than each owning their own. Earlier versions
# of this script called /usr/comma/set_adb.sh as a black box and then unbound/rebound around it,
# but that intermediate bind/unbind churn made the *next* bind flaky (functionfs needs its
# userspace side, adbd, settled before the gadget can (re)bind). So instead we replicate set_adb.sh's
# handful of setup lines directly here and do exactly one bind at the end, covering whichever
# functions (ADB, mass storage) are currently enabled.
#
# Without composing like this, comma's adb-param-watcher systemd unit (which fires whenever
# /data/params/d/AdbEnabled is touched, even to the same value) would rebuild g1 with only its own
# functions and silently drop ours.

set -e

# serialize invocations: rapid toggling can otherwise race on the same /config/usb_gadget/g1 tree
# and leave it in a half-built state
LOCKFILE="/tmp/set_usb_storage.lock"
exec 9>"$LOCKFILE"
flock 9

IMG="/data/media/0/usb_storage.img"
LOOP_MNT="/tmp/usb_storage_mnt"
REALDATA="/data/media/0/realdata"
UDC_NAME="a600000.dwc3"
GADGET="/config/usb_gadget/g1"
SAFETY_MARGIN_KB=$((2 * 1024 * 1024))   # keep 2GB free on /data after the image
CAP_KB=$((4 * 1024 * 1024))             # never build more than a 4GB snapshot (FAT32 + dir overhead eats into this)

build_image() {
  avail_kb=$(df --output=avail -k /data | tail -1)
  budget_kb=$((avail_kb - SAFETY_MARGIN_KB))
  if [ "$budget_kb" -gt "$CAP_KB" ]; then
    budget_kb=$CAP_KB
  fi
  if [ "$budget_kb" -lt $((512 * 1024)) ]; then
    echo "Not enough free space on /data to build a USB storage snapshot" >&2
    exit 1
  fi

  echo "Building ${budget_kb}KB FAT32 snapshot image at $IMG"
  sudo rm -f "$IMG"
  sudo fallocate -l "${budget_kb}K" "$IMG" || sudo dd if=/dev/zero of="$IMG" bs=1M count=$((budget_kb / 1024))
  sudo mkfs.vfat -F 32 -n IQPILOT "$IMG"

  sudo mkdir -p "$LOOP_MNT"
  LOOP_DEV=$(sudo losetup -f)
  sudo losetup "$LOOP_DEV" "$IMG"
  sudo mount -t vfat "$LOOP_DEV" "$LOOP_MNT"

  # select the most recent files up to budget, then copy them in one rsync
  # pass (this script already runs as root, and one process beats thousands
  # of per-file forked sudo/mkdir/cp calls, which was previously the actual
  # bottleneck, not disk throughput).
  copy_budget_kb=$((budget_kb * 90 / 100))
  filelist=$(mktemp)
  find "$REALDATA" -type f -printf '%T@ %s %P\n' 2>/dev/null | sort -rn | awk -v budget="$copy_budget_kb" '
    { used += int(($2 + 1023) / 1024); if (used > budget) { exit } print $3 }
  ' > "$filelist"
  mkdir -p "$LOOP_MNT/realdata"
  # FAT32 has no concept of unix owner/group/perms, so don't ask rsync to preserve them
  rsync -rt --files-from="$filelist" "$REALDATA/" "$LOOP_MNT/realdata/"
  echo "Copied $(wc -l < "$filelist") files into snapshot"
  rm -f "$filelist"

  sudo umount "$LOOP_MNT"
  sudo losetup -d "$LOOP_DEV"
}

unbind() {
  if [ -d "$GADGET" ]; then
    cd "$GADGET"
    echo "" | sudo tee UDC >/dev/null 2>&1 || true
  fi
}

set_attr() {
  [ "$(cat "$1" 2>/dev/null)" = "$2" ] && return 0
  echo "$2" | sudo tee "$1" >/dev/null 2>&1 || true
}

ensure_base() {
  if ! mountpoint -q /config; then
    sudo mount -t configfs none /config
  fi
  sudo mkdir -p "$GADGET/strings/0x409" "$GADGET/configs/c.1/strings/0x409"
  cd "$GADGET"
  # `[ -s ]` never guards a configfs attribute: an unset idVendor still reads back
  # as "0x0000", so those writes were all skipped and the gadget stayed nameless
  set_attr idVendor 0x04D8
  set_attr idProduct 0x1235
  set_attr strings/0x409/serialnumber "$(sed -e 's/^.*androidboot.serialno=//' -e 's/ .*$//' /proc/cmdline)"
  set_attr strings/0x409/manufacturer "comma.ai"
  set_attr strings/0x409/product "IQ.Pilot"
  set_attr configs/c.1/MaxPower 250
  set_attr configs/c.1/strings/0x409/configuration "IQ.Pilot"
  enable_os_desc
}

# MS OS descriptors let Windows bind WinUSB to the NutCracker bulk interface with no driver install.
# Windows caches a "no OS descriptors" answer per VID/PID/bcdDevice, so bcdDevice changes with this.
enable_os_desc() {
  [ -d "$GADGET/os_desc" ] || return 0
  set_attr bcdDevice 0x0501
  set_attr os_desc/b_vendor_code 0xcd
  set_attr os_desc/qw_sign MSFT100
  set_attr os_desc/use 1
  [ -L "$GADGET/os_desc/c.1" ] || sudo ln -s "$GADGET/configs/c.1" "$GADGET/os_desc/c.1" 2>/dev/null || true
}

add_adb() {
  # same rationale as add_mass_storage: start from a clean slate to avoid stale busy attributes
  remove_adb
  cd "$GADGET"
  sudo mkdir -p functions/ffs.adb
  sudo mkdir -p /dev/usb-ffs/adb
  if ! mountpoint -q /dev/usb-ffs/adb; then
    sudo mount -t functionfs adb /dev/usb-ffs/adb
  fi
  sudo rm -f configs/c.1/ffs.adb
  sudo ln -s functions/ffs.adb configs/c.1/
  setprop service.adb.tcp.port -1 2>/dev/null || true
  sudo systemctl start adbd
  # adbd needs a moment to open the ffs endpoint and negotiate descriptors before the gadget can bind
  sleep 1
}

remove_adb() {
  sudo systemctl stop adbd || true
  if [ -d "$GADGET" ]; then
    cd "$GADGET"
    sudo rm -f configs/c.1/ffs.adb
    sudo umount /dev/usb-ffs/adb 2>/dev/null || true
    sudo rmdir functions/ffs.adb 2>/dev/null || true
  fi
}

# ncm carries the usb0 ethernet link. ADB needs it, but so does the Mac-backed
# model worker with ADB off, so it is enabled independently of either.
# the kernel randomises the ncm MACs every boot, so macOS sees a new adapter each
# time and orphans the network service holding the link's static address
ncm_id() {
  local id
  id=$(tr -dc '0-9a-f' < /data/params/d/DongleId 2>/dev/null | tail -c 6)
  [ ${#id} -eq 6 ] || id="000001"
  echo "$id"
}

add_ncm() {
  remove_ncm
  cd "$GADGET"
  sudo mkdir -p functions/ncm.0
  local id
  id=$(ncm_id)
  # best effort: some kernels create the ncm netdev lazily and fail these writes
  # with ENODEV, and a pinned MAC is never worth losing the whole gadget over
  echo "02:49:51:${id:0:2}:${id:2:2}:${id:4:2}" | sudo tee functions/ncm.0/host_addr >/dev/null 2>&1 || true
  echo "06:49:51:${id:0:2}:${id:2:2}:${id:4:2}" | sudo tee functions/ncm.0/dev_addr >/dev/null 2>&1 || true
  sudo rm -f configs/c.1/ncm.0
  sudo ln -s functions/ncm.0 configs/c.1/
}

remove_ncm() {
  if [ -d "$GADGET" ]; then
    cd "$GADGET"
    sudo rm -f configs/c.1/ncm.0
    sudo rmdir functions/ncm.0 2>/dev/null || true
  fi
}

add_mass_storage() {
  # a function group that's ever been bound before can refuse attribute writes ("Device or
  # resource busy") until it's torn down and recreated fresh, so always start from a clean slate
  remove_mass_storage
  cd "$GADGET"
  sudo mkdir -p functions/mass_storage.0
  echo 1 | sudo tee functions/mass_storage.0/stall >/dev/null
  echo 1 | sudo tee functions/mass_storage.0/lun.0/removable >/dev/null
  echo 1 | sudo tee functions/mass_storage.0/lun.0/ro >/dev/null
  echo "$IMG" | sudo tee functions/mass_storage.0/lun.0/file >/dev/null
  sudo rm -f configs/c.1/mass_storage.0
  sudo ln -s functions/mass_storage.0 configs/c.1/
}

remove_mass_storage() {
  if [ -d "$GADGET" ]; then
    cd "$GADGET"
    sudo rm -f configs/c.1/mass_storage.0
    sudo rmdir functions/mass_storage.0 2>/dev/null || true
  fi
}

bind() {
  cd "$GADGET"
  for attempt in $(seq 1 20); do
    if echo "$UDC_NAME" | sudo tee UDC >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "$UDC_NAME" | sudo tee UDC
}

read_bool_param() {
  [ -f "$1" ] && [ "$(< "$1")" == "1" ]
}

USB_STORAGE_ENABLE=0
read_bool_param "/data/params/d/UsbStorageEnabled" && USB_STORAGE_ENABLE=1
ADB_ENABLE=0
read_bool_param "/data/params/d/AdbEnabled" && ADB_ENABLE=1
EMAC_ENABLE=0
read_bool_param "/data/params/d/IQEmacEnabled" && EMAC_ENABLE=1

unbind
ensure_base

if [ "$ADB_ENABLE" == "1" ] || [ "$EMAC_ENABLE" == "1" ]; then
  add_ncm
else
  remove_ncm
fi

if [ "$ADB_ENABLE" == "1" ]; then
  add_adb
else
  remove_adb
fi

if [ "$USB_STORAGE_ENABLE" == "1" ]; then
  echo "Enabling USB storage mode"
  if [ ! -f "$IMG" ] || [ "$1" == "--rebuild" ]; then
    build_image
  fi
  add_mass_storage
else
  echo "Disabling USB storage mode"
  remove_mass_storage
fi

bind
