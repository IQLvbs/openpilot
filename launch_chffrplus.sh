#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

export HOME="${HOME:-/home/comma}"
export IQPILOT_PROPRIETARY_ROOT="$DIR/artifacts"
source "$DIR/launch_env.sh"
export PATH="/usr/local/venv/bin:$PATH"

function agnos_version_allowed {
  local current_version="${1#IQ.OS }"
  local expected_version="${2#IQ.OS }"
  local compat_version

  if [ "$current_version" = "$expected_version" ] || [[ "$current_version" = "$expected_version"-* ]]; then
    return 0
  fi

  IFS=',' read -r -a compat_versions <<< "${AGNOS_COMPAT_VERSIONS:-}"
  for compat_version in "${compat_versions[@]}"; do
    compat_version="${compat_version#"${compat_version%%[![:space:]]*}"}"
    compat_version="${compat_version%"${compat_version##*[![:space:]]}"}"
    compat_version="${compat_version#IQ.OS }"
    if [ -n "$compat_version" ] && [ "$current_version" = "$compat_version" ]; then
      return 0
    fi
  done

  return 1
}

function set_lite_hw() {
  if grep -q "tici" /sys/firmware/devicetree/base/model 2>/dev/null; then
    output=$(i2cget -y 0 0x10 0x00 2>/dev/null)
    if [ -z "$output" ]; then
      echo "C3 Lite hardware detected"
      export LITE=1
      echo "1" > /tmp/lite_hw
    fi
  fi
}

function install_iq_command() {
  local rc_file="$HOME/.bashrc"

  [ -w "$HOME" ] || return 0
  touch "$rc_file" 2>/dev/null || return 0
  local rc_temp="${rc_file}.iq-tmp.$$"
  awk '/^# >>> iq command/{skip=1} /^# <<< iq command/{skip=0; next} skip{next}
       !/^alias op=.*tools\/op\.sh/ && !/^alias iq=.*tools\/iq\.sh/' "$rc_file" > "$rc_temp" || return 0
  mv "$rc_temp" "$rc_file" || return 0
  cat >> "$rc_file" <<'IQFN'
# >>> iq command (auto-managed, do not edit) >>>
unalias iq op 2>/dev/null
iq() {
  local d="$PWD"
  while [ "$d" != "/" ]; do
    if [ -x "$d/iqpilot/tools/iq.sh" ] && { [ -f "$d/launch_iqpilot.sh" ] || [ -f "$d/launch_openpilot.sh" ]; }; then
      "$d/iqpilot/tools/iq.sh" "$@"; return
    fi
    d="$(dirname "$d")"
  done
  for d in /data/iqpilot /data/openpilot; do
    if [ -x "$d/iqpilot/tools/iq.sh" ] && { [ -f "$d/launch_iqpilot.sh" ] || [ -f "$d/launch_openpilot.sh" ]; }; then
      "$d/iqpilot/tools/iq.sh" "$@"; return
    fi
  done
  echo "iq: no IQ.Pilot checkout found (searched up from $PWD, then /data/iqpilot, /data/openpilot)" >&2
  return 1
}
# <<< iq command (auto-managed) <<<
IQFN
}

function agnos_init {
  # On IQ.OS 4.9.9.2+ the root iqpilot-boot-prepare oneshot performs this setup (no owner sudo, verity-safe rootfs); its done flag means skip.
  if [ ! -f /run/iqpilot-boot-prepare.done ]; then
    # TODO: move this to agnos
    sudo -n rm -f /data/etc/NetworkManager/system-connections/*.nmmeta 2>/dev/null || true

    sudo -n abctl --set_success 2>/dev/null || true

    sudo -n chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0 2>/dev/null || true
    sudo -n chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0 2>/dev/null || true

    if [ "$(readlink /etc/alternatives/iptables)" != "/usr/sbin/iptables-legacy" ]; then
      sudo -n mount -o remount,rw / 2>/dev/null || true
      sudo -n ln -sf /usr/sbin/iptables-legacy /etc/alternatives/iptables 2>/dev/null || true
      sudo -n ln -sf /usr/sbin/iptables-legacy-restore /etc/alternatives/iptables-restore 2>/dev/null || true
      sudo -n ln -sf /usr/sbin/iptables-legacy-save /etc/alternatives/iptables-save 2>/dev/null || true
      sudo -n mount -o remount,ro / 2>/dev/null || true
    fi
  fi

  # Check if AGNOS update is required
  CURRENT_AGNOS_VERSION=$(< /VERSION)
  if ! agnos_version_allowed "$CURRENT_AGNOS_VERSION" "$AGNOS_VERSION"; then
    AGNOS_PY="$DIR/iqpilot/system/hardware/tici/agnos.py"
    MANIFEST="$DIR/iqpilot/system/hardware/tici/agnos.json"
    if [ -f /sys/firmware/devicetree/base/model ]; then
      DEVICE_MODEL="$(tr -d '\0' </sys/firmware/devicetree/base/model)"
      case "$DEVICE_MODEL" in
        *"comma tici"*|*"comma three"*)
          # comma 3 keeps its stock 15.1 bootloaders: comma's 16 abl has a board-id
          # check that rejects comma 3 ("Unsupported firmware detected"). It still gets
          # the IQ boot (comma_tici.dtb) + IQ.OS 3.4 system via this hybrid manifest.
          MANIFEST="$DIR/iqpilot/system/hardware/tici/agnos_tici_15_1.json"
          ;;
      esac
    fi
    if $AGNOS_PY --verify $MANIFEST; then
      sudo -n reboot
    fi
    # On stock AGNOS the updater picks updater_weston (weston is up), a compiled display binary that
    # can't get a compositor surface on a first boot, so the IQ.OS flash silently wedges on the comma
    # logo. There, flash headless via agnos.py --swap (identical download+write+swap, no surface).
    # On IQ.OS weston is down -> the updater picks updater_magic (DRM), which renders the on-screen
    # progress prompt fine, so fall through and keep it. This mirrors the updater's own weston check.
    if systemctl is-active --quiet weston-ready; then
      if $AGNOS_PY --swap $MANIFEST; then
        sudo -n reboot
      fi
    fi
    $DIR/iqpilot/system/hardware/tici/updater $AGNOS_PY $MANIFEST
  fi
}

function launch {
  # Remove orphaned git lock if it exists on boot
  [ -f "$DIR/.git/index.lock" ] && rm -f $DIR/.git/index.lock
  for server_dir in "$HOME/.cursor-server" "$HOME/.windsurf-server" "$HOME/.vscode-server"; do
    [ -L "$server_dir" ] && unlink "$server_dir"
  done

  install_iq_command

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The DIR init file has to exist, with a newer modtime than anything in
  #    the DIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${DIR}/.overlay_init" ]; then
    find ${DIR}/.git -newer ${DIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${DIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $DIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $DIR
          cd $DIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # Python env + proprietary runtime sync. Lives in artifacts/runtime/env_sync.sh so updated.py
  # can run the identical steps at update-apply time (offroad, stack idle); by the
  # time this runs at boot it is normally a warm no-op. Real work still lands here
  # for local dev edits or a failed apply-time prep, under the boot spinner as before.
  source "$DIR/artifacts/runtime/env_sync.sh"
  sync_python_env || return 1
  if [ ! -f "$DIR/prebuilt" ] && [ ! -x "$DIR/.venv/bin/python3" ]; then
    return 1
  fi
  export PATH="$DIR/.venv/bin:$PATH"

  RUNTIME_COMPAT_ROOT="$DIR/.iqpilot/runtime_root"
  mkdir -p "$RUNTIME_COMPAT_ROOT"
  ln -sfn "$DIR/iqpilot" "$RUNTIME_COMPAT_ROOT/iqpilot"
  ln -sfn iqpilot "$RUNTIME_COMPAT_ROOT/openpilot"
  ln -sfn "$DIR/iqpilot/system" "$RUNTIME_COMPAT_ROOT/system"

  ln -sfn $(pwd) /data/pythonpath
  export IQPILOT_SOURCE_ROOT="${IQPILOT_SOURCE_ROOT:-$DIR/iqpilot}"
  export PYTHONSAFEPATH=1
  VERIFIED_PYTHON_ROOT="/usr/libexec/iqpilot/python"
  IQPILOT_PYTHONPATH="$VENV_SITE_PACKAGES:$PWD"
  if [ -n "$(iq_package_runtime_root "$PWD")" ]; then
    IQPILOT_PYTHONPATH="$PWD/artifacts/package_runtime:$IQPILOT_PYTHONPATH"
  fi
  if [ -d "$VERIFIED_PYTHON_ROOT" ]; then
    export PYTHONPATH="$VERIFIED_PYTHON_ROOT:$IQPILOT_PYTHONPATH"
  else
    export PYTHONPATH="$IQPILOT_PYTHONPATH"
  fi

  if iq_service_dropins_missing && [ -x "$IQOS_SERVICE_DROPINS_HELPER" ]; then
    sudo -n "$IQOS_SERVICE_DROPINS_HELPER" 2>/dev/null || true
  fi

  # Install independent systemd services BEFORE build so they run even
  # when openpilot fails to compile — SSH/BLE/hephaestusd must always
  # be reachable for device recovery.
  # On IQ.OS 4.9.9.2+ the root boot-prepare oneshot already installed+started these; skip the owner-sudo path.
  # Exception: if boot-prepare ran before the checkout existed (first boot after flash), the services weren't started.
  if [ ! -f /run/iqpilot-boot-prepare.done ] || ! systemctl is-active --quiet hephaestusd.service; then
  for service_name in hephaestusd ble-transportd flockd; do
    service_src="$DIR/iqpilot/system/${service_name}.service"
    service_dst="/etc/systemd/system/${service_name}.service"
    service_lib="/lib/systemd/system/${service_name}.service"
    service_dropin="/run/systemd/system/${service_name}.service.d"
    service_exec="$(grep '^ExecStart=' "$service_src")"
    sudo -n mkdir -p "$service_dropin" 2>/dev/null || mkdir -p "$service_dropin" 2>/dev/null || true
    printf '[Service]\nWorkingDirectory=%s\nEnvironment="IQPILOT_SOURCE_ROOT=%s/iqpilot"\nEnvironment="IQPILOT_PROPRIETARY_ROOT=%s/artifacts"\nEnvironment="PYTHONPATH=%s"\nExecStart=\n%s\n' "$RUNTIME_COMPAT_ROOT" "$DIR" "$DIR" "$PYTHONPATH" "$service_exec" | sudo -n tee "$service_dropin/iqpilot-packages.conf" >/dev/null 2>&1 || true
    sudo -n systemctl daemon-reload 2>/dev/null || true
    if [ -f "$service_lib" ] && grep -q "/usr/libexec/iqpilot/iqpilot_bundle_runner" "$service_lib"; then
      if [ -f "$service_dst" ]; then
        sudo -n mount -o remount,rw / 2>/dev/null || true
        sudo -n rm -f "$service_dst" 2>/dev/null || true
        sudo -n systemctl daemon-reload 2>/dev/null || true
        sudo -n mount -o remount,ro / 2>/dev/null || true
      fi
      sudo -n systemctl enable "${service_name}.service" 2>/dev/null || true
      # --no-block: these units ExecStartPre-wait for the verified runtime. Blocking here
      # made a missing runner stall the whole install for the unit's 600s timeout (x3 units).
      if systemctl is-active --quiet "${service_name}.service"; then
        sudo -n systemctl restart --no-block "${service_name}.service" 2>/dev/null || true
      else
        sudo -n systemctl start --no-block "${service_name}.service" 2>/dev/null || true
      fi
    elif [ -f "$service_src" ]; then
      if [ ! -f "$service_dst" ] || ! cmp -s "$service_src" "$service_dst"; then
        sudo -n mount -o remount,rw / 2>/dev/null || true
        sudo -n cp "$service_src" "$service_dst" 2>/dev/null || true
        sudo -n systemctl daemon-reload 2>/dev/null || true
        sudo -n systemctl enable "${service_name}.service" 2>/dev/null || true
        sudo -n mount -o remount,ro / 2>/dev/null || true
        sudo -n systemctl restart --no-block "${service_name}.service" 2>/dev/null || true
      elif ! systemctl is-active --quiet "${service_name}.service"; then
        sudo -n systemctl start --no-block "${service_name}.service" 2>/dev/null || true
      fi
    fi
  done
  fi

  # detect mr.one C3 Lite hardware
  set_lite_hw

  # hardware specific init
  if [ -f /AGNOS ]; then
    if ! id -nG 2>/dev/null | grep -qw otaflash; then
      sudo -n usermod -aG otaflash "$(whoami)" 2>/dev/null || true
    fi
    agnos_init

    # IQ.OS 4.9.9.2+ sets up zram via the baked iqos-zram.service; only the legacy owner-sudo path runs it here.
    if [ ! -f /run/iqpilot-boot-prepare.done ]; then
      sudo -n "$DIR/iqpilot/system/hardware/tici/zram_setup.sh" 2>/dev/null || true
    fi
  fi

  # IQ.OS 4.9.9.2+ bakes branding (bg.jpg, splash, passwd/shadow) into the image; applying from /data as root is an escalation path, so skip it there.
  if [ ! -f /run/iqpilot-boot-prepare.done ] && [ -f "$DIR/artifacts/runtime/apply_boot_branding.py" ]; then
    sudo -n python3 "$DIR/artifacts/runtime/apply_boot_branding.py" 2>/dev/null || true
  fi

  # /home is an ephemeral overlay (resets each boot); re-inject the source line. best-effort
  if [ -f "$DIR/iqpilot/tools/iqpilot/git-pretty.sh" ] && [ -w "$HOME/.bashrc" ]; then
    grep -q 'iqpilot/git-pretty.sh' "$HOME/.bashrc" 2>/dev/null || \
      echo "[ -f $DIR/iqpilot/tools/iqpilot/git-pretty.sh ] && source $DIR/iqpilot/tools/iqpilot/git-pretty.sh" >> "$HOME/.bashrc" || true
  fi

  # write tmux scrollback to a file
  tmux capture-pane -pq -S-1000 > /tmp/launch_log

  # start manager
  cd "$DIR/iqpilot/system/manager"
  export PWD="$(pwd)"
  MANAGER_PYTHON="$DIR/.venv/bin/python3"
  if [ ! -f $DIR/prebuilt ]; then
    if pkill -f /tmp/installer 2>/dev/null; then sleep 1; fi
    "$MANAGER_PYTHON" ./build.py
    for service_name in hephaestusd ble-transportd flockd; do
      sudo -n systemctl restart --no-block "${service_name}.service" 2>/dev/null || true
    done
  else
    MANAGER_PYTHON="/usr/local/venv/bin/python3"
  fi

  "$MANAGER_PYTHON" ./manager.py

  # if broken, keep on screen error
  while true; do sleep 1; done
}

launch
