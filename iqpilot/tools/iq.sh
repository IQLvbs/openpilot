#!/usr/bin/env bash
# Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos

set -euo pipefail

IQ_RESET='\033[0m'
IQ_BOLD='\033[1m'
IQ_MUTED='\033[38;5;245m'
IQ_RED='\033[38;5;204m'
IQ_GREEN='\033[38;5;120m'
IQ_CYAN='\033[38;2;38;145;184m'
IQ_BLUE='\033[38;2;51;112;176m'
IQ_PURPLE='\033[38;5;141m'
IQ_PINK='\033[38;5;211m'
IQ_ROOT=''
IQ_DRY=0
IQ_NO_VERIFY=0
IQ_LOG_FILE=''
IQ_REBOOT=0

if [[ ! -t 1 || "${NO_COLOR:-}" != '' ]]; then
  IQ_RESET='' IQ_BOLD='' IQ_MUTED='' IQ_RED='' IQ_GREEN='' IQ_CYAN='' IQ_BLUE='' IQ_PURPLE='' IQ_PINK=''
fi

iq_line() {
  printf '%b%s%b\n' "$IQ_CYAN" '━━━━━━━━━━━━' "$IQ_RESET"
}

iq_title() {
  printf '%bI%b%bQ%b%b.%b%bP%b%bi%b%bl%b%bo%b%bt%b  %b%s%b\n' "$IQ_CYAN" "$IQ_RESET" "$IQ_BLUE" "$IQ_RESET" "$IQ_PURPLE" "$IQ_RESET" "$IQ_PINK" "$IQ_RESET" "$IQ_PURPLE" "$IQ_RESET" "$IQ_BLUE" "$IQ_RESET" "$IQ_CYAN" "$IQ_RESET" "$IQ_CYAN" "$IQ_RESET" "$IQ_MUTED" "$1" "$IQ_RESET"
}

iq_ok() {
  printf '  %b●%b %s\n' "$IQ_GREEN" "$IQ_RESET" "$1"
}

iq_fail() {
  printf '  %b●%b %s\n' "$IQ_RED" "$IQ_RESET" "$1" >&2
}

iq_note() {
  printf '  %b●%b %s\n' "$IQ_CYAN" "$IQ_RESET" "$1"
}

iq_find_root() {
  local candidate="${IQ_ROOT:-$PWD}"
  while [[ "$candidate" != / ]]; do
    if [[ ( -f "$candidate/launch_iqpilot.sh" || -f "$candidate/launch_openpilot.sh" ) && -d "$candidate/iqpilot" ]]; then
      IQ_ROOT="$candidate"
      return 0
    fi
    candidate="$(cd "$candidate/.." && pwd)"
  done
  for candidate in "$HOME/iqpilot" "$HOME/openpilot" /data/iqpilot /data/openpilot; do
    if [[ ( -f "$candidate/launch_iqpilot.sh" || -f "$candidate/launch_openpilot.sh" ) && -d "$candidate/iqpilot" ]]; then
      IQ_ROOT="$candidate"
      return 0
    fi
  done
  return 1
}

iq_require_root() {
  if ! iq_find_root; then
    iq_fail 'IQ.Pilot checkout not found. Run this inside the checkout or use --dir PATH.'
    return 1
  fi
}

iq_run() {
  local rendered
  printf -v rendered '%q ' "$@"
  rendered="${rendered% }"
  printf '%b›%b %b%s%b\n' "$IQ_PINK" "$IQ_RESET" "$IQ_MUTED" "$rendered" "$IQ_RESET"
  [[ "$IQ_DRY" = 1 ]] || "$@"
}

iq_check() {
  iq_title 'environment check'
  iq_require_root
  iq_ok "checkout  $IQ_ROOT"
  command -v git >/dev/null 2>&1 || { iq_fail 'git is not installed'; return 1; }
  iq_ok "git       $(git --version | sed 's/git version //')"
  command -v python3 >/dev/null 2>&1 || { iq_fail 'python3 is not installed'; return 1; }
  iq_ok "python    $(python3 --version | sed 's/Python //')"
  if [[ -x "$IQ_ROOT/.venv/bin/python3" ]]; then
    iq_ok 'venv      ready'
  else
    iq_note 'venv      not created yet — run iq setup'
  fi
}

iq_install() {
  local shell_name rc_file iq_script command
  shell_name="$(basename "${SHELL:-bash}")"
  rc_file="$HOME/.${shell_name}rc"
  [[ "$(uname)" = Darwin && "$shell_name" = bash ]] && rc_file="$HOME/.bash_profile"
  iq_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/iq.sh"
  command="alias iq='${iq_script} \"\$@\"'"
  touch "$rc_file"
  grep -Fqx "$command" "$rc_file" 2>/dev/null || printf '\n%s\n' "$command" >> "$rc_file"
  iq_title 'command installed'
  iq_ok "restart your shell, then use iq  ${rc_file}"
}

iq_setup() {
  local script
  iq_require_root
  if [[ -f /AGNOS ]]; then
    iq_title 'setup'
    iq_note 'IQ.OS manages system dependencies and the base Python runtime'
    iq_note 'use iq pkg to synchronize private packages when needed'
    iq_check
    return 0
  fi
  case "$OSTYPE" in
    linux-gnu*) script="$IQ_ROOT/iqpilot/tools/ubuntu_setup.sh" ;;
    darwin*) script="$IQ_ROOT/iqpilot/tools/mac_setup.sh" ;;
    *) iq_fail "unsupported platform: $OSTYPE"; return 1 ;;
  esac
  iq_title 'setup'
  iq_run "$script"
  if command -v git-lfs >/dev/null 2>&1; then
    iq_run git -C "$IQ_ROOT" lfs pull
  fi
  iq_ok 'setup complete'
}

iq_venv() {
  iq_require_root
  [[ -f "$IQ_ROOT/.venv/bin/activate" ]] || { iq_fail 'venv not found — run iq setup first'; return 1; }
  case "$(basename "${SHELL:-bash}")" in
    zsh) ZDOTDIR="$(mktemp -d)"; printf 'source %q\nsource %q\n' "$HOME/.zshrc" "$IQ_ROOT/.venv/bin/activate" > "$ZDOTDIR/.zshrc"; zsh ;;
    *) bash --rcfile <(printf 'source %q\nsource %q\n' "$HOME/.bashrc" "$IQ_ROOT/.venv/bin/activate") ;;
  esac
}

iq_build() {
  iq_require_root
  if [[ -f /AGNOS ]]; then
    iq_run "$IQ_ROOT/iqpilot/system/manager/build.py"
  else
    (cd "$IQ_ROOT" && iq_run scons "$@")
  fi
}

iq_quality() {
  iq_require_root
  (cd "$IQ_ROOT" && iq_run scripts/lint/lint.sh "$@")
}

iq_desktop_tool() {
  local name="$1" launcher="$2"
  shift 2
  iq_require_root
  if [[ ! -x "$IQ_ROOT/$launcher" ]]; then
    if [[ -f /AGNOS ]]; then
      iq_note "$name is not included in IQ.OS checkouts"
      return 0
    fi
    iq_fail "$name launcher is missing: $IQ_ROOT/$launcher"
    return 1
  fi
  iq_title "opening $name"
  (cd "$IQ_ROOT" && iq_run "$IQ_ROOT/$launcher" "$@")
}

iq_cabana() {
  iq_desktop_tool 'Cabana' 'iqpilot/tools/cabana/cabana' "$@"
}

iq_juggle() {
  iq_desktop_tool 'Jotpluggler' 'iqpilot/tools/jotpluggler/pluggle.py' "$@"
}

iq_brew_deps() {
  [[ "$(uname)" = Darwin ]] || return 0
  local brewfile="$IQ_ROOT/iqpilot/tools/Brewfile"
  [[ -f "$brewfile" ]] || return 0
  if ! command -v brew >/dev/null 2>&1; then
    iq_fail 'Homebrew is not installed; run iqpilot/tools/mac_setup.sh first'
    return 1
  fi
  local formulae status=0
  formulae="$(mktemp)"
  grep '^brew ' "$brewfile" > "$formulae"
  export HOMEBREW_NO_AUTO_UPDATE=1
  if ! brew bundle check --no-upgrade --file="$formulae" >/dev/null 2>&1; then
    iq_note 'installing missing Homebrew packages needed to build native packages'
    iq_run brew bundle install --no-upgrade --file="$formulae" || status=$?
  fi
  rm -f "$formulae"
  return "$status"
}

iq_pkg() {
  iq_require_root
  iq_brew_deps
  # the venv python has the component packages (iqdbc etc.); bare python3 does not
  local py=python3
  [[ -x "$IQ_ROOT/.venv/bin/python3" ]] && py="$IQ_ROOT/.venv/bin/python3"
  if [[ -f "$IQ_ROOT/iqpilot/tools/scripts/setup_private_packages.py" ]]; then
    (cd "$IQ_ROOT" && iq_run "$py" iqpilot/tools/scripts/setup_private_packages.py "$@")
  else
    iq_note 'private package sources are not present in this checkout'
  fi
  if [[ -f "$IQ_ROOT/artifacts/runtime/ensure_private_installed.sh" ]]; then
    (cd "$IQ_ROOT" && iq_run bash artifacts/runtime/ensure_private_installed.sh)
  fi
}

iq_update() {
  local fast=0
  iq_require_root
  if [[ $# -gt 0 ]]; then
    [[ $# = 1 && "$1" = f ]] || { iq_fail 'usage: iq update [f] [r]'; return 1; }
    fast=1
  fi
  iq_title 'updating IQ.Pilot'
  iq_run git -C "$IQ_ROOT" pull
  iq_pkg
  [[ "$fast" = 1 ]] && iq_fast_restart
}

iq_status() {
  local branch commit tree
  iq_require_root
  branch="$(git -C "$IQ_ROOT" branch --show-current 2>/dev/null || printf detached)"
  commit="$(git -C "$IQ_ROOT" rev-parse --short HEAD 2>/dev/null || printf unknown)"
  tree=clean
  [[ -n "$(git -C "$IQ_ROOT" status --porcelain 2>/dev/null)" ]] && tree=modified
  iq_title 'status'
  iq_note "root     $IQ_ROOT"
  iq_note "branch   $branch"
  iq_note "commit   $commit"
  [[ "$tree" = clean ]] && iq_ok "tree     $tree" || iq_note "tree     $tree"
}

iq_switch() {
  local remote=origin branch
  iq_require_root
  [[ $# -ge 1 ]] || { iq_fail 'usage: iq switch [REMOTE] BRANCH'; return 1; }
  [[ $# -ge 2 ]] && { remote="$1"; shift; }
  branch="$1"
  iq_title "switching to $remote/$branch"
  iq_note 'this discards uncommitted changes and untracked files'
  iq_run git -C "$IQ_ROOT" fetch "$remote" "$branch:refs/remotes/$remote/$branch"
  iq_run git -C "$IQ_ROOT" checkout -B "$branch" --track "$remote/$branch"
  iq_run git -C "$IQ_ROOT" reset --hard "$remote/$branch"
  iq_run git -C "$IQ_ROOT" clean -df
}

iq_service() {
  local action="$1"
  [[ -f /AGNOS ]] || { iq_note "${action} is available on IQ.OS devices only"; return 0; }
  iq_run sudo systemctl "$action" iq
}

iq_wait_for_tmux() {
  local expected="$1" attempt active
  [[ "$IQ_DRY" = 1 ]] && { iq_note "would verify IQ tmux session ${expected}"; return 0; }
  for ((attempt = 0; attempt < 15; attempt++)); do
    if tmux has-session -t iq 2>/dev/null; then active=running; else active=stopped; fi
    [[ "$active" = "$expected" ]] && return 0
    sleep 1
  done
  return 1
}

iq_fast_restart() {
  [[ -f /AGNOS || "$IQ_DRY" = 1 ]] || { iq_fail 'fast update restart is available on IQ.OS devices only'; return 1; }
  iq_title 'fast restarting IQ.Pilot'
  iq_run sudo systemctl stop iq
  if ! iq_wait_for_tmux stopped; then
    iq_fail 'IQ tmux session did not stop; attempt a reboot.'
    return 1
  fi
  iq_run sudo systemctl start iq
  if ! iq_wait_for_tmux running; then
    iq_fail 'IQ tmux session did not start; attempt a reboot.'
    return 1
  fi
  iq_ok 'IQ.Pilot restarted'
}

iq_help() {
  iq_title 'command center'
  printf '%bUsage%b  iq [--dir PATH] [--dry] COMMAND [ARGS] [r]\n\n' "$IQ_BOLD" "$IQ_RESET"
  printf '%bCOMMAND%b\n' "$IQ_PURPLE" "$IQ_RESET"
  printf '  %bsetup%b     install development dependencies\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bcheck%b     verify checkout, Git, Python, and venv\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bbuild%b     build IQ.Pilot\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bquality%b   run code-quality checks\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bcabana%b    open the CAN analysis tool\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bjuggle%b    open the log plotting tool\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bpkg%b       authenticate and synchronize private packages\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bupdate%b    pull IQ.Pilot, synchronize packages, optionally fast restart\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bstatus%b    show checkout, branch, commit, and tree state\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bswitch%b    replace the checkout with another branch\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bvenv%b      open a shell with the project venv active\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bstart%b     start IQ.Pilot on IQ.OS\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %bstop%b      stop IQ.Pilot on IQ.OS\n' "$IQ_CYAN" "$IQ_RESET"
  printf '  %binstall%b   install the iq shell command\n' "$IQ_CYAN" "$IQ_RESET"
  printf '\n  %br%b         reboot after a successful command\n' "$IQ_CYAN" "$IQ_RESET"
  printf '\n%bExamples%b\n' "$IQ_PURPLE" "$IQ_RESET"
  printf '  iq setup     iq update     iq update f     iq update f r\n'
  iq_line
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir) [[ $# -ge 2 ]] || { iq_fail 'missing path after --dir'; exit 2; }; IQ_ROOT="$2"; shift 2 ;;
    --dry) IQ_DRY=1; shift ;;
    -n|--no-verify) IQ_NO_VERIFY=1; shift ;;
    -l|--log) [[ $# -ge 2 ]] || { iq_fail 'missing file after --log'; exit 2; }; IQ_LOG_FILE="$2"; shift 2 ;;
    -h|--help|help) iq_help; exit 0 ;;
    *) break ;;
  esac
done

command="${1:-help}"
[[ $# -gt 0 ]] && shift
if [[ $# -gt 0 && "${!#}" = r ]]; then
  IQ_REBOOT=1
  set -- "${@:1:$#-1}"
fi
case "$command" in
  help) iq_help ;;
  setup) iq_setup "$@" ;;
  check) iq_check "$@" ;;
  build) iq_build "$@" ;;
  quality) iq_quality "$@" ;;
  cabana) iq_cabana "$@" ;;
  juggle) iq_juggle "$@" ;;
  pkg) iq_pkg "$@" ;;
  update) iq_update "$@" ;;
  status) iq_status "$@" ;;
  switch) iq_switch "$@" ;;
  venv) iq_venv "$@" ;;
  start) iq_service start "$@" ;;
  stop) iq_service stop "$@" ;;
  install) iq_install "$@" ;;
  *) iq_fail "unknown command: $command"; iq_help; exit 2 ;;
esac

if [[ "$IQ_REBOOT" = 1 ]]; then
  iq_title 'restarting IQ.OS'
  iq_run sudo reboot
fi
