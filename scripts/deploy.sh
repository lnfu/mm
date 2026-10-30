#!/usr/bin/env bash
#
#   scripts/deploy.sh [gpu|robot|all] [--build] [--dry-run]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/.env" ]; then
  echo "no $ROOT/.env: cp .env.example .env, then edit it" >&2
  exit 1
fi
# shellcheck source=SCRIPTDIR/../.env.example
source "$ROOT/.env"

GPU_PKGS=(mm_interfaces mm_perception)
ROBOT_PKGS=(mm_interfaces mm_ik mm_grasp)

usage() {
  echo "usage: $0 [gpu|robot|all] [--build] [--dry-run]" >&2
  exit 1
}

target=all
build=false
dry_run=()
for arg in "$@"; do
  case "$arg" in
    gpu|robot|all) target=$arg ;;
    --build) build=true ;;
    -n|--dry-run) dry_run=(--dry-run) ;;
    *) usage ;;
  esac
done

deploy() {
  local name=$1 host=$2
  shift 2
  local pkgs=("$@")

  echo "==> $name ($host): ${pkgs[*]}"
  for pkg in "${pkgs[@]}"; do
    rsync -az --delete --mkpath --itemize-changes "${dry_run[@]}" \
      --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
      "$ROOT/src/$pkg/" "$host:$REMOTE_WS/src/$pkg/"
  done

  if $build && [ ${#dry_run[@]} -eq 0 ]; then
    echo "==> $name: colcon build"
    # Non-interactive ssh does not read ~/.bashrc, so source ROS explicitly.
    # shellcheck disable=SC2029  # REMOTE_WS and pkgs are meant to expand locally
    ssh "$host" "cd ~/$REMOTE_WS \
      && source /opt/ros/jazzy/setup.bash \
      && if [ -f .venv/bin/activate ]; then source .venv/bin/activate; \
         else echo 'no ~/$REMOTE_WS/.venv, run scripts/setup_venv.sh first' >&2; exit 1; fi \
      && colcon build --packages-select ${pkgs[*]}"
  fi
}

case "$target" in
  gpu) deploy gpu "$GPU_HOST" "${GPU_PKGS[@]}" ;;
  robot) deploy robot "$ROBOT_HOST" "${ROBOT_PKGS[@]}" ;;
  all)
    deploy gpu "$GPU_HOST" "${GPU_PKGS[@]}"
    deploy robot "$ROBOT_HOST" "${ROBOT_PKGS[@]}"
    ;;
esac
