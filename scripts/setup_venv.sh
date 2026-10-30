#!/usr/bin/env bash
#
#   scripts/setup_venv.sh [gpu|robot|all]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/.env" ]; then
  echo "no $ROOT/.env: cp .env.example .env, then edit it" >&2
  exit 1
fi
# shellcheck source=SCRIPTDIR/../.env.example
source "$ROOT/.env"

GPU_DEPS=(ultralytics "numpy<2")
ROBOT_DEPS=(pin pin-pink daqp "numpy<2")

GPU_CHECK='import numpy, cv2, torch, ultralytics, rclpy
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| ultralytics", ultralytics.__version__)'
ROBOT_CHECK='import numpy, cv2, pinocchio, pink, daqp, rclpy
print("numpy", numpy.__version__, "| pinocchio", pinocchio.__version__, "| pink", pink.__version__)'

# Runs on the remote. Args: <workspace dir> <python check> <packages...>
read -r -d '' REMOTE_SCRIPT <<'EOF' || true
set -euo pipefail
ws=$1 check=$2
shift 2
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
  echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
mkdir -p ~/"$ws"
cd ~/"$ws"
if [ -d .venv ]; then
  echo "reusing ~/$ws/.venv"
else
  # --system-site-packages keeps rclpy and the other apt packages visible.
  uv venv --python /usr/bin/python3 --system-site-packages .venv
fi
touch .venv/COLCON_IGNORE
set +u  # ROS setup scripts reference unset variables
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
set -u
uv pip install "$@"
python -c "$check"
EOF

setup() {
  local name=$1 host=$2 check=$3
  shift 3
  echo "==> $name ($host): $*"
  # shellcheck disable=SC2029  # intentional: the quoted args expand locally
  ssh "$host" "bash -c $(printf '%q' "$REMOTE_SCRIPT") _ $(printf '%q ' "$REMOTE_WS" "$check" "$@")"
}

target=${1:-all}
case "$target" in
  gpu) setup gpu "$GPU_HOST" "$GPU_CHECK" "${GPU_DEPS[@]}" ;;
  robot) setup robot "$ROBOT_HOST" "$ROBOT_CHECK" "${ROBOT_DEPS[@]}" ;;
  all)
    setup gpu "$GPU_HOST" "$GPU_CHECK" "${GPU_DEPS[@]}"
    setup robot "$ROBOT_HOST" "$ROBOT_CHECK" "${ROBOT_DEPS[@]}"
    ;;
  *) echo "usage: $0 [gpu|robot|all]" >&2; exit 1 ;;
esac
