#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash gr00t/eval/real_robot/TOPPRA/speed_rl/launch_client.sh CONFIG.local.env" >&2
    exit 2
fi

config_path=$1
if [[ ! -f "$config_path" ]]; then
    echo "configuration file does not exist: $config_path" >&2
    exit 2
fi

set -a
source "$config_path"
set +a

: "${POLICY_SERVER_HOST:?set POLICY_SERVER_HOST in the client configuration}"
: "${LEFT_TWIST_THRESHOLDS:?set certified LEFT_TWIST_THRESHOLDS in the client configuration}"
: "${RIGHT_TWIST_THRESHOLDS:?set certified RIGHT_TWIST_THRESHOLDS in the client configuration}"

python_bin=${PYTHON_BIN:-.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
    echo "Python executable is unavailable: $python_bin (run uv sync --all-extras)" >&2
    exit 2
fi

if [[ -n "${ROS_SETUP:-}" ]]; then
    if [[ ! -f "$ROS_SETUP" ]]; then
        echo "ROS setup file does not exist: $ROS_SETUP" >&2
        exit 2
    fi
    source "$ROS_SETUP"
fi

command=(
    "$python_bin" -m gr00t.eval.real_robot.TOPPRA.speed_rl
    --server-host "$POLICY_SERVER_HOST"
    --server-port "${POLICY_SERVER_PORT:-47866}"
    --task "${TASK:-move parcel onto conveyor belt one by one}"
    --phase "${PHASE:-calibration}"
    --inference-mode "${INFERENCE_MODE:-sync}"
    --state-root "${STATE_ROOT:-logs/kion_speed_rl/state}"
    --log-root "${LOG_ROOT:-logs/kion_speed_rl/episodes}"
    --control-frequency "${CONTROL_FREQUENCY:-250}"
    --policy-frequency "${POLICY_FREQUENCY:-30}"
    --tts-samples "${TTS_SAMPLES:-1}"
    --servo-gain "${SERVO_GAIN:-800}"
    --control-interface "${CONTROL_INTERFACE:-gui}"
    --ros-namespace "${ROS_NAMESPACE:-/gr00t_rollout}"
    --left-twist-thresholds "$LEFT_TWIST_THRESHOLDS"
    --right-twist-thresholds "$RIGHT_TWIST_THRESHOLDS"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    command+=(--dry-run)
fi
if [[ "${DISABLE_PINCH:-0}" == "1" ]]; then
    command+=(--disable-pinch)
fi
if [[ -n "${ASYNC_VERIFICATION:-}" ]]; then
    command+=(--async-verification "$ASYNC_VERIFICATION")
fi

exec "${command[@]}"
