#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: bash gr00t/eval/real_robot/TOPPRA/rollout/launch.sh CONFIG.local.env" >&2
    exit 2
fi

config_path="$1"
if [[ ! -f "${config_path}" ]]; then
    echo "configuration file does not exist: ${config_path}" >&2
    exit 2
fi

set -a
source "${config_path}"
set +a

: "${POLICY_SERVER_HOST:?set POLICY_SERVER_HOST in the rollout configuration}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable is unavailable: ${python_bin} (run uv sync --all-extras)" >&2
    exit 2
fi
if [[ -n "${ROS_SETUP:-}" ]]; then
    if [[ ! -f "${ROS_SETUP}" ]]; then
        echo "ROS setup file does not exist: ${ROS_SETUP}" >&2
        exit 2
    fi
    source "${ROS_SETUP}"
fi

mode="${ROLLOUT_MODE:-plain}"
if [[ "${mode}" != "plain" && "${mode}" != "speed-rl" ]]; then
    echo "ROLLOUT_MODE must be plain or speed-rl, got: ${mode}" >&2
    exit 2
fi

command=(
    "${python_bin}" -m gr00t.eval.real_robot.TOPPRA.rollout "${mode}"
    --server-host "${POLICY_SERVER_HOST}"
    --server-port "${POLICY_SERVER_PORT:-47866}"
    --task "${TASK:-move parcel onto conveyor belt one by one}"
    --inference-mode "${INFERENCE_MODE:-sync}"
    --policy-frequency "${POLICY_FREQUENCY:-30}"
    --control-frequency "${CONTROL_FREQUENCY:-250}"
    --tts-samples "${TTS_SAMPLES:-1}"
    --servo-gain "${SERVO_GAIN:-800}"
    --episode-duration-s "${EPISODE_DURATION_S:-80}"
    --control-interface "${CONTROL_INTERFACE:-gui}"
    --ros-namespace "${ROS_NAMESPACE:-/gr00t_rollout}"
)

if [[ "${mode}" == "plain" ]]; then
    command+=(
        --max-linear-velocity "${MAX_LINEAR_VELOCITY:-1.0}"
        --max-angular-velocity "${MAX_ANGULAR_VELOCITY:-3.0}"
        --max-linear-acceleration "${MAX_LINEAR_ACCELERATION:-5.0}"
        --max-angular-acceleration "${MAX_ANGULAR_ACCELERATION:-15.0}"
        --safety-margin "${SAFETY_MARGIN:-0.9}"
        --log-root "${PLAIN_LOG_ROOT:-logs/kion_rollout}"
    )
else
    : "${LEFT_TWIST_THRESHOLDS:?set certified LEFT_TWIST_THRESHOLDS for Speed-RL}"
    : "${RIGHT_TWIST_THRESHOLDS:?set certified RIGHT_TWIST_THRESHOLDS for Speed-RL}"
    command+=(
        --phase "${SPEED_RL_PHASE:-calibration}"
        --left-twist-thresholds "${LEFT_TWIST_THRESHOLDS}"
        --right-twist-thresholds "${RIGHT_TWIST_THRESHOLDS}"
        --state-root "${SPEED_RL_STATE_ROOT:-logs/kion_speed_rl/state}"
        --log-root "${SPEED_RL_LOG_ROOT:-logs/kion_speed_rl/episodes}"
    )
    if [[ -n "${SPEED_RL_CHECKPOINT:-}" ]]; then
        command+=(--checkpoint "${SPEED_RL_CHECKPOINT}")
    fi
    if [[ -n "${ASYNC_VERIFICATION:-}" ]]; then
        command+=(--async-verification "${ASYNC_VERIFICATION}")
    fi
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    command+=(--dry-run)
fi
if [[ "${DISABLE_PINCH:-0}" == "1" ]]; then
    command+=(--disable-pinch)
fi

exec "${command[@]}"
