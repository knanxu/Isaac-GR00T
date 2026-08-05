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
if [[ -n "${ROBOT_ROS_SETUP:-}" ]]; then
    if [[ ! -f "$ROBOT_ROS_SETUP" ]]; then
        echo "Robot ROS setup file does not exist: $ROBOT_ROS_SETUP" >&2
        exit 2
    fi
    source "$ROBOT_ROS_SETUP"
fi

command=(
    "$python_bin" -m gr00t.eval.real_robot.TOPPRA.speed_rl
    --server-host "$POLICY_SERVER_HOST"
    --server-port "${POLICY_SERVER_PORT:-47866}"
    --timeout-ms "${TIMEOUT_MS:-15000}"
    --task "${TASK:-move parcel onto conveyor belt one by one}"
    --phase "${PHASE:-calibration}"
    --inference-mode "${INFERENCE_MODE:-sync}"
    --state-root "${STATE_ROOT:-logs/kion_speed_rl/state}"
    --log-root "${LOG_ROOT:-logs/kion_speed_rl/episodes}"
    --control-frequency "${CONTROL_FREQUENCY:-250}"
    --policy-frequency "${POLICY_FREQUENCY:-30}"
    --refill-threshold "${REFILL_THRESHOLD:-20}"
    --max-latency-s "${MAX_LATENCY_S:-0.35}"
    --scheduling-margin-s "${SCHEDULING_MARGIN_S:-0.05}"
    --handoff-margin-s "${HANDOFF_MARGIN_S:-0.05}"
    --open-loop-horizon "${OPEN_LOOP_HORIZON:-8}"
    --tts-samples "${TTS_SAMPLES:-1}"
    --tts-waypoint-count "${TTS_WAYPOINT_COUNT:-5}"
    --velocity-filter "${VELOCITY_FILTER:-1.0}"
    --servo-gain "${SERVO_GAIN:-800}"
    --startup-timeout-s "${STARTUP_TIMEOUT_S:-30}"
    --max-state-age-s "${MAX_STATE_AGE_S:-0.25}"
    --max-image-age-s "${MAX_IMAGE_AGE_S:-1.0}"
    --episode-duration-s "${EPISODE_DURATION_S:-80}"
    --control-interface "${CONTROL_INTERFACE:-gui}"
    --ros-namespace "${ROS_NAMESPACE:-/gr00t_rollout}"
    --pinch-max-rate-hz "${PINCH_MAX_RATE_HZ:-30}"
    --left-camera-topic "${LEFT_CAMERA_TOPIC:-/zj_humanoid/sensor/left_wrist/image_raw/compressed}"
    --right-camera-topic "${RIGHT_CAMERA_TOPIC:-/zj_humanoid/sensor/right_wrist/image_raw/compressed}"
    --head-camera-topic "${HEAD_CAMERA_TOPIC:-/zj_humanoid/sensor/realsense_head/color/image_raw/compressed}"
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
if [[ -n "${CHECKPOINT:-}" ]]; then
    command+=(--checkpoint "$CHECKPOINT")
fi

exec "${command[@]}"
