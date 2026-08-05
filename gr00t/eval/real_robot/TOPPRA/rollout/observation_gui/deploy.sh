#!/usr/bin/env bash
set -Eeuo pipefail

readonly INTEGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly GUI_ROOT="${1:-${OBSERVATION_GUI_ROOT:-/home/xukainan/ObservationGUILite-v1.0.0}}"

if [[ ! -x "${GUI_ROOT}/deploy.sh" ]]; then
    echo "ObservationGUILite deploy.sh is unavailable: ${GUI_ROOT}/deploy.sh" >&2
    exit 2
fi

overlay_dir="$(mktemp -d -t gr00t-rollout-gui.XXXXXX)"
cleanup() {
    rm -rf -- "${overlay_dir}"
}
trap cleanup EXIT

cp -a --reflink=auto "${GUI_ROOT}/." "${overlay_dir}/"
cp "${INTEGRATION_DIR}/main.py" "${overlay_dir}/main.py"
cp "${INTEGRATION_DIR}/rollout_passive.py" "${overlay_dir}/agents/rollout_passive.py"
cp "${INTEGRATION_DIR}/rollout_control.py" "${overlay_dir}/gui/rollout_control.py"

echo "Launching ObservationGUILite in read-only external-rollout mode"
echo "ROS namespace: ${GR00T_ROLLOUT_NAMESPACE:-/gr00t_rollout}"
bash "${overlay_dir}/deploy.sh"
