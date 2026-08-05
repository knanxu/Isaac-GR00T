#!/usr/bin/env bash
set -Eeuo pipefail

readonly INTEGRATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly GUI_ROOT="${1:-${OBSERVATION_GUI_ROOT:-/home/xukainan/ObservationGUILite-v1.0.0}}"
runtime_root="${OBSERVATION_GUI_RUNTIME_ROOT:-logs/observation_gui_rollout}"

if [[ ! -x "${GUI_ROOT}/deploy.sh" ]]; then
    echo "ObservationGUILite deploy.sh is unavailable: ${GUI_ROOT}/deploy.sh" >&2
    exit 2
fi

mkdir -p "${runtime_root}/app"
readonly RUNTIME_ROOT="$(cd -- "${runtime_root}" && pwd)"
readonly overlay_dir="${RUNTIME_ROOT}/app"

for entry in \
    .dockerignore \
    FiraCode-SemiBold.ttf \
    agents \
    build \
    controlloop.py \
    deploy.sh \
    docker \
    gui \
    observations \
    pyproject.toml \
    utils \
    uv.lock; do
    cp -a --reflink=auto "${GUI_ROOT}/${entry}" "${overlay_dir}/"
done
cp "${INTEGRATION_DIR}/main.py" "${overlay_dir}/main.py"
cp "${INTEGRATION_DIR}/rollout_passive.py" "${overlay_dir}/agents/rollout_passive.py"
cp "${INTEGRATION_DIR}/rollout_control.py" "${overlay_dir}/gui/rollout_control.py"

echo "Launching ObservationGUILite in read-only external-rollout mode"
echo "ROS namespace: /gr00t_rollout"
echo "Persistent dataset: ${overlay_dir}/datasets/parcel_rollout_view"
export IMAGE_NAME="${IMAGE_NAME:-gr00t-rollout-gui:latest}"
export CONTAINER_NAME="${CONTAINER_NAME:-gr00t-rollout-gui}"
bash "${overlay_dir}/deploy.sh"
