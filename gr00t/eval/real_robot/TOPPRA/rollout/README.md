# Unified real-robot rollout

This is the single laptop-side entrypoint for synchronous/asynchronous TOPPRA and the independent
Speed-RL extension. The frozen rollout and Kion client remain unchanged. The rollout process is the
only 250 Hz owner of dual-arm Servo and hand commands; ObservationGUILite observes at 30 Hz and
labels/records episodes without driving the robot.

## Fresh-laptop deployment

```bash
git clone <YOUR_REPOSITORY_URL> Isaac-GR00T
git clone <OBSERVATION_GUI_REPOSITORY_URL> ObservationGUILite-v1.0.0
cd Isaac-GR00T
uv sync --all-extras
cp gr00t/eval/real_robot/TOPPRA/rollout/rollout.env.example rollout.local.env
```

Edit `rollout.local.env` once:

- set `ROS_SETUP` and `ROBOT_ROS_SETUP` so the selected Python can import ROS and Kion messages;
- keep the SSH tunnel endpoint as `POLICY_SERVER_HOST=127.0.0.1` and port `47866`;
- check all three camera topics against the robot;
- keep `TTS_SAMPLES=1` for normal synchronous rollout;
- keep `DRY_RUN=1` until observation, server, TOPPRA and labels have been checked.

If the server is reached through SSH, run this in a separate laptop terminal:

```bash
ssh -N -L 47866:127.0.0.1:47866 -p 20243 chenlu@47.97.38.144
```

Then perform the dry-run preflight:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/robot_ws/devel/setup.bash
.venv/bin/python -c \
  "from gr00t.eval.real_robot.TOPPRA.kion_client.client import load_ros_types; print(load_ros_types())"
.venv/bin/python -m gr00t.eval.real_robot.TOPPRA.speed_rl.probe_server \
  --server-host 127.0.0.1 --server-port 47866 --requests 10 --warmup-requests 2
bash gr00t/eval/real_robot/TOPPRA/rollout/launch.sh rollout.local.env
```

The plain client only needs action output. The Speed-RL probe additionally requires the exact
`last_block_pre_norm` feature contract from the updated server.

## Rollout modes

Start with these values in `rollout.local.env`:

```text
ROLLOUT_MODE=plain
INFERENCE_MODE=sync
TTS_SAMPLES=1
DRY_RUN=1
```

Synchronous rollout has one candidate and does not use multi-candidate TTS. For raw asynchronous
testing change only `INFERENCE_MODE=async`; it continues to use the frozen handoff/refill logic.
Set `MAX_LATENCY_S` at or above the measured end-to-end P99 before an async hardware run. TTS values
of 4/8 deliberately multiply candidate computation and payload size and are not normal rollout
defaults.

For Speed-RL use:

```text
ROLLOUT_MODE=speed-rl
INFERENCE_MODE=sync
SPEED_RL_PHASE=calibration
LEFT_TWIST_THRESHOLDS=VX,VY,VZ,WX,WY,WZ
RIGHT_TWIST_THRESHOLDS=VX,VY,VZ,WX,WY,WZ
```

The measured-twist thresholds must come from hardware certification; the code has no guessed
defaults. Full phase and async gates are documented in `../speed_rl/README.md`.

## ObservationGUILite and episode labels

Keep the rollout client running with `CONTROL_INTERFACE=gui`, then launch:

```bash
export ROS_MASTER_URI=http://ROBOT_IP:11311
export ROS_IP=LAPTOP_ROBOT_NETWORK_IP
bash gr00t/eval/real_robot/TOPPRA/rollout/observation_gui/deploy.sh \
  /home/xukainan/ObservationGUILite-v1.0.0
```

The panel controls `/gr00t_rollout/{start,success,failure,abort,approve}` and displays the latched
status. `success`/`failure` stop consuming trajectories and retain the last target; `abort` closes
Servo. The GUI automatically begins passive dataset recording on `start`, saves success/failure
episodes with an external outcome sidecar, and discards aborted episodes.

Persistent outputs are:

```text
logs/kion_rollout/<episode>/metadata.json, tcp_tracking.csv, outcome.json
logs/kion_speed_rl/episodes/<episode>/metadata.json, tcp_tracking.csv, outcome.json
logs/kion_speed_rl/state/                 # replay, checkpoint and phase state
logs/observation_gui_rollout/app/datasets/parcel_rollout_view/
```

Do not delete the Speed-RL state directory between launches. A `finalization_error` blocks the next
episode so a failed checkpoint or label write cannot be silently skipped.
