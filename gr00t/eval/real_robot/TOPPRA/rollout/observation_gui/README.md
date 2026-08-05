# ObservationGUILite rollout view

This overlay reuses the existing ObservationGUILite Docker deployment without allowing its 30 Hz
control loop to publish robot actions. The Isaac-GR00T rollout process remains the only owner of the
250 Hz dual-arm Servo and hand commands.

Start a plain or Speed-RL client with `--control-interface gui`, then run:

```bash
export ROS_MASTER_URI=http://ROBOT_IP:11311
export ROS_IP=LAPTOP_ROBOT_NETWORK_IP
bash gr00t/eval/real_robot/TOPPRA/rollout/observation_gui/deploy.sh \
  /home/xukainan/ObservationGUILite-v1.0.0
```

The overlay is assembled under `logs/observation_gui_rollout/app`; it does not modify the supplied
ObservationGUILite checkout. That directory is mounted into Docker, so recorded datasets survive a
container exit. Override its parent with `OBSERVATION_GUI_RUNTIME_ROOT` if required.

The GUI exposes Start, Success, Failure, Abort and Speed Calibration Approval buttons and displays
trajectory, inference, replay and safety status from `/gr00t_rollout/status`. On Start it enters
passive recording. Success/failure saves the episode under
`logs/observation_gui_rollout/app/datasets/parcel_rollout_view` with the external outcome in the
debug-event JSON sidecar; abort discards it. Use the fixed `/gr00t_rollout` namespace for this first
version.
