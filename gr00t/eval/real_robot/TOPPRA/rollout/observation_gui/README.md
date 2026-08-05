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

The overlay is assembled in a temporary directory. It does not modify the supplied
ObservationGUILite checkout. The GUI exposes Start, Success, Failure, Abort and Speed Calibration
Approval buttons and displays trajectory, inference, Speed-RL and safety status from
`/gr00t_rollout/status`.
