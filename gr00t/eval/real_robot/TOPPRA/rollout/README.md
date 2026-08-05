# Unified real-robot rollout

This package is the deployment entrypoint for:

- plain TOPPRA with synchronous or asynchronous GR00T inference;
- TOPPRA plus the four-level Speed-RL policy;
- terminal or ObservationGUILite episode control;
- one 250 Hz owner for dual-arm pose, pinch and tracking logs.

The frozen rollout implementation and original Kion client remain available, but new real-robot
episodes should use this entrypoint so start/success/failure/abort have identical semantics.

## Configure and launch

```bash
cp gr00t/eval/real_robot/TOPPRA/rollout/rollout.env.example rollout.local.env
# Edit server, ROS setup and hardware values.
bash gr00t/eval/real_robot/TOPPRA/rollout/launch.sh rollout.local.env
```

Start with `ROLLOUT_MODE=plain`, `INFERENCE_MODE=sync`, `TTS_SAMPLES=1` and `DRY_RUN=1`.
For Speed-RL set `ROLLOUT_MODE=speed-rl`; certified left/right twist thresholds are then mandatory.

Direct CLI equivalents are:

```bash
python -m gr00t.eval.real_robot.TOPPRA.rollout plain --inference-mode sync --dry-run
python -m gr00t.eval.real_robot.TOPPRA.rollout plain --inference-mode async --dry-run
python -m gr00t.eval.real_robot.TOPPRA.rollout speed-rl \
  --inference-mode sync \
  --left-twist-thresholds VX,VY,VZ,WX,WY,WZ \
  --right-twist-thresholds VX,VY,VZ,WX,WY,WZ \
  --dry-run
```

## ObservationGUILite

With the client running using `CONTROL_INTERFACE=gui`, launch the read-only GUI overlay:

```bash
bash gr00t/eval/real_robot/TOPPRA/rollout/observation_gui/deploy.sh \
  /home/xukainan/ObservationGUILite-v1.0.0
```

The GUI controls `/gr00t_rollout/{start,success,failure,abort,approve}` and displays the latched
`/gr00t_rollout/status` topic. Success and failure stop trajectory consumption and keep publishing
the last arm target. Abort clears Servo. The GUI never publishes rollout arm or hand commands.

Each plain episode writes `tcp_tracking.csv`, `metadata.json` and `outcome.json`. Speed-RL keeps its
existing replay, checkpoint, calibration and online-training state in its configured state root.
