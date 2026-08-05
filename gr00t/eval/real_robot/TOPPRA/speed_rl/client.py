from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from enum import Enum
import json
import logging
import math
from pathlib import Path
from queue import Empty, Queue
import shlex
import threading
import time
from typing import Any

import numpy as np

from ..eval_toppra_bimanual import DEFAULT_TASK, CartesianLimits, GR00TPolicyClient
from ..kion_client.client import (
    LEFT_TARGET_KEY,
    LEFT_TWIST_KEY,
    RIGHT_TARGET_KEY,
    RIGHT_TWIST_KEY,
    KionDualArmServo,
    KionObservationBuffer,
    KionTopics,
    RosTypes,
    load_ros_types,
)
from ..kion_client.tracking import TrackingRecorder, make_tracking_row
from ..rollout.operator import RosEpisodeBridge
from ..rollout.pinch import KionPinchExecutor
from .agent import SpeedRLAgent
from .config import RainbowConfig
from .contract import FeatureContract
from .learner import LearnerProcess, SpeedActor
from .phases import CalibrationManager, EpisodeOutcome, GreedyAcceptance, OnlineEpisodeBudget
from .replay import Transition
from .safety import SpeedViolationMonitor, parse_twist_thresholds


LOGGER = logging.getLogger(__name__)
LEFT_PINCH_KEY = "action.left_pinch"
RIGHT_PINCH_KEY = "action.right_pinch"


class EpisodeState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    TERMINATED = "terminated"
    ABORTED = "aborted"
    CLOSED = "closed"


class SpeedRLKionClient:
    """Episode-driven 250 Hz Speed-RL client with terminal and ROS GUI control."""

    def __init__(
        self,
        *,
        ros_types: RosTypes,
        topics: KionTopics,
        agent: SpeedRLAgent,
        actor: SpeedActor,
        learner: LearnerProcess,
        safety: SpeedViolationMonitor,
        calibration: CalibrationManager,
        phase: str,
        checkpoint_path: Path,
        log_root: Path,
        control_frequency: float,
        servo_gain: int,
        startup_timeout_s: float,
        max_state_age_s: float,
        max_image_age_s: float,
        task: str,
        dry_run: bool,
        episode_duration_s: float = 80.0,
        control_interface: str = "gui",
        ros_namespace: str = "/gr00t_rollout",
        pinch_enabled: bool = True,
        pinch_max_rate_hz: float = 30.0,
    ) -> None:
        self.ros_types = ros_types
        self.rospy = ros_types.rospy
        self.topics = topics
        self.agent = agent
        self.actor = actor
        self.learner = learner
        self.safety = safety
        self.calibration = calibration
        self.phase = phase
        self.checkpoint_path = checkpoint_path
        self.log_root = log_root
        self.control_frequency = float(control_frequency)
        self.control_period_ns = int(round(1e9 / self.control_frequency))
        self.servo_gain = int(servo_gain)
        self.startup_timeout_s = float(startup_timeout_s)
        self.max_state_age_s = float(max_state_age_s)
        self.max_image_age_s = float(max_image_age_s)
        self.task = task
        self.dry_run = bool(dry_run)
        self.episode_duration_s = float(episode_duration_s)
        self.control_interface = control_interface
        self.ros_namespace = ros_namespace
        self.pinch_enabled = bool(pinch_enabled)
        self.pinch_max_rate_hz = float(pinch_max_rate_hz)
        self.observations = KionObservationBuffer(ros_types, topics)
        self.servo: KionDualArmServo | None = None
        self.pinch: KionPinchExecutor | None = None
        self.recorder: TrackingRecorder | None = None
        self.state = EpisodeState.IDLE
        self.episode_number = 0
        self._episode_started_s: float | None = None
        self._last_target: dict[str, np.ndarray] | None = None
        self._sample_index = 0
        self._commands: Queue[str] = Queue()
        self._shutdown = threading.Event()
        self._command_thread: threading.Thread | None = None
        self._ros_bridge: RosEpisodeBridge | None = None
        self._finalizer: threading.Thread | None = None
        self._finalization_error: str | None = None
        self._last_outcome: str | None = None
        self._tracking_log: str | None = None
        self._last_command_error: str | None = None
        self._online_budget = OnlineEpisodeBudget(
            22,
            calibration.state_path.parent / "online_budget.json",
        )
        self._greedy_acceptance = GreedyAcceptance()

    def run(self) -> None:
        try:
            self._run_control_loop()
        except BaseException:
            if self.state == EpisodeState.RUNNING:
                try:
                    self._finish_episode(
                        success=False,
                        reason="control-loop fault",
                        control_fault=True,
                    )
                except BaseException:
                    LOGGER.exception("Failed to persist the control-fault episode")
            raise
        finally:
            self.close()

    def _run_control_loop(self) -> None:
        self.observations.wait_until_ready(self.startup_timeout_s)
        self._start_control_interfaces()
        LOGGER.info(
            "Speed-RL client ready: interface=%s namespace=%s",
            self.control_interface,
            self.ros_namespace,
        )
        next_tick_ns = time.monotonic_ns()
        last_deadline_warning_ns = 0
        while not self.rospy.is_shutdown() and not self._shutdown.is_set():
            loop_started_ns = time.monotonic_ns()
            snapshot = self.observations.snapshot(
                max_state_age_s=self.max_state_age_s,
                max_image_age_s=self.max_image_age_s,
            )
            twist_stale = LEFT_TWIST_KEY in snapshot.stale_fields or RIGHT_TWIST_KEY in (
                snapshot.stale_fields
            )
            safety_status = self.safety.update(
                snapshot.left_twist,
                snapshot.right_twist,
                stale=twist_stale,
            )
            self.agent.set_action_mask(safety_status.action_mask)
            self._drain_commands()

            if (
                self.state == EpisodeState.RUNNING
                and self._episode_started_s is not None
                and time.monotonic() - self._episode_started_s >= self.episode_duration_s
            ):
                self._finish_episode(success=False, reason="80-second timeout")

            target_left: np.ndarray | None = None
            target_right: np.ndarray | None = None
            diagnostics: dict[str, Any] = {
                "execution_state": self.state.value,
                "active_sequence_id": None,
                "active_buffer_size": 0,
            }
            if self.state == EpisodeState.RUNNING:
                if snapshot.stale_fields:
                    target_left, target_right = self._hold_target(snapshot)
                    diagnostics["execution_state"] = "state_stale_hold"
                else:
                    action = self.agent.act(snapshot.observation, self.task)
                    target_left = np.asarray(action[LEFT_TARGET_KEY], dtype=np.float64)
                    target_right = np.asarray(action[RIGHT_TARGET_KEY], dtype=np.float64)
                    self._last_target = {
                        "left": target_left.copy(),
                        "right": target_right.copy(),
                    }
                    if self.pinch is not None:
                        self.pinch.publish(action[LEFT_PINCH_KEY], action[RIGHT_PINCH_KEY])
                    diagnostics = self.agent.diagnostics()
            elif self.state == EpisodeState.TERMINATED and self._last_target is not None:
                target_left = self._last_target["left"]
                target_right = self._last_target["right"]

            if self.servo is not None and target_left is not None and target_right is not None:
                published_monotonic_ns, published_ros_ns = self.servo.publish(
                    target_left,
                    target_right,
                )
                if self.recorder is not None:
                    self.recorder.record(
                        make_tracking_row(
                            sample_index=self._sample_index,
                            loop_started_monotonic_ns=loop_started_ns,
                            published_monotonic_ns=published_monotonic_ns,
                            published_ros_ns=published_ros_ns,
                            loop_duration_ns=time.monotonic_ns() - loop_started_ns,
                            deadline_lateness_ns=max(loop_started_ns - next_tick_ns, 0),
                            active_sequence_id=diagnostics.get("active_sequence_id"),
                            execution_state=str(
                                diagnostics.get("execution_state", self.state.value)
                            ),
                            active_buffer_size=int(diagnostics.get("active_buffer_size") or 0),
                            left_pose_ros_ns=snapshot.left_pose_ros_ns,
                            left_pose_received_monotonic_ns=(
                                snapshot.left_pose_received_monotonic_ns
                            ),
                            right_pose_ros_ns=snapshot.right_pose_ros_ns,
                            right_pose_received_monotonic_ns=(
                                snapshot.right_pose_received_monotonic_ns
                            ),
                            twist_ros_ns=snapshot.twist_ros_ns,
                            twist_received_monotonic_ns=(snapshot.twist_received_monotonic_ns),
                            target_left=target_left,
                            measured_left=snapshot.left_pose,
                            measured_left_twist=snapshot.left_twist,
                            target_right=target_right,
                            measured_right=snapshot.right_pose,
                            measured_right_twist=snapshot.right_twist,
                        )
                    )
                    self._sample_index += 1

            if self._ros_bridge is not None:
                self._ros_bridge.publish_status()

            next_tick_ns += self.control_period_ns
            remaining_ns = next_tick_ns - time.monotonic_ns()
            if remaining_ns > 0:
                time.sleep(remaining_ns * 1e-9)
            else:
                now_ns = time.monotonic_ns()
                if now_ns - last_deadline_warning_ns >= 1_000_000_000:
                    LOGGER.warning(
                        "Control deadline missed by %.3f ms; samples are not caught up",
                        -remaining_ns * 1e-6,
                    )
                    last_deadline_warning_ns = now_ns
                next_tick_ns = now_ns + self.control_period_ns

    def _hold_target(self, snapshot: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._last_target is None:
            self._last_target = {
                "left": snapshot.left_pose.copy(),
                "right": snapshot.right_pose.copy(),
            }
        return self._last_target["left"], self._last_target["right"]

    def enqueue_command(self, command: str) -> None:
        self._commands.put(command)

    def _start_control_interfaces(self) -> None:
        if self.control_interface in {"terminal", "both"}:
            self._command_thread = threading.Thread(
                target=self._terminal_loop,
                daemon=True,
                name="SpeedRLTerminal",
            )
            self._command_thread.start()
        if self.control_interface in {"gui", "both"}:
            self._ros_bridge = RosEpisodeBridge(
                self.rospy,
                self.enqueue_command,
                self.status,
                namespace=self.ros_namespace,
            )
            self._ros_bridge.publish_status(force=True)

    def _terminal_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = input("speed-rl> ")
            except EOFError:
                self.enqueue_command("quit")
                return
            self.enqueue_command(line)

    def _drain_commands(self) -> None:
        while True:
            try:
                line = self._commands.get_nowait()
            except Empty:
                return
            try:
                self.handle_command(line)
                self._last_command_error = None
            except Exception as exc:
                self._last_command_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Speed-RL command failed: %s", line)

    def handle_command(self, line: str) -> dict[str, Any] | None:
        parts = shlex.split(line)
        if not parts:
            return None
        command = parts[0].lower()
        if command == "start":
            self._start_episode()
            return self.status()
        if command == "success":
            self._finish_episode(success=True, reason="operator success")
            return self.status()
        if command == "failure":
            self._finish_episode(success=False, reason="operator failure")
            return self.status()
        if command == "abort":
            self._abort_episode()
            return self.status()
        if command == "approve":
            if len(parts) < 2:
                raise ValueError("approve requires a tracking review note or report path")
            self.calibration.approve_current_speed(" ".join(parts[1:]))
            return self.status()
        if command == "status":
            status = self.status()
            LOGGER.info("Speed-RL status: %s", json.dumps(status, sort_keys=True))
            return status
        if command == "quit":
            if self.state == EpisodeState.RUNNING:
                raise RuntimeError("quit is refused while an episode is RUNNING; abort first")
            self._shutdown.set()
            return self.status()
        raise ValueError(f"Unknown Speed-RL command {command!r}")

    def _start_episode(self) -> None:
        if self.state == EpisodeState.RUNNING:
            raise RuntimeError("An episode is already running")
        if self._finalizer is not None and self._finalizer.is_alive():
            raise RuntimeError("The previous episode is still finalizing")
        if self._finalization_error is not None:
            raise RuntimeError(
                f"The previous episode failed to finalize: {self._finalization_error}"
            )
        if self.phase == "calibration":
            if self.calibration.state.complete:
                raise RuntimeError(
                    "Calibration is complete; restart the client with --phase online"
                )
            fixed_action = self.calibration.fixed_action
        else:
            fixed_action = None
        if self.phase == "online":
            if not self.calibration.state.complete:
                raise RuntimeError("All four manually approved calibration speeds are required")
            stats = self.learner.stats()
            if not stats["online_training_ready"]:
                raise RuntimeError("Online replay gate requires 256 transitions and 32 per speed")
            if self._online_budget.remaining <= 0:
                raise RuntimeError("The 22-episode online training budget is exhausted")
        if self.phase == "greedy" and len(self._greedy_acceptance.outcomes) >= 5:
            raise RuntimeError("The five-episode greedy acceptance run is complete")

        self._close_episode_resources()
        self.learner.install_actor_weights(self.actor)
        self.agent.reset_execution()
        self.agent.set_fixed_action(fixed_action)
        self.agent.set_greedy(self.phase == "greedy")
        self.safety.reset_episode()
        self.agent.set_action_mask(self.safety.status().action_mask)
        self.servo = KionDualArmServo(
            self.ros_types,
            self.topics,
            control_frequency=self.control_frequency,
            gain=self.servo_gain,
            dry_run=self.dry_run,
        )
        self.servo.wait_for_connection(self.startup_timeout_s)
        self.servo.configure(self.startup_timeout_s)
        if self.pinch_enabled:
            self.pinch = KionPinchExecutor(
                self.rospy,
                self.ros_types.hand_namespace,
                max_rate_hz=self.pinch_max_rate_hz,
                dry_run=self.dry_run,
            )
        self.episode_number += 1
        run_directory = self.log_root / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_episode_{self.episode_number:03d}"
        )
        self.recorder = TrackingRecorder(
            run_directory,
            {
                "episode": self.episode_number,
                "phase": self.phase,
                "task": self.task,
                "policy_version": self.actor.policy_version,
                "fixed_action": fixed_action,
                "control_frequency": self.control_frequency,
                "speed_rl_contract": self.agent.feature_contract.to_dict(),
                "topics": asdict(self.topics),
            },
        )
        self._tracking_log = str(self.recorder.csv_path)
        self._sample_index = 0
        self._last_target = None
        self._last_outcome = None
        self._episode_started_s = time.monotonic()
        self.state = EpisodeState.RUNNING
        LOGGER.info(
            "Started Speed-RL episode=%d phase=%s fixed_action=%s policy_version=%d",
            self.episode_number,
            self.phase,
            fixed_action,
            self.actor.policy_version,
        )

    def _finish_episode(
        self,
        *,
        success: bool,
        reason: str,
        control_fault: bool = False,
    ) -> None:
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("No RUNNING episode to finish")
        duration_s = time.monotonic() - float(self._episode_started_s)
        tracking_log = None if self.recorder is None else str(self.recorder.csv_path)
        outcome = EpisodeOutcome(
            success=success,
            abort=False,
            speed_violation=self.safety.status().violation_latched,
            control_fault=control_fault,
            tracking_log=tracking_log,
            duration_s=duration_s,
        )
        transitions = self.agent.finish_episode(outcome)
        recorder = self.recorder
        self.recorder = None
        self.state = EpisodeState.TERMINATED
        self._last_outcome = "success" if success else "failure"
        self._schedule_finalization(
            outcome,
            transitions,
            recorder,
            reason=reason,
        )

    def _abort_episode(self) -> None:
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("No RUNNING episode to abort")
        duration_s = time.monotonic() - float(self._episode_started_s)
        tracking_log = None if self.recorder is None else str(self.recorder.csv_path)
        outcome = EpisodeOutcome(
            success=False,
            abort=True,
            speed_violation=self.safety.status().violation_latched,
            tracking_log=tracking_log,
            duration_s=duration_s,
        )
        transitions = self.agent.finish_episode(outcome)
        recorder = self.recorder
        self.recorder = None
        if self.servo is not None:
            self.servo.close()
            self.servo = None
        if self.pinch is not None:
            self.pinch.close()
            self.pinch = None
        self.state = EpisodeState.ABORTED
        self._last_outcome = "abort"
        LOGGER.warning(
            "Aborted episode=%d; servo is closed and replay finalization is asynchronous",
            self.episode_number,
        )
        self._schedule_finalization(
            outcome,
            transitions,
            recorder,
            reason="operator abort",
        )

    def _schedule_finalization(
        self,
        outcome: EpisodeOutcome,
        transitions: list[Transition],
        recorder: TrackingRecorder | None,
        *,
        reason: str,
    ) -> None:
        if self._finalizer is not None and self._finalizer.is_alive():
            raise RuntimeError("Another episode finalizer is already running")
        self._finalization_error = None

        def finalize() -> None:
            try:
                if recorder is not None:
                    recorder.close()
                self.learner.add_episode(transitions)
                if self.phase == "calibration":
                    self.calibration.record_episode(outcome, len(transitions))
                elif self.phase == "online":
                    self._online_budget.record()
                    self.learner.train(
                        min(2 * len(transitions), 256),
                        require_online_gate=True,
                    )
                else:
                    self._greedy_acceptance.add(outcome)
                    LOGGER.info("Greedy acceptance: %s", self._greedy_acceptance.result())
                self.learner.save(self.checkpoint_path)
                LOGGER.info(
                    "Finalized episode=%d result=%s transitions=%d speed_violation=%s reason=%s",
                    self.episode_number,
                    "success" if outcome.success else "failure",
                    len(transitions),
                    outcome.speed_violation,
                    reason,
                )
            except BaseException as exc:
                self._finalization_error = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Episode finalization failed")

        self._finalizer = threading.Thread(
            target=finalize,
            daemon=True,
            name=f"SpeedRLEpisodeFinalizer{self.episode_number}",
        )
        self._finalizer.start()

    def status(self) -> dict[str, Any]:
        diagnostics = self.agent.diagnostics()
        safety = self.safety.status()
        return {
            "state": self.state.value,
            "mode": "speed-rl",
            "inference_mode": self.agent.config.inference_mode,
            "task": self.task,
            "episode": self.episode_number,
            "phase": self.phase,
            "episode_elapsed_s": (
                None
                if self._episode_started_s is None or self.state != EpisodeState.RUNNING
                else time.monotonic() - self._episode_started_s
            ),
            "active_buffer_size": diagnostics.get("active_buffer_size"),
            "active_sequence_id": diagnostics.get("active_sequence_id"),
            "execution_state": diagnostics.get("execution_state", self.state.value),
            "inference_latency_s": diagnostics.get("latest_latency_s"),
            "planning_latency_s": diagnostics.get("latest_planning_latency_s"),
            "latest_error": diagnostics.get("latest_error"),
            "command_error": self._last_command_error,
            "speed_action": diagnostics.get("speed_rl_active_action_index"),
            "speed_scale": diagnostics.get("speed_rl_active_scale"),
            "policy_version": diagnostics.get("speed_rl_policy_version"),
            "speed_violation": safety.violation_latched,
            "twist_stale": safety.twist_stale,
            "action_mask": safety.action_mask,
            "calibration": asdict(self.calibration.state),
            "online_episodes_remaining": self._online_budget.remaining,
            "greedy_acceptance": self._greedy_acceptance.result(),
            "last_outcome": self._last_outcome,
            "tracking_log": self._tracking_log,
            "finalization_in_progress": (
                self._finalizer is not None and self._finalizer.is_alive()
            ),
            "finalization_error": self._finalization_error,
        }

    def _close_episode_resources(self) -> None:
        if self.pinch is not None:
            self.pinch.close()
            self.pinch = None
        if self.servo is not None:
            self.servo.close()
            self.servo = None
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def close(self) -> None:
        if self.state == EpisodeState.CLOSED:
            return
        self._shutdown.set()
        if self.state == EpisodeState.RUNNING:
            self._abort_episode()
        if self._ros_bridge is not None:
            self._ros_bridge.close()
            self._ros_bridge = None
        self._close_episode_resources()
        if self._finalizer is not None and self._finalizer.is_alive():
            self._finalizer.join(timeout=130.0)
            if self._finalizer.is_alive():
                raise RuntimeError("Timed out finalizing the last Speed-RL episode")
        self.agent.teardown()
        self.observations.close()
        self.state = EpisodeState.CLOSED


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent Speed-RL over the bimanual GR00T TOPPRA rollout."
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=15_000)
    parser.add_argument("--policy-frequency", type=_positive_float, default=30.0)
    parser.add_argument("--control-frequency", type=_positive_float, default=250.0)
    parser.add_argument("--inference-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--tts-samples", type=int, default=1)
    parser.add_argument("--tts-waypoint-count", type=int, default=5)
    parser.add_argument(
        "--feature-dim",
        type=int,
        help="Optional assertion; normally discovered from the policy server",
    )
    parser.add_argument(
        "--feature-model-id",
        help="Optional assertion; normally discovered from the policy server",
    )
    parser.add_argument(
        "--feature-layer",
        help="Optional assertion; normally discovered from the policy server",
    )
    parser.add_argument(
        "--feature-dtype",
        help="Optional assertion; normally discovered from the policy server",
    )
    parser.add_argument("--left-twist-thresholds", required=True)
    parser.add_argument("--right-twist-thresholds", required=True)
    parser.add_argument(
        "--phase", choices=("calibration", "online", "greedy"), default="calibration"
    )
    parser.add_argument("--state-root", type=Path, default=Path("logs/kion_speed_rl/state"))
    parser.add_argument("--log-root", type=Path, default=Path("logs/kion_speed_rl/episodes"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--async-verification", type=Path)
    parser.add_argument("--refill-threshold", type=int, default=20)
    parser.add_argument("--max-latency-s", type=float, default=0.12)
    parser.add_argument("--scheduling-margin-s", type=float, default=0.05)
    parser.add_argument("--handoff-margin-s", type=float, default=0.05)
    parser.add_argument("--open-loop-horizon", type=int, default=8)
    parser.add_argument("--velocity-filter", type=_positive_float, default=1.0)
    parser.add_argument("--servo-gain", type=int, default=800)
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=30.0)
    parser.add_argument("--max-state-age-s", type=_positive_float, default=0.25)
    parser.add_argument("--max-image-age-s", type=_positive_float, default=1.0)
    parser.add_argument("--episode-duration-s", type=_positive_float, default=80.0)
    parser.add_argument("--node-name", default="gr00t_kion_speed_rl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--control-interface",
        choices=("gui", "terminal", "both"),
        default="gui",
    )
    parser.add_argument("--ros-namespace", default="/gr00t_rollout")
    parser.add_argument("--disable-pinch", action="store_true")
    parser.add_argument("--pinch-max-rate-hz", type=_positive_float, default=30.0)
    return parser.parse_args(argv)


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if args.feature_dim is not None and args.feature_dim < 1:
        raise ValueError("--feature-dim must be positive")
    if not 100 <= args.servo_gain <= 1000:
        raise ValueError("--servo-gain must be in the SDK range [100, 1000]")
    if args.left_twist_thresholds is None or args.right_twist_thresholds is None:
        raise ValueError(
            "Certified hardware limits must be supplied explicitly through "
            "--left-twist-thresholds and --right-twist-thresholds"
        )
    if args.inference_mode == "async":
        if args.async_verification is None:
            raise ValueError(
                "Async Speed-RL is gated by --async-verification from a real-robot raw rollout"
            )
        verification = json.loads(args.async_verification.read_text(encoding="utf-8"))
        if verification.get("raw_async_rollout_verified") is not True:
            raise ValueError("Raw asynchronous rollout verification has not passed")
        if args.phase == "calibration":
            raise ValueError("Speed calibration and initial online training must use sync mode")
        if (
            args.phase == "online"
            and verification.get("speed_rl_async_greedy_verified") is not True
        ):
            raise ValueError(
                "Async online training requires a passed Speed-RL async greedy verification"
            )


def discover_server_contract(host: str, port: int, timeout_ms: int) -> FeatureContract:
    client = GR00TPolicyClient(host, port, timeout_ms=timeout_ms)
    try:
        client.ping()
        value = client.call_endpoint("get_speed_rl_contract", requires_input=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover the Speed-RL contract from {host}:{port}. "
            "Deploy this repository on the inference server, start it with "
            "--speed-rl-features, and retry."
        ) from exc
    finally:
        client.close()
    if not isinstance(value, Mapping):
        raise ValueError(f"Server Speed-RL contract must be a mapping, got {type(value)!r}")
    return FeatureContract.from_mapping(value)


def _resolve_feature_contract(args: argparse.Namespace) -> FeatureContract:
    contract = discover_server_contract(args.server_host, args.server_port, args.timeout_ms)
    assertions = {
        "feature_dim": args.feature_dim,
        "model_id": args.feature_model_id,
        "layer": args.feature_layer,
        "dtype": args.feature_dtype,
    }
    mismatches = {
        field: (expected, getattr(contract, field))
        for field, expected in assertions.items()
        if expected is not None and expected != getattr(contract, field)
    }
    if mismatches:
        raise ValueError(
            f"CLI feature assertions do not match the policy server contract: {mismatches}"
        )
    return contract


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_runtime_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    contract = _resolve_feature_contract(args)
    LOGGER.info("Discovered Speed-RL server contract: %s", contract.to_dict())
    rainbow_config = RainbowConfig(feature_dim=contract.feature_dim)
    actor = SpeedActor(rainbow_config)
    learner = LearnerProcess(rainbow_config, contract)
    checkpoint = args.checkpoint or args.state_root / "speed_rl.pt"
    if (args.inference_mode == "async" or args.phase == "greedy") and not checkpoint.exists():
        learner.close()
        raise ValueError(
            "Async or greedy Speed-RL requires an existing synchronous-training checkpoint"
        )
    if checkpoint.exists():
        learner.load(checkpoint)
    learner_stats = learner.stats()
    calibration = CalibrationManager(args.state_root / "calibration.json")
    safety = SpeedViolationMonitor(
        parse_twist_thresholds(args.left_twist_thresholds),
        parse_twist_thresholds(args.right_twist_thresholds),
    )
    limits = CartesianLimits(
        max_linear_velocity=(1.0,) * 3,
        max_angular_velocity=(3.0,) * 3,
        max_linear_acceleration=(5.0,) * 3,
        max_angular_acceleration=(15.0,) * 3,
        safety_margin=1.0,
    )
    ros_types = load_ros_types()
    ros_types.rospy.init_node(args.node_name, disable_signals=False)
    agent = SpeedRLAgent(
        host=args.server_host,
        port=args.server_port,
        fps=args.policy_frequency,
        refill_threshold=args.refill_threshold,
        max_latency_s=args.max_latency_s,
        task_default=args.task,
        timeout_ms=args.timeout_ms,
        left_limits=limits,
        right_limits=limits,
        inference_mode=args.inference_mode,
        control_frequency=args.control_frequency,
        scheduling_margin_s=args.scheduling_margin_s,
        handoff_margin_s=args.handoff_margin_s,
        open_loop_horizon=args.open_loop_horizon,
        tts_samples=args.tts_samples,
        tts_waypoint_count=args.tts_waypoint_count,
        left_velocity_state_key=LEFT_TWIST_KEY,
        right_velocity_state_key=RIGHT_TWIST_KEY,
        velocity_filter=args.velocity_filter,
        include_velocity_in_policy_observation=False,
        feature_contract=contract,
        speed_actor=actor,
        greedy=args.phase == "greedy",
        activated_decisions=learner_stats["activated_decisions"],
    )
    client = SpeedRLKionClient(
        ros_types=ros_types,
        topics=KionTopics(),
        agent=agent,
        actor=actor,
        learner=learner,
        safety=safety,
        calibration=calibration,
        phase=args.phase,
        checkpoint_path=checkpoint,
        log_root=args.log_root,
        control_frequency=args.control_frequency,
        servo_gain=args.servo_gain,
        startup_timeout_s=args.startup_timeout_s,
        max_state_age_s=args.max_state_age_s,
        max_image_age_s=args.max_image_age_s,
        task=args.task,
        dry_run=args.dry_run,
        episode_duration_s=args.episode_duration_s,
        control_interface=args.control_interface,
        ros_namespace=args.ros_namespace,
        pinch_enabled=not args.disable_pinch,
        pinch_max_rate_hz=args.pinch_max_rate_hz,
    )
    try:
        client.run()
    finally:
        if client.state != EpisodeState.CLOSED:
            client.close()
        learner.close()
