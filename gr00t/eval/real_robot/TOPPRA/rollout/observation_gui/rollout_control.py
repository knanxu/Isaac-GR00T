from __future__ import annotations

import json
import os
from queue import Empty, Full, Queue
import threading
import time
from typing import Any

import dearpygui.dearpygui as dpg
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from gui.module import GUIModule


class RolloutControlPanel(GUIModule):
    """ObservationGUILite panel for an external 250 Hz GR00T rollout process."""

    _HIDDEN_STANDARD_ITEMS = (
        "task_input",
        "btn_record",
        "btn_inference",
        "btn_training",
        "btn_stop",
        "btn_save",
        "btn_discard",
        "chk_recording_states",
        "btn_success",
        "status_text",
    )

    def __init__(self, namespace: str = "/gr00t_rollout") -> None:
        super().__init__("GR00T Rollout", placement="top")
        self.namespace = "/" + namespace.strip("/")
        self._status: dict[str, Any] = {"state": "connecting"}
        self._status_lock = threading.Lock()
        self._last_status_s = 0.0
        self._last_result = "Waiting for rollout status"
        self._commands: Queue[str] = Queue(maxsize=8)
        self._shutdown = threading.Event()
        self._subscriber = rospy.Subscriber(
            f"{self.namespace}/status",
            String,
            self._status_callback,
            queue_size=1,
        )
        self._worker = threading.Thread(
            target=self._command_loop,
            daemon=True,
            name="RolloutGUICommands",
        )
        self._worker.start()

    def _status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        with self._status_lock:
            self._status = status
            self._last_status_s = time.monotonic()

    def _enqueue(self, command: str) -> None:
        try:
            self._commands.put_nowait(command)
        except Full:
            self._last_result = "Command queue is busy"

    def _command_loop(self) -> None:
        while not self._shutdown.is_set() and not rospy.is_shutdown():
            try:
                command = self._commands.get(timeout=0.2)
            except Empty:
                continue
            service_name = f"{self.namespace}/{command}"
            try:
                rospy.wait_for_service(service_name, timeout=2.0)
                response = rospy.ServiceProxy(service_name, Trigger)()
                self._last_result = response.message
            except Exception as exc:
                self._last_result = f"{command} failed: {exc}"

    def setup_gui(self) -> None:
        for item in self._HIDDEN_STANDARD_ITEMS:
            if dpg.does_item_exist(item):
                dpg.configure_item(item, show=False)

        dpg.add_spacer(height=8)
        dpg.add_text("External GR00T Rollout", color=(255, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Start Episode",
                height=60,
                tag="rollout_start",
                callback=lambda: self._enqueue("start"),
            )
            dpg.add_button(
                label="Success",
                height=60,
                tag="rollout_success",
                callback=lambda: self._enqueue("success"),
            )
            dpg.add_button(
                label="Failure",
                height=60,
                tag="rollout_failure",
                callback=lambda: self._enqueue("failure"),
            )
            dpg.add_button(
                label="Abort / Close Servo",
                height=60,
                tag="rollout_abort",
                callback=lambda: self._enqueue("abort"),
            )
            dpg.add_button(
                label="Approve Calibrated Speed",
                height=60,
                tag="rollout_approve",
                callback=lambda: self._enqueue("approve"),
            )
        dpg.add_text("State: connecting", tag="rollout_state_text")
        dpg.add_text("Trajectory: ---", tag="rollout_trajectory_text")
        dpg.add_text("Speed-RL: ---", tag="rollout_speed_text")
        dpg.add_text("Safety: ---", tag="rollout_safety_text")
        dpg.add_text(self._last_result, tag="rollout_command_text", wrap=1800)

    @staticmethod
    def _fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "---"
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
        return f"{value}{suffix}"

    def update_gui(self) -> None:
        with self._status_lock:
            status = dict(self._status)
            age_s = time.monotonic() - self._last_status_s if self._last_status_s else None
        state = str(status.get("state", "connecting"))
        running = state == "running"
        finalizing = bool(status.get("finalization_in_progress", False))
        mode = status.get("mode", "---")
        phase = status.get("phase")
        stale_label = "no status" if age_s is None else f"status age {age_s:.1f}s"

        dpg.set_value(
            "rollout_state_text",
            f"State: {state} | mode={mode} | inference={status.get('inference_mode', '---')} "
            f"| episode={status.get('episode', 0)} | task={status.get('task', '---')} "
            f"| {stale_label}",
        )
        dpg.set_value(
            "rollout_trajectory_text",
            "Trajectory: sequence={} | buffer={} | inference={} | planning={} | error={}".format(
                self._fmt(status.get("active_sequence_id")),
                self._fmt(status.get("active_buffer_size")),
                self._fmt(status.get("inference_latency_s"), " s"),
                self._fmt(status.get("planning_latency_s"), " s"),
                status.get("command_error") or status.get("latest_error") or "none",
            ),
        )
        dpg.set_value(
            "rollout_speed_text",
            "Speed-RL: phase={} | action={} | scale={} | policy={}".format(
                phase or "disabled",
                self._fmt(status.get("speed_action")),
                self._fmt(status.get("speed_scale")),
                self._fmt(status.get("policy_version")),
            ),
        )
        dpg.set_value(
            "rollout_safety_text",
            "Safety: violation={} | twist_stale={} | mask={} | outcome={}".format(
                status.get("speed_violation", False),
                status.get("twist_stale", "---"),
                status.get("action_mask", "---"),
                status.get("last_outcome", "---"),
            ),
        )
        dpg.set_value("rollout_command_text", self._last_result)
        dpg.configure_item("rollout_start", enabled=not running and not finalizing)
        dpg.configure_item("rollout_success", enabled=running)
        dpg.configure_item("rollout_failure", enabled=running)
        dpg.configure_item("rollout_abort", enabled=running)
        awaiting_approval = bool((status.get("calibration") or {}).get("awaiting_approval"))
        dpg.configure_item(
            "rollout_approve",
            enabled=mode == "speed-rl" and awaiting_approval and not running,
        )


def configured_namespace() -> str:
    return os.environ.get("GR00T_ROLLOUT_NAMESPACE", "/gr00t_rollout")
