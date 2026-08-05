from __future__ import annotations

from enum import Enum, auto
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from gr00t.eval.real_robot.TOPPRA.kion_client.client import KionTopics
from gr00t.eval.real_robot.TOPPRA.rollout import plain_client as plain_module
from gr00t.eval.real_robot.TOPPRA.rollout.observation import (
    CRITICAL_ROLLOUT_KEYS,
    LOCAL_TWIST_KEYS,
    RolloutObservationBuffer,
    policy_observation,
    rollout_staleness,
)
from gr00t.eval.real_robot.TOPPRA.rollout.operator import RosEpisodeBridge, status_json
from gr00t.eval.real_robot.TOPPRA.rollout.pinch import (
    LEFT_HIGH,
    LEFT_LOW,
    RIGHT_HIGH,
    RIGHT_LOW,
    pinch_posture,
)
from gr00t.eval.real_robot.TOPPRA.rollout.plain_client import _parse_args as parse_plain_args
from gr00t.eval.real_robot.TOPPRA.speed_rl.client import _parse_args as parse_speed_args
import numpy as np


class _TriggerResponse:
    def __init__(self, *, success: bool, message: str) -> None:
        self.success = success
        self.message = message


class _String:
    def __init__(self) -> None:
        self.data = ""


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[Any] = []
        self.closed = False

    def publish(self, message: Any) -> None:
        self.messages.append(message)

    def unregister(self) -> None:
        self.closed = True


class _Service:
    def __init__(self, name: str, callback: Any) -> None:
        self.name = name
        self.callback = callback
        self.closed = False

    def shutdown(self, _reason: str) -> None:
        self.closed = True


class _Rospy:
    def __init__(self) -> None:
        self.publisher = _Publisher()
        self.services: dict[str, _Service] = {}

    def Publisher(self, *_args: Any, **_kwargs: Any) -> _Publisher:  # noqa: N802
        return self.publisher

    def Service(self, name: str, _service_type: Any, callback: Any) -> _Service:  # noqa: N802
        service = _Service(name, callback)
        self.services[name] = service
        return service


def test_rollout_status_json_normalizes_numpy_and_nonfinite_values() -> None:
    encoded = status_json(
        {
            "mask": np.array([True, False]),
            "action": np.int64(2),
            "latency": float("nan"),
            "nested": (np.float32(1.25),),
        }
    )
    assert json.loads(encoded) == {
        "action": 2,
        "latency": None,
        "mask": [True, False],
        "nested": [1.25],
    }


def test_ros_episode_bridge_queues_commands_and_publishes_status(monkeypatch) -> None:
    fake_rospy = _Rospy()
    commands: list[str] = []

    def fake_import(name: str) -> Any:
        if name == "std_srvs.srv":
            return SimpleNamespace(Trigger=object, TriggerResponse=_TriggerResponse)
        if name == "std_msgs.msg":
            return SimpleNamespace(String=_String)
        raise ImportError(name)

    monkeypatch.setattr(
        "gr00t.eval.real_robot.TOPPRA.rollout.operator.importlib.import_module",
        fake_import,
    )
    bridge = RosEpisodeBridge(
        fake_rospy,
        commands.append,
        lambda: {"state": "running", "speed_scale": np.float32(1.3)},
        namespace="test_rollout",
    )
    response = fake_rospy.services["/test_rollout/start"].callback(None)
    assert response.success
    assert commands == ["start"]
    fake_rospy.services["/test_rollout/approve"].callback(None)
    assert commands[-1].startswith("approve ObservationGUILite")

    status_response = fake_rospy.services["/test_rollout/status"].callback(None)
    assert json.loads(status_response.message)["state"] == "running"
    bridge.publish_status(force=True)
    assert json.loads(fake_rospy.publisher.messages[-1].data)["speed_scale"] == np.float32(1.3)
    bridge.close()
    assert fake_rospy.publisher.closed
    assert all(service.closed for service in fake_rospy.services.values())


def test_pinch_mapping_matches_observation_gui_calibration() -> None:
    np.testing.assert_allclose(pinch_posture(np.array([0.0]), LEFT_LOW, LEFT_HIGH), LEFT_LOW)
    np.testing.assert_allclose(pinch_posture(np.array([1.0]), LEFT_LOW, LEFT_HIGH), LEFT_HIGH)
    np.testing.assert_allclose(pinch_posture(np.array([-1.0]), RIGHT_LOW, RIGHT_HIGH), RIGHT_LOW)
    np.testing.assert_allclose(pinch_posture(np.array([2.0]), RIGHT_LOW, RIGHT_HIGH), RIGHT_HIGH)


def test_unified_clients_default_to_sync_single_candidate_gui_control() -> None:
    plain = parse_plain_args([])
    speed = parse_speed_args(
        [
            "--left-twist-thresholds",
            "1,1,1,1,1,1",
            "--right-twist-thresholds",
            "1,1,1,1,1,1",
        ]
    )
    for args in (plain, speed):
        assert args.inference_mode == "sync"
        assert args.tts_samples == 1
        assert args.control_interface == "gui"
        assert args.ros_namespace == "/gr00t_rollout"
        assert args.server_port == 47866


def test_rollout_observation_contract_ignores_pressure_and_omits_stale_twist() -> None:
    assert all("finger_pressure" not in key for key in RolloutObservationBuffer.REQUIRED_KEYS)
    critical_key = next(iter(CRITICAL_ROLLOUT_KEYS))
    critical, twist_stale = rollout_staleness((critical_key, LOCAL_TWIST_KEYS[0]))
    assert critical == (critical_key,)
    assert twist_stale

    observation = {
        key: np.zeros(1, dtype=np.float32) for key in RolloutObservationBuffer.REQUIRED_KEYS
    }
    observation["observation.state.left_finger_pressure"] = np.ones(6, dtype=np.float32)
    selected = policy_observation(observation, twist_stale=True)
    assert not any(key in selected for key in LOCAL_TWIST_KEYS)
    assert "observation.state.left_finger_pressure" not in selected


def test_observation_gui_passively_saves_labels_and_discards_abort(monkeypatch) -> None:
    class Mode(Enum):
        IDLE = auto()
        RECORDING = auto()
        REVIEWING = auto()

    dearpygui_package = ModuleType("dearpygui")
    dearpygui_module = ModuleType("dearpygui.dearpygui")
    dearpygui_package.dearpygui = dearpygui_module
    gui_package = ModuleType("gui")
    gui_module = ModuleType("gui.module")
    gui_module.GUIModule = object
    gui_package.module = gui_module
    controlloop = ModuleType("controlloop")
    controlloop.Mode = Mode
    std_msgs = ModuleType("std_msgs.msg")
    std_msgs.String = object
    std_srvs = ModuleType("std_srvs.srv")
    std_srvs.Trigger = object
    for name, module in {
        "dearpygui": dearpygui_package,
        "dearpygui.dearpygui": dearpygui_module,
        "gui": gui_package,
        "gui.module": gui_module,
        "controlloop": controlloop,
        "rospy": ModuleType("rospy"),
        "std_msgs.msg": std_msgs,
        "std_srvs.srv": std_srvs,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        Path(__file__).parents[3]
        / "gr00t/eval/real_robot/TOPPRA/rollout/observation_gui/rollout_control.py"
    )
    spec = importlib.util.spec_from_file_location("test_rollout_control_overlay", source)
    assert spec is not None and spec.loader is not None
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)

    class FakeState:
        def __init__(self) -> None:
            self.mode = Mode.IDLE
            self.task = ""
            self.episode_success = False

        def set_mode(self, mode: Mode) -> None:
            self.mode = mode

    class FakeLoop:
        def __init__(self) -> None:
            self.frames: list[dict[str, Any]] = []
            self.debug_events: list[dict[str, Any]] = []
            self.saved: list[tuple[bool, list[dict[str, Any]]]] = []
            self.discarded = 0

        def save_episode(self) -> None:
            self.saved.append((state.episode_success, list(self.debug_events)))
            self.frames.clear()
            self.debug_events.clear()

        def discard_episode(self) -> None:
            self.discarded += 1
            self.frames.clear()

    state = FakeState()
    loop = FakeLoop()
    panel = overlay.RolloutControlPanel.__new__(overlay.RolloutControlPanel)
    panel._control_loop = loop
    panel._control_state = state
    panel._recording_episode = None
    panel._saved_episode = None
    panel._last_result = ""

    panel._sync_passive_recording({"state": "running", "episode": 3, "task": "parcel"})
    assert state.mode is Mode.RECORDING
    assert state.task == "parcel"
    loop.frames.extend([{}, {}, {}])
    panel._sync_passive_recording(
        {
            "state": "terminated",
            "episode": 3,
            "last_outcome": "success",
            "last_safe_success": True,
            "speed_violation": False,
            "tracking_log": "tracking.csv",
        }
    )
    assert panel._saved_episode == 3
    assert loop.saved[0][0] is True
    assert loop.saved[0][1][0]["events"][0]["outcome"] == "success"

    panel._sync_passive_recording({"state": "running", "episode": 4, "task": "parcel"})
    panel._sync_passive_recording({"state": "aborted", "episode": 4, "last_outcome": "abort"})
    assert loop.discarded == 1


def test_plain_episode_lifecycle_recreates_agent_and_persists_operator_outcome(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeObservations:
        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeServo:
        instances = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.closed = False
            self.instances.append(self)

        def wait_for_connection(self, _timeout: float) -> None:
            return None

        def configure(self, _timeout: float) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakePinch(FakeServo):
        pass

    class FakeRecorder:
        def __init__(self, directory, _metadata) -> None:
            self.directory = directory
            self.directory.mkdir(parents=True)
            self.csv_path = self.directory / "tcp_tracking.csv"
            self.csv_path.touch()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeAgent:
        def __init__(self) -> None:
            self.closed = False

        def diagnostics(self) -> dict[str, Any]:
            return {"execution_state": "hold"}

        def teardown(self) -> None:
            self.closed = True

    monkeypatch.setattr(plain_module, "RolloutObservationBuffer", FakeObservations)
    monkeypatch.setattr(plain_module, "KionDualArmServo", FakeServo)
    monkeypatch.setattr(plain_module, "KionPinchExecutor", FakePinch)
    monkeypatch.setattr(plain_module, "TrackingRecorder", FakeRecorder)

    agents: list[FakeAgent] = []

    def agent_factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    ros_types = SimpleNamespace(rospy=object(), hand_namespace="test.hand.msg")
    client = plain_module.PlainKionEpisodeClient(
        ros_types=ros_types,
        topics=KionTopics(),
        agent_factory=agent_factory,
        log_root=tmp_path,
        control_frequency=250,
        servo_gain=800,
        startup_timeout_s=1,
        max_state_age_s=1,
        max_image_age_s=1,
        task="parcel",
        inference_mode="sync",
        dry_run=True,
    )
    client._start_episode()
    assert client.state is plain_module.EpisodeState.RUNNING
    first_directory = client.recorder.directory
    client._finish_episode(success=True, reason="operator success")
    client._finalizer.join(timeout=2)
    assert client.state is plain_module.EpisodeState.TERMINATED
    assert json.loads((first_directory / "outcome.json").read_text())["success"] is True

    client._start_episode()
    assert len(agents) == 2
    assert agents[0].closed
    client._abort_episode(reason="operator abort")
    client._finalizer.join(timeout=2)
    assert client.state is plain_module.EpisodeState.ABORTED
    assert all(item.closed for item in FakeServo.instances)
    client.close()
    assert client.state is plain_module.EpisodeState.CLOSED
