from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from gr00t.eval.real_robot.TOPPRA.kion_client.client import KionTopics
from gr00t.eval.real_robot.TOPPRA.rollout import plain_client as plain_module
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

    monkeypatch.setattr(plain_module, "KionObservationBuffer", FakeObservations)
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
