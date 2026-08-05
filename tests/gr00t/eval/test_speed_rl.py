from __future__ import annotations

import random
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from gr00t.eval.real_robot.TOPPRA.eval_toppra_bimanual import (
    BimanualToppraRolloutConfig,
    CartesianLimits,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl import client as speed_rl_client
from gr00t.eval.real_robot.TOPPRA.speed_rl.agent import SpeedRLAgent, speed_planner_configs
from gr00t.eval.real_robot.TOPPRA.speed_rl.baseline import verify_frozen_baseline
from gr00t.eval.real_robot.TOPPRA.speed_rl.client import _parse_args, _validate_runtime_args
from gr00t.eval.real_robot.TOPPRA.speed_rl.config import RainbowConfig
from gr00t.eval.real_robot.TOPPRA.speed_rl.contract import (
    FeatureContract,
    extract_candidate_features,
    pool_candidate_feature,
    trim_candidate_features,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.learner import LearnerProcess, RainbowLearner, SpeedActor
from gr00t.eval.real_robot.TOPPRA.speed_rl.network import (
    DuelingC51Network,
    project_categorical_distribution,
    select_masked_action,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.phases import CalibrationManager, EpisodeOutcome
from gr00t.eval.real_robot.TOPPRA.speed_rl.probe_server import build_kion_probe_observation
from gr00t.eval.real_robot.TOPPRA.speed_rl.replay import (
    PrioritizedReplayBuffer,
    Transition,
    build_n_step_transitions,
)
from gr00t.eval.real_robot.TOPPRA.speed_rl.safety import SpeedViolationMonitor


FEATURE_DIM = 8
CONTRACT = FeatureContract(
    feature_dim=FEATURE_DIM,
    model_id="synthetic-vla",
    layer="action_decoder.hidden",
)


def synthetic_bandit_episodes(
    feature_dim: int,
    count: int,
    *,
    seed: int = 0,
) -> list[list[Transition]]:
    rng = np.random.default_rng(seed)
    mask = np.ones(4, dtype=np.bool_)
    episodes: list[list[Transition]] = []
    for index in range(count):
        best_action = index % 4
        state = rng.normal(0.0, 0.02, feature_dim).astype(np.float32)
        state[best_action] += 1.0
        action = (index // 4) % 4
        reward = 100.0 if action == best_action else 0.0
        episodes.append(
            [
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=state.copy(),
                    done=True,
                    action_mask=mask.copy(),
                    next_action_mask=mask.copy(),
                )
            ]
        )
    return episodes


def _base_limits() -> CartesianLimits:
    return CartesianLimits(
        max_linear_velocity=(1.0,) * 3,
        max_angular_velocity=(3.0,) * 3,
        max_linear_acceleration=(5.0,) * 3,
        max_angular_acceleration=(15.0,) * 3,
        safety_margin=1.0,
    )


def _transition(action: int, reward: float = 1.0, done: bool = False) -> Transition:
    state = np.full(FEATURE_DIM, action, dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    return Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=state + 1.0,
        done=done,
        action_mask=mask,
        next_action_mask=mask,
    )


def test_frozen_toppra_and_kion_baseline_hashes() -> None:
    assert len(verify_frozen_baseline()) == 7


def test_dueling_c51_shape_layernorm_and_arbitrary_mask() -> None:
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    network = DuelingC51Network(config)
    logits = network(torch.randn(3, FEATURE_DIM))
    assert logits.shape == (3, 4, 51)
    assert isinstance(network.input_norm, torch.nn.LayerNorm)
    assert network.backbone[0].out_features == 256
    assert network.backbone[2].out_features == 256

    action = select_masked_action(
        network,
        np.zeros(FEATURE_DIM, dtype=np.float32),
        np.array([False, False, True, False]),
        epsilon=1.0,
        rng=random.Random(0),
    )
    assert action == 2
    with pytest.raises(ValueError, match="at least one"):
        select_masked_action(
            network,
            np.zeros(FEATURE_DIM, dtype=np.float32),
            np.zeros(4, dtype=np.bool_),
            epsilon=0.0,
        )


def test_c51_projection_preserves_probability_and_terminal_reward() -> None:
    support = torch.linspace(0.0, 260.0, 51)
    probabilities = torch.zeros(2, 51)
    probabilities[:, 10] = 1.0
    projected = project_categorical_distribution(
        probabilities,
        rewards=torch.tensor([26.0, 130.0]),
        dones=torch.tensor([False, True]),
        discounts=torch.tensor([0.99**3, 0.99**3]),
        support=support,
    )
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(2))
    expected_terminal = torch.sum(projected[1] * support)
    assert float(expected_terminal) == pytest.approx(130.0, abs=1e-4)


def test_three_step_return_and_prioritized_replay_sampling() -> None:
    episode = [
        _transition(0, reward=1.0),
        _transition(1, reward=2.0),
        _transition(2, reward=3.0),
        _transition(3, reward=4.0, done=True),
    ]
    n_step = build_n_step_transitions(episode, n_step=3, gamma=0.99)
    assert n_step[0].steps == 3
    assert n_step[0].reward == pytest.approx(1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)
    assert not n_step[0].done
    assert n_step[1].done

    replay = PrioritizedReplayBuffer(16, alpha=0.6, seed=0)
    replay.extend(n_step)
    replay.update_priorities(
        np.arange(4, dtype=np.int64),
        np.array([1.0, 1.0, 1.0, 100.0]),
    )
    hits = 0
    for _ in range(200):
        hits += int(replay.sample(1, beta=0.4).indices[0] == 3)
    assert hits > 100


def test_feature_contract_and_action_trim_stay_aligned() -> None:
    actions = {"left_delta_tcp": np.zeros((2, 6, 6), dtype=np.float32)}
    features = np.arange(2 * 6 * FEATURE_DIM, dtype=np.float32).reshape(
        2,
        6,
        FEATURE_DIM,
    )
    response = (
        actions,
        {
            "speed_rl": {
                "action_features": features,
                "contract": CONTRACT.to_dict(),
            }
        },
    )
    extracted = extract_candidate_features(response, CONTRACT, (6, 6))
    trimmed = trim_candidate_features(extracted, 2)
    assert [item.shape for item in trimmed] == [(4, FEATURE_DIM), (4, FEATURE_DIM)]
    np.testing.assert_array_equal(trimmed[1][0], features[1, 2])
    np.testing.assert_allclose(pool_candidate_feature(trimmed[0]), features[0, 2:].mean(axis=0))

    bad_response = (
        actions,
        {
            "speed_rl": {
                "action_features": features.astype(np.float64),
                "contract": CONTRACT.to_dict(),
            }
        },
    )
    with pytest.raises(ValueError, match="dtype"):
        extract_candidate_features(bad_response, CONTRACT, (6, 6))


def test_speed_planners_use_exact_unmargined_constraints() -> None:
    base = BimanualToppraRolloutConfig(
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    configs = speed_planner_configs(base)
    for config, scale in zip(configs, (0.7, 1.0, 1.3, 1.6), strict=True):
        np.testing.assert_allclose(
            config.left_limits.velocity,
            np.array([1.0, 1.0, 1.0, 3.0, 3.0, 3.0]) * scale,
        )
        np.testing.assert_allclose(
            config.right_limits.acceleration,
            np.array([5.0, 5.0, 5.0, 15.0, 15.0, 15.0]) * scale**2,
        )
        assert config.left_limits.safety_margin == 1.0


def test_speed_violation_latches_on_frame_100_and_stale_masks_fast_actions() -> None:
    monitor = SpeedViolationMonitor(np.ones(6), np.ones(6))
    over = np.array([1.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    zero = np.zeros(6)
    for _ in range(99):
        status = monitor.update(over, zero, stale=False)
    assert not status.violation_latched
    status = monitor.update(over, zero, stale=False)
    assert status.violation_latched

    monitor.reset_episode()
    for _ in range(99):
        monitor.update(over, zero, stale=False)
    stale = monitor.update(over, zero, stale=True)
    assert stale.consecutive_counts[0][0] == 0
    assert stale.action_mask == (True, True, False, False)
    recovered = monitor.update(zero, zero, stale=False)
    assert recovered.action_mask == (True, True, True, True)


def test_calibration_requires_two_valid_episodes_and_explicit_approval(tmp_path) -> None:
    calibration = CalibrationManager(tmp_path / "calibration.json")
    assert calibration.fixed_action == 0
    calibration.record_episode(EpisodeOutcome(success=True, tracking_log="one.csv"), 3)
    assert calibration.fixed_action == 0
    calibration.record_episode(EpisodeOutcome(success=False, tracking_log="two.csv"), 2)
    assert calibration.state.awaiting_approval
    with pytest.raises(RuntimeError, match="approval"):
        _ = calibration.fixed_action
    calibration.approve_current_speed("reviewed tracking_report.json")
    assert calibration.fixed_action == 1

    calibration.record_episode(EpisodeOutcome(success=False, abort=True), 1)
    assert calibration.state.stopped
    with pytest.raises(RuntimeError, match="stopped"):
        _ = calibration.fixed_action


def test_async_runtime_gate_requires_raw_and_speed_rl_greedy_verification(tmp_path) -> None:
    common = [
        "--feature-dim",
        str(FEATURE_DIM),
        "--feature-model-id",
        CONTRACT.model_id,
        "--feature-layer",
        CONTRACT.layer,
        "--left-twist-thresholds",
        "1,1,1,1,1,1",
        "--right-twist-thresholds",
        "1,1,1,1,1,1",
    ]
    _validate_runtime_args(_parse_args(common))
    with pytest.raises(ValueError, match="async-verification"):
        _validate_runtime_args(
            _parse_args([*common, "--inference-mode", "async", "--phase", "greedy"])
        )

    verification = tmp_path / "async.json"
    verification.write_text('{"raw_async_rollout_verified": true}', encoding="utf-8")
    _validate_runtime_args(
        _parse_args(
            [
                *common,
                "--inference-mode",
                "async",
                "--phase",
                "greedy",
                "--async-verification",
                str(verification),
            ]
        )
    )
    with pytest.raises(ValueError, match="async greedy"):
        _validate_runtime_args(
            _parse_args(
                [
                    *common,
                    "--inference-mode",
                    "async",
                    "--phase",
                    "online",
                    "--async-verification",
                    str(verification),
                ]
            )
        )


def test_server_contract_is_discovered_and_optional_cli_values_are_assertions(monkeypatch) -> None:
    args = _parse_args(
        [
            "--left-twist-thresholds",
            "1,1,1,1,1,1",
            "--right-twist-thresholds",
            "1,1,1,1,1,1",
        ]
    )
    _validate_runtime_args(args)
    monkeypatch.setattr(speed_rl_client, "discover_server_contract", lambda *args: CONTRACT)
    assert speed_rl_client._resolve_feature_contract(args) == CONTRACT

    args.feature_model_id = "wrong-model"
    with pytest.raises(ValueError, match="assertions"):
        speed_rl_client._resolve_feature_contract(args)


def test_kion_probe_observation_uses_server_modality_keys() -> None:
    config = {
        "video": {"modality_keys": ["head", "left", "right"]},
        "state": {
            "modality_keys": [
                "left_tcp",
                "left_wrist_force",
                "right_tcp",
                "right_wrist_force",
            ]
        },
        "language": {"modality_keys": ["annotation.human.task_description"]},
    }
    observation = build_kion_probe_observation(
        config,
        task="test task",
        batch_size=2,
        image_size=32,
    )
    assert observation["video"]["head"].shape == (2, 1, 32, 32, 3)
    assert observation["state"]["left_tcp"].shape == (2, 1, 7)
    assert observation["state"]["left_wrist_force"].shape == (2, 1, 3)
    assert observation["language"]["annotation.human.task_description"] == [
        ["test task"],
        ["test task"],
    ]


def test_synthetic_rainbow_training_converges_and_checkpoint_roundtrips(tmp_path) -> None:
    torch.set_num_threads(1)
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    learner = RainbowLearner(config, CONTRACT, seed=3)
    for episode in synthetic_bandit_episodes(FEATURE_DIM, 1024, seed=4):
        learner.add_episode(episode)
    result = learner.train_updates(300, require_online_gate=False)
    assert result["updates"] == 300

    probes = torch.eye(FEATURE_DIM, dtype=torch.float32)[:4]
    predicted = torch.argmax(learner.online.q_values(probes), dim=1)
    torch.testing.assert_close(predicted, torch.arange(4))

    path = tmp_path / "speed_rl.pt"
    learner.save_checkpoint(path)
    restored = RainbowLearner(config, CONTRACT, seed=9)
    restored.load_checkpoint(path)
    assert restored.stats() == learner.stats()
    for expected, actual in zip(
        learner.online.parameters(),
        restored.online.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)

    incompatible = RainbowLearner(
        config,
        FeatureContract(
            feature_dim=FEATURE_DIM,
            model_id="other-model",
            layer=CONTRACT.layer,
        ),
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        incompatible.load_checkpoint(path)


@pytest.mark.serial
def test_spawn_learner_process_replay_and_checkpoint(tmp_path) -> None:
    config = RainbowConfig(feature_dim=FEATURE_DIM)
    actor = SpeedActor(config, seed=0)
    checkpoint = tmp_path / "spawn.pt"
    with LearnerProcess(config, CONTRACT, response_timeout_s=30.0) as learner:
        assert learner.add_episode([_transition(0, done=True)]) == 1
        assert learner.stats()["replay_size"] == 1
        assert learner.install_actor_weights(actor) == 0
        learner.save(checkpoint)
    assert checkpoint.exists()


class _DelayedFeaturePolicy:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.started = threading.Event()

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del observation, options
        self.started.set()
        time.sleep(self.delay_s)
        horizon = 12
        left = np.zeros((1, horizon, 6), dtype=np.float32)
        right = np.zeros_like(left)
        left[..., 0] = 0.003
        right[..., 1] = -0.003
        pinch = np.zeros((1, horizon, 1), dtype=np.float32)
        return (
            {
                "left_delta_tcp": left,
                "right_delta_tcp": right,
                "left_pinch": pinch,
                "right_pinch": pinch.copy(),
            },
            {
                "speed_rl": {
                    "action_features": np.ones(
                        (1, horizon, FEATURE_DIM),
                        dtype=np.float32,
                    ),
                    "contract": CONTRACT.to_dict(),
                }
            },
        )

    def close(self) -> None:
        return None


def _observation() -> dict[str, np.ndarray]:
    return {
        "observation.state.left_tcp": np.array(
            [0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        "observation.state.right_tcp": np.array(
            [0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    }


def _wait_for_generation(agent: SpeedRLAgent, generation: int) -> None:
    deadline = time.perf_counter() + 3.0
    while agent.diagnostics()["completed_generation"] < generation:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Speed-RL fake policy did not complete")
        time.sleep(0.002)


@pytest.mark.parametrize("delay_s", (0.1, 0.3))
def test_fake_server_latency_keeps_act_below_control_budget(delay_s: float) -> None:
    policy = _DelayedFeaturePolicy(delay_s)
    actor = SpeedActor(RainbowConfig(feature_dim=FEATURE_DIM), seed=0)
    agent = SpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    observation = _observation()
    try:
        agent.act(observation)
        assert policy.started.wait(timeout=1.0)
        durations_ms = []
        while agent.diagnostics()["completed_generation"] < 1:
            # Thread CPU time excludes unrelated host preemption on shared CI workers.
            started_ns = time.thread_time_ns()
            agent.act(observation)
            durations_ms.append((time.thread_time_ns() - started_ns) * 1e-6)
            time.sleep(0.0002)
        assert np.percentile(durations_ms, 99) < 1.0
        assert max(durations_ms) < 4.0
        assert len(agent.trajectory_decisions) == 1
        assert agent.diagnostics()["speed_rl_activated_decisions"] == 0

        agent.act(observation)
        assert agent.diagnostics()["speed_rl_activated_decisions"] == 1
        transitions = agent.finish_episode(EpisodeOutcome(success=True))
        assert len(transitions) == 1
        assert any(
            transitions[0].reward == pytest.approx(expected)
            for expected in (0.7**2, 1.0, 1.3**2, 1.6**2)
        )
    finally:
        agent.teardown()


def test_reset_execution_invalidates_late_policy_result() -> None:
    policy = _DelayedFeaturePolicy(0.1)
    actor = SpeedActor(RainbowConfig(feature_dim=FEATURE_DIM), seed=0)
    agent = SpeedRLAgent(
        policy_client=policy,
        feature_contract=CONTRACT,
        speed_actor=actor,
        inference_mode="sync",
        tts_samples=1,
        left_limits=_base_limits(),
        right_limits=_base_limits(),
    )
    try:
        agent.act(_observation())
        assert policy.started.wait(timeout=1.0)
        agent.reset_execution()
        time.sleep(0.2)
        assert agent.diagnostics()["ready_sequence_id"] is None
        assert agent.trajectory_decisions == {}
        assert agent.diagnostics()["speed_rl_episode_epoch"] == 1
    finally:
        agent.teardown()
