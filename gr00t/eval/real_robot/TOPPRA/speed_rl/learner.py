from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import random
import threading
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor

from .config import RainbowConfig
from .contract import FeatureContract
from .network import (
    DuelingC51Network,
    cpu_state_dict,
    load_cpu_state_dict,
    project_categorical_distribution,
    select_masked_action,
)
from .replay import PrioritizedReplayBuffer, Transition, build_n_step_transitions


class SpeedActor:
    """CPU actor whose weights remain frozen for one real-robot episode."""

    def __init__(
        self,
        config: RainbowConfig,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.network = DuelingC51Network(config).cpu().eval()
        self.policy_version = 0
        self._rng = random.Random(seed)

    def install_weights(self, state_dict: dict[str, Tensor], policy_version: int) -> None:
        load_cpu_state_dict(self.network, state_dict)
        self.network.eval()
        self.policy_version = int(policy_version)

    def select(
        self,
        feature: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float,
        fixed_action: int | None = None,
    ) -> int:
        if fixed_action is not None:
            action = int(fixed_action)
            if action < 0 or action >= self.config.action_count:
                raise ValueError("fixed_action is out of range")
            if not bool(np.asarray(action_mask)[action]):
                raise RuntimeError("The fixed calibration speed is currently infeasible")
            return action
        return select_masked_action(
            self.network,
            np.asarray(feature, dtype=np.float32),
            np.asarray(action_mask),
            epsilon=epsilon,
            rng=self._rng,
        )


class RainbowLearner:
    """Rainbow learner state owned by the spawned learner process."""

    def __init__(
        self,
        config: RainbowConfig,
        contract: FeatureContract,
        *,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.config = config
        self.contract = contract
        self.online = DuelingC51Network(config).cpu()
        self.target = DuelingC51Network(config).cpu()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = PrioritizedReplayBuffer(
            config.replay_capacity,
            alpha=config.per_alpha,
            seed=seed,
        )
        self.policy_version = 0
        self.update_count = 0
        self.activated_decisions = 0

    @property
    def online_training_ready(self) -> bool:
        counts = self.replay.action_counts
        return (
            len(self.replay) >= 256
            and len(counts) == self.config.action_count
            and min(counts) >= 32
        )

    def add_episode(self, transitions: Sequence[Transition]) -> int:
        converted = build_n_step_transitions(
            transitions,
            n_step=self.config.n_step,
            gamma=self.config.gamma,
        )
        self.replay.extend(converted)
        self.activated_decisions += len(transitions)
        return len(converted)

    def train_updates(
        self,
        requested_updates: int,
        *,
        require_online_gate: bool,
    ) -> dict[str, Any]:
        updates = max(0, int(requested_updates))
        if require_online_gate and not self.online_training_ready:
            raise RuntimeError(
                "Online training gate is closed: replay requires at least 256 transitions "
                "and 32 transitions for every speed"
            )
        if len(self.replay) < self.config.batch_size:
            return {"updates": 0, "mean_loss": None}

        losses: list[float] = []
        for _ in range(updates):
            beta = self.config.beta(self.activated_decisions)
            sample = self.replay.sample(self.config.batch_size, beta)
            batch = sample.transitions
            states = torch.from_numpy(np.stack([item.state for item in batch]))
            actions = torch.as_tensor([item.action for item in batch], dtype=torch.long)
            rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
            next_states = torch.from_numpy(np.stack([item.next_state for item in batch]))
            dones = torch.as_tensor([item.done for item in batch], dtype=torch.bool)
            next_masks = torch.from_numpy(
                np.stack([item.next_action_mask for item in batch])
            ).bool()
            discounts = torch.as_tensor([item.discount for item in batch], dtype=torch.float32)
            importance = torch.from_numpy(sample.weights)

            logits = self.online(states)
            chosen_logits = logits[torch.arange(len(batch)), actions]
            log_probabilities = torch.log_softmax(chosen_logits, dim=-1)
            with torch.no_grad():
                online_next_q = self.online.q_values(next_states)
                online_next_q = online_next_q.masked_fill(~next_masks, -torch.inf)
                next_actions = torch.argmax(online_next_q, dim=1)
                target_next_probabilities = self.target.probabilities(next_states)[
                    torch.arange(len(batch)), next_actions
                ]
                target_distribution = project_categorical_distribution(
                    target_next_probabilities,
                    rewards,
                    dones,
                    discounts,
                    self.online.support,
                )

            per_item_loss = -(target_distribution * log_probabilities).sum(dim=1)
            loss = torch.mean(importance * per_item_loss)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():
                for target_parameter, online_parameter in zip(
                    self.target.parameters(),
                    self.online.parameters(),
                    strict=True,
                ):
                    target_parameter.lerp_(online_parameter, self.config.tau)
            self.replay.update_priorities(
                sample.indices,
                per_item_loss.detach().cpu().numpy(),
            )
            self.update_count += 1
            self.policy_version += 1
            losses.append(float(loss.detach()))
        return {
            "updates": len(losses),
            "mean_loss": None if not losses else float(np.mean(losses)),
        }

    def actor_snapshot(self) -> dict[str, Any]:
        return {
            "state_dict": cpu_state_dict(self.online),
            "policy_version": self.policy_version,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "replay_size": len(self.replay),
            "action_counts": self.replay.action_counts,
            "policy_version": self.policy_version,
            "update_count": self.update_count,
            "activated_decisions": self.activated_decisions,
            "online_training_ready": self.online_training_ready,
            "beta": self.config.beta(self.activated_decisions),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "config": asdict(self.config),
            "contract": self.contract.to_dict(),
            "online": cpu_state_dict(self.online),
            "target": cpu_state_dict(self.target),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "policy_version": self.policy_version,
            "update_count": self.update_count,
            "activated_decisions": self.activated_decisions,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if int(state.get("format_version", -1)) != 1:
            raise ValueError("Unsupported Speed-RL checkpoint format")
        checkpoint_config = RainbowConfig(**state["config"])
        if checkpoint_config != self.config:
            raise ValueError("Speed-RL checkpoint Rainbow configuration does not match")
        self.contract.validate_compatible(FeatureContract.from_mapping(state["contract"]))
        load_cpu_state_dict(self.online, state["online"])
        load_cpu_state_dict(self.target, state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.policy_version = int(state["policy_version"])
        self.update_count = int(state["update_count"])
        self.activated_decisions = int(state["activated_decisions"])

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        torch.save(self.checkpoint_state(), temporary)
        temporary.replace(target)

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            state = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(Path(path), map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError("Speed-RL checkpoint root must be a mapping")
        self.load_checkpoint_state(state)


def _learner_worker(
    config: RainbowConfig,
    contract: FeatureContract,
    command_queue: Any,
    result_queue: Any,
    seed: int,
) -> None:
    torch.set_num_threads(1)
    learner = RainbowLearner(config, contract, seed=seed)
    while True:
        command = command_queue.get()
        request_id = command["request_id"]
        operation = command["operation"]
        try:
            if operation == "stop":
                result_queue.put({"request_id": request_id, "ok": True, "result": None})
                return
            if operation == "add_episode":
                result = learner.add_episode(command["transitions"])
            elif operation == "train":
                result = learner.train_updates(
                    command["updates"],
                    require_online_gate=command["require_online_gate"],
                )
            elif operation == "snapshot":
                result = learner.actor_snapshot()
            elif operation == "stats":
                result = learner.stats()
            elif operation == "save":
                learner.save_checkpoint(command["path"])
                result = learner.stats()
            elif operation == "load":
                learner.load_checkpoint(command["path"])
                result = learner.stats()
            else:
                raise ValueError(f"Unknown learner operation {operation!r}")
            result_queue.put({"request_id": request_id, "ok": True, "result": result})
        except BaseException as exc:
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class LearnerProcess:
    """Request/reply interface to a dedicated multiprocessing-spawn learner."""

    def __init__(
        self,
        config: RainbowConfig,
        contract: FeatureContract,
        *,
        seed: int = 0,
        response_timeout_s: float = 120.0,
    ) -> None:
        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._results = context.Queue()
        self._process = context.Process(
            target=_learner_worker,
            args=(config, contract, self._commands, self._results, int(seed)),
            name="SpeedRLRainbowLearner",
            daemon=True,
        )
        self._process.start()
        self._timeout_s = float(response_timeout_s)
        self._lock = threading.Lock()
        self._next_request_id = 0
        self._closed = False

    def _call(self, operation: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("Speed-RL learner process is closed")
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            self._commands.put({"request_id": request_id, "operation": operation, **payload})
            try:
                response = self._results.get(timeout=self._timeout_s)
            except Empty as exc:
                if not self._process.is_alive():
                    raise RuntimeError("Speed-RL learner process exited unexpectedly") from exc
                raise TimeoutError(
                    f"Timed out waiting for learner operation {operation!r}"
                ) from exc
            if response.get("request_id") != request_id:
                raise RuntimeError("Speed-RL learner response order was corrupted")
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Speed-RL learner operation failed"))
            return response.get("result")

    def add_episode(self, transitions: Sequence[Transition]) -> int:
        return int(self._call("add_episode", transitions=tuple(transitions)))

    def train(self, updates: int, *, require_online_gate: bool = True) -> dict[str, Any]:
        return dict(
            self._call(
                "train",
                updates=int(updates),
                require_online_gate=bool(require_online_gate),
            )
        )

    def install_actor_weights(self, actor: SpeedActor) -> int:
        snapshot = self._call("snapshot")
        actor.install_weights(snapshot["state_dict"], snapshot["policy_version"])
        return actor.policy_version

    def stats(self) -> dict[str, Any]:
        return dict(self._call("stats"))

    def save(self, path: str | Path) -> dict[str, Any]:
        return dict(self._call("save", path=str(path)))

    def load(self, path: str | Path) -> dict[str, Any]:
        return dict(self._call("load", path=str(path)))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._call("stop")
        finally:
            self._closed = True
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._commands.close()
            self._results.close()

    def __enter__(self) -> LearnerProcess:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
