from __future__ import annotations

from dataclasses import dataclass
import math


SPEED_SCALES = (0.7, 1.0, 1.3, 1.6)
ACTION_COUNT = len(SPEED_SCALES)
ATOM_COUNT = 51
SUPPORT_MIN = 0.0
SUPPORT_MAX = 260.0


@dataclass(frozen=True)
class RainbowConfig:
    feature_dim: int
    hidden_dim: int = 256
    action_count: int = ACTION_COUNT
    atom_count: int = ATOM_COUNT
    support_min: float = SUPPORT_MIN
    support_max: float = SUPPORT_MAX
    gamma: float = 0.99
    learning_rate: float = 1e-4
    tau: float = 0.005
    max_grad_norm: float = 10.0
    batch_size: int = 64
    replay_capacity: int = 20_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 1_000
    n_step: int = 3

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim < 1 or self.action_count < 1 or self.atom_count < 2:
            raise ValueError("network dimensions must be positive")
        if not self.support_min < self.support_max:
            raise ValueError("support_min must be less than support_max")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.learning_rate <= 0.0 or self.tau <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("optimizer parameters must be positive")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if not 0.0 <= self.per_alpha <= 1.0:
            raise ValueError("per_alpha must be in [0, 1]")
        if not 0.0 <= self.per_beta_start <= 1.0 or self.per_beta_steps < 1:
            raise ValueError("invalid PER beta schedule")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        numeric = (
            self.support_min,
            self.support_max,
            self.gamma,
            self.learning_rate,
            self.tau,
            self.max_grad_norm,
            self.per_alpha,
            self.per_beta_start,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Rainbow parameters must be finite")

    def beta(self, activated_decisions: int) -> float:
        fraction = min(max(int(activated_decisions), 0) / self.per_beta_steps, 1.0)
        return self.per_beta_start + fraction * (1.0 - self.per_beta_start)


def epsilon_for_decisions(activated_decisions: int) -> float:
    fraction = min(max(int(activated_decisions), 0) / 1_000.0, 1.0)
    return 0.20 + fraction * (0.05 - 0.20)
