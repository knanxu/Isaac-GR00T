# SPDX-License-Identifier: Apache-2.0
"""Drift-only action MSE logging; the optimization objective is unchanged."""

import torch

from gr00t.experiment.trainer import Gr00tTrainer


class _ActionMSEMetric:
    """Accumulate per-example errors after HF gathers/trims evaluation batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.error_sum = 0.0
        self.valid_examples = 0.0

    def __call__(self, prediction, compute_result=True):
        self.error_sum += torch.as_tensor(prediction.predictions).double().sum().item()
        self.valid_examples += torch.as_tensor(prediction.label_ids).double().sum().item()
        if not compute_result:
            return {}
        result = {"action_mse": self.error_sum / max(self.valid_examples, 1.0)}
        self.reset()
        return result


class DriftingTrainer(Gr00tTrainer):
    """Report one predicted action chunk's MSE against its paired expert chunk."""

    def __init__(self, *args, **kwargs):
        self._action_mse_metric = _ActionMSEMetric()
        self._train_action_mse = None
        kwargs["compute_metrics"] = self._action_mse_metric
        super().__init__(*args, **kwargs)
        if getattr(self.model.config, "action_head_type", None) != "drifting":
            raise ValueError("DriftingTrainer requires an explicitly enabled drifting action head")
        self.can_return_loss = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if model.training:
            statistics = outputs["action_mse_statistics"].detach().double().sum(dim=0)
            if self._train_action_mse is None:
                self._train_action_mse = statistics
            else:
                self._train_action_mse += statistics
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs and self._train_action_mse is not None:
            totals = self._nested_gather(self._train_action_mse).reshape(-1, 2).sum(dim=0)
            logs = {**logs, "action_mse": (totals[0] / totals[1].clamp_min(1)).item()}
            self._train_action_mse.zero_()
        super().log(logs, start_time=start_time)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Gather only two scalars per example, not actions or backbone features.
        ignored = list(ignore_keys or []) + [
            "action_loss",
            "action_mask",
            "backbone_features",
            "state_features",
        ]
        loss, statistics, _ = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignored
        )
        # HF's metric path expects predictions and labels. Here they carry the
        # per-example MSE and valid-example indicator, respectively.
        return loss, statistics[:, :1], statistics[:, 1:]

    def evaluation_loop(self, *args, **kwargs):
        self._action_mse_metric.reset()
        return super().evaluation_loop(*args, **kwargs)
