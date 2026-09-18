# SPDX-License-Identifier: Apache-2.0
"""Action MSE semantics and aggregation, separate from drifting optimization."""

from unittest.mock import patch

from gr00t.experiment.drifting_trainer import DriftingTrainer
from gr00t.model.modules.drifting_loss import action_mse_statistics
import pytest
from tests.gr00t.model.test_action_head import _make_action_input, _make_backbone_output
from tests.gr00t.model.test_drifting_action_head import _drifting_head
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PretrainedConfig, TrainingArguments


def test_mse_excludes_padding_and_averages_each_chunk_before_the_batch():
    predictions = torch.tensor(
        [[[1.0, 2.0, float("nan")]], [[3.0, float("nan"), 1e6]], [[1e6, 1e6, 1e6]]],
        requires_grad=True,
    )
    expert = torch.zeros_like(predictions, requires_grad=True)
    mask = torch.tensor([[[1, 1, 0]], [[1, 0, 0]], [[0, 0, 0]]])
    statistics = action_mse_statistics(predictions, expert, mask)
    torch.testing.assert_close(statistics, torch.tensor([[2.5, 1.0], [9.0, 1.0], [0.0, 0.0]]))
    assert not statistics.requires_grad
    assert statistics[:, 0].sum() / statistics[:, 1].sum() == 5.75


def test_metric_uses_one_existing_prediction_not_a_sample_distribution():
    head, config = _drifting_head()
    inputs = _make_action_input(config)
    inputs.action.zero_()
    predictions = torch.full(
        (2, config.drifting_gen_per_label, config.action_horizon, config.max_action_dim), 100.0
    )
    predictions[0, 0] = 1.0
    predictions[1, 0] = 3.0
    with patch.object(head, "_predict_drifting", return_value=predictions.flatten(0, 1)) as predict:
        outputs = head(_make_backbone_output(config), inputs)
    assert predict.call_count == 1
    torch.testing.assert_close(
        outputs["action_mse_statistics"], torch.tensor([[1.0, 1.0], [9.0, 1.0]])
    )
    assert not outputs["action_mse_statistics"].requires_grad


class _EvaluationData(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return {"x": torch.tensor([[float(index + 1)]])}


class _EvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.config = PretrainedConfig(action_head_type="drifting")

    def forward(self, x):
        predicted = x + self.weight
        return {
            "loss": self.weight.square() + 7.0,
            "action_mse_statistics": action_mse_statistics(
                predicted, torch.zeros_like(x), torch.ones_like(x)
            ),
            "action_loss": torch.zeros_like(x),
            "action_mask": torch.ones_like(x),
            "backbone_features": torch.zeros_like(x),
            "state_features": torch.zeros_like(x),
        }


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("batch_metrics", [True, False])
def test_validation_mse_weights_short_last_batch_and_resets(tmp_path, batch_size, batch_metrics):
    trainer = DriftingTrainer(
        model=_EvaluationModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            disable_tqdm=True,
            per_device_eval_batch_size=batch_size,
            batch_eval_metrics=batch_metrics,
            prediction_loss_only=True,
            remove_unused_columns=False,
        ),
        eval_dataset=_EvaluationData(),
    )
    for _ in range(2):
        result = trainer.evaluate()
        assert result["eval_action_mse"] == pytest.approx(11.0)
        assert result["eval_loss"] == pytest.approx(7.0)


def test_training_log_mse_is_independent_of_optimization_loss(tmp_path):
    trainer = DriftingTrainer(
        model=_EvaluationModel(),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none"),
    )
    trainer.model.train()
    for values in [[1.0, 2.0], [3.0]]:
        x = torch.tensor(values).reshape(-1, 1, 1)
        loss = trainer.compute_loss(trainer.model, {"x": x})
        assert loss.item() == 7.0
        loss.backward()
        assert trainer.model.weight.grad.item() == 0.0  # MSE must not add its gradient.
    trainer.log({"loss": 7.0})
    assert trainer.state.log_history[-1]["action_mse"] == pytest.approx(14 / 3)
    assert trainer.state.log_history[-1]["loss"] == 7.0
    assert not trainer._train_action_mse.any()
