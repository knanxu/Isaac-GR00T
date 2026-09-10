# SPDX-License-Identifier: Apache-2.0
"""Keep W&B curves aligned with optimizer steps in mixed Trainer/tqdm output."""

import pytest
from scripts.training.watch_drifting_wandb import parse_metrics


def test_eval_progress_does_not_replace_training_step():
    log = """
\r 5%|bar| 1000/20000 [1:20:00<24:00:00, 5.2s/it]
{'loss': 8.7, 'grad_norm': 9.0, 'learning_rate': 0.0001}
\r 100%|bar| 77/77 [00:12<00:00, 6.0it/s]\x1b[A
\x1b[A{'eval_loss': 8.9, 'eval_runtime': 14.0}
\r 5%|bar| 1000/20000 [1:20:14<24:00:00, 5.2s/it]
{'train_runtime': 5000, 'train_loss': 8.7}
"""
    assert parse_metrics(log, 20000) == [
        {
            "train/global_step": 1000,
            "train/loss": 8.7,
            "train/grad_norm": 9.0,
            "train/learning_rate": 0.0001,
        },
        {"train/global_step": 1000, "eval/loss": 8.9, "eval/runtime": 14.0},
    ]


def test_incomplete_append_is_only_logged_once_complete():
    log = "10/20000 [00:50<24:00:00, 5.0s/it]\r{'loss': 8."
    assert parse_metrics(log, 20000) == []
    assert parse_metrics(log + "5}", 20000) == [{"train/global_step": 10, "train/loss": 8.5}]


def test_missing_step_is_rejected_instead_of_fabricating_history():
    with pytest.raises(ValueError, match="preceding training step"):
        parse_metrics("{'loss': 8.5}", 20000)


def test_nonfinite_metrics_are_not_silently_dropped():
    with pytest.raises(ValueError, match="Non-finite"):
        parse_metrics("10/20000 [x]\r{'loss': 1e999}", 20000)
