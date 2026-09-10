# SPDX-License-Identifier: Apache-2.0
"""CPU tests for drifting semantics, masks, gradients, and checkpoint compatibility."""

import copy
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as model_module
from gr00t.model.modules.drifting_loss import drifting_loss
import pytest
from tests.gr00t.model.test_action_head import (
    _make_action_input,
    _make_backbone_output,
    _small_config,
)
import torch
from torch import nn
from transformers import AutoModel
from transformers.feature_extraction_utils import BatchFeature


def _drifting_head(**overrides):
    config = _small_config(action_head_type="drifting", **overrides)
    return model_module.Gr00tN1d7ActionHead(config), config


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("per_timestep", [False, True])
def test_drifting_backward_and_masked_targets(alternate, per_timestep):
    head, config = _drifting_head(
        use_alternate_vl_dit=alternate,
        drifting_per_timestep_loss=per_timestep,
        state_history_length=2,
    )
    inputs = _make_action_input(config)
    inputs.action_mask[:, -1] = 0
    inputs.action_mask[:, :, -2:] = 0
    changed = copy.deepcopy(inputs)
    changed.action[~inputs.action_mask.bool()] = 1e6
    backbone = _make_backbone_output(config)
    backbone["backbone_attention_mask"] = backbone.backbone_attention_mask.bool()
    backbone.image_mask[:, :4] = False
    backbone.backbone_features.requires_grad_()
    torch.manual_seed(4)
    training_backbone = copy.deepcopy(backbone)
    vl_features = training_backbone.backbone_features
    result = head(training_backbone, copy.deepcopy(inputs))
    torch.manual_seed(4)
    repeated = head(copy.deepcopy(backbone), changed)
    torch.testing.assert_close(result["loss"], repeated["loss"], rtol=0, atol=0)
    assert result["action_loss"].shape == inputs.action.shape
    assert not result["action_loss"][~inputs.action_mask.bool()].any()
    assert torch.isfinite(result["loss"])
    assert result["loss"].item() > 0
    result["loss"].backward()
    assert vl_features.grad is not None
    assert vl_features.grad.abs().sum() > 0
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert head.action_decoder.layer2.W.grad.abs().sum() > 0


@pytest.mark.parametrize("alternate", [False, True])
def test_drifting_inference_is_one_direct_prediction(alternate):
    head, config = _drifting_head(use_alternate_vl_dit=alternate, num_inference_timesteps=7)
    head.eval()
    inputs = _make_action_input(config)
    del inputs["action"]
    backbone = _make_backbone_output(config)
    backbone["backbone_attention_mask"] = backbone.backbone_attention_mask.bool()
    backbone.image_mask[:, :4] = False
    # A constant decoder output distinguishes direct prediction from noise + velocity.
    with (
        patch.object(head.model, "forward", wraps=head.model.forward) as dit_forward,
        patch.object(
            head.action_decoder,
            "forward",
            side_effect=lambda features, ids: features.new_full(
                (features.shape[0], features.shape[1], config.max_action_dim), 0.125
            ),
        ),
    ):
        output = head.get_action(backbone, inputs)
    assert dit_forward.call_count == 1
    assert not dit_forward.call_args.kwargs["timestep"].any()
    torch.testing.assert_close(
        output.action_pred,
        torch.full_like(inputs.state[:, :1], 0.125).expand(
            2, config.action_horizon, config.max_action_dim
        ),
    )
    assert not output.action_pred.requires_grad


def test_drifting_rejects_flow_matching_rtc():
    head, config = _drifting_head()
    with pytest.raises(ValueError, match="RTC"):
        head.get_action(_make_backbone_output(config), _make_action_input(config))


@pytest.mark.parametrize("alternate", [False, True])
def test_bf16_action_head_training_and_inference(alternate):
    head, config = _drifting_head(use_alternate_vl_dit=alternate)
    inputs = _make_action_input(config)
    backbone = _make_backbone_output(config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = head(copy.deepcopy(backbone), copy.deepcopy(inputs))["loss"]
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(head.action_decoder.layer2.W.grad).all()
    head.eval().bfloat16()
    inputs["state"] = inputs.state.bfloat16()
    del inputs["action"]
    backbone["backbone_features"] = backbone.backbone_features.bfloat16()
    output = head.get_action(backbone, inputs)
    assert output.action_pred.dtype == torch.bfloat16
    assert torch.isfinite(output.action_pred).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_drifting_loss_padding_invariance_and_gradients(bf16):
    torch.manual_seed(17)
    dtype = torch.bfloat16 if bf16 else torch.float32
    gen = torch.randn(2, 4, 3, dtype=dtype).requires_grad_()
    pos = torch.randn(2, 1, 3, dtype=dtype).requires_grad_()
    temperatures = [0.02, 0.05, 0.2]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        error = drifting_loss(gen, pos, torch.ones(2, 3), temperatures)
        padded_gen = torch.cat((gen.detach(), torch.full((2, 4, 7), 1e5)), dim=-1)
        padded_gen.requires_grad_()
        padded_pos = torch.cat((pos.detach(), torch.full((2, 1, 7), -1e5)), dim=-1)
        padded_mask = torch.cat((torch.ones(2, 3), torch.zeros(2, 7)), dim=-1)
        padded_error = drifting_loss(padded_gen, padded_pos, padded_mask, temperatures)
    assert error.dtype == torch.float32
    torch.testing.assert_close(error, padded_error[:, :3])
    error.sum().backward()
    padded_error.sum().backward()
    torch.testing.assert_close(gen.grad.float(), padded_gen.grad[:, :, :3], rtol=0.01, atol=1e-4)
    assert pos.grad is None  # Targets must be detached.
    assert not padded_gen.grad[:, :, 3:].any()


def test_fully_masked_rows_do_not_change_valid_loss():
    gen = torch.randn(1, 4, 3, requires_grad=True)
    pos = torch.randn(1, 1, 3)
    baseline = drifting_loss(gen, pos, torch.ones(1, 3), [0.2])
    batched = drifting_loss(
        torch.cat((gen, torch.zeros_like(gen))),
        torch.cat((pos, torch.zeros_like(pos))),
        torch.tensor([[1, 1, 1], [0, 0, 0]]),
        [0.2],
    )
    torch.testing.assert_close(baseline, batched[:1])
    assert not batched[1].any()
    empty = drifting_loss(gen, pos, torch.zeros(1, 3), [0.2])
    empty.sum().backward()
    assert not empty.any()
    assert not gen.grad.any()


def test_drifting_force_attracts_samples_to_demonstration():
    gen = torch.tensor([[[-1.0], [1.0]]], requires_grad=True)
    positive = torch.tensor([[[5.0]]])
    drifting_loss(gen, positive, torch.ones(1, 1), [0.2]).sum().backward()
    assert gen.grad.sum() < 0  # Gradient descent moves the centroid toward +5.


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_head_type": "typo"},
        {"drifting_gen_per_label": 1},
        {"drifting_temperatures": []},
        {"drifting_temperatures": [0.0]},
        {"drifting_temperatures": [float("nan")]},
    ],
)
def test_invalid_head_config(overrides):
    with pytest.raises(ValueError):
        Gr00tN1d7Config(**({"action_head_type": "drifting"} | overrides))


class _TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def prepare_input(self, inputs):
        return inputs

    def forward(self, inputs):
        features = inputs["test_vl"] * self.weight
        return BatchFeature(
            data={
                "backbone_features": features,
                "backbone_attention_mask": torch.ones(
                    features.shape[:2], device=features.device, dtype=torch.bool
                ),
                "image_mask": torch.zeros(
                    features.shape[:2], device=features.device, dtype=torch.bool
                ),
            }
        )


@pytest.fixture
def tiny_full_model(monkeypatch):
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7

    monkeypatch.setattr(model_module, "get_backbone_cls", lambda config: _TinyBackbone)
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    monkeypatch.setattr(processing_gr00t_n1d7, "Gr00tN1d7DataCollator", lambda **kwargs: None)
    return model_module.Gr00tN1d7(_small_config())


def test_hf_old_checkpoint_load_override_and_drifting_roundtrip(tmp_path, tiny_full_model):
    legacy = tmp_path / "legacy"
    tiny_full_model.save_pretrained(legacy)
    config_path = legacy / "config.json"
    old_config = json.loads(config_path.read_text())
    for key in list(old_config):
        if key == "action_head_type" or key.startswith("drifting_"):
            del old_config[key]
    config_path.write_text(json.dumps(old_config))
    fm, fm_info = AutoModel.from_pretrained(legacy, output_loading_info=True)
    drift, drift_info = AutoModel.from_pretrained(
        legacy,
        action_head_type="drifting",
        drifting_gen_per_label=3,
        output_loading_info=True,
    )
    for info in (fm_info, drift_info):
        assert not info["missing_keys"]
        assert not info["unexpected_keys"]
        assert not info["mismatched_keys"]
    assert fm.config.action_head_type == "flow_matching"
    assert drift.config.action_head_type == "drifting"
    assert drift.action_head.config is drift.config
    assert drift.config.drifting_gen_per_label == 3
    for key, value in tiny_full_model.state_dict().items():
        torch.testing.assert_close(fm.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(drift.state_dict()[key], value, rtol=0, atol=0)
    assert fm.state_dict().keys() == drift.state_dict().keys()
    drift.save_pretrained(tmp_path / "drift")
    reloaded = AutoModel.from_pretrained(tmp_path / "drift")
    assert reloaded.config.action_head_type == "drifting"
    assert reloaded.config.drifting_gen_per_label == 3
    for model in (drift, reloaded):
        model.eval()
    inputs = _make_action_input(drift.config)
    del inputs["action"]
    backbone = _make_backbone_output(drift.config)
    torch.manual_seed(7)
    before = drift.action_head.get_action(
        copy.deepcopy(backbone), copy.deepcopy(inputs)
    ).action_pred
    torch.manual_seed(7)
    after = reloaded.action_head.get_action(backbone, inputs).action_pred
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_training_pipeline_applies_head_override(tmp_path, tiny_full_model):
    from gr00t.configs.base_config import Config
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    source = tmp_path / "flow_matching"
    tiny_full_model.save_pretrained(source)
    original_config = (source / "config.json").read_bytes()
    run_config = Config(
        model=_small_config(
            action_head_type="drifting",
            drifting_gen_per_label=2,
            drifting_per_timestep_loss=False,
            drifting_temperatures=[0.1, 0.3],
        )
    )
    run_config.training.start_from_checkpoint = str(source)
    model = Gr00tN1d7Pipeline(run_config, tmp_path)._create_model()
    assert model.action_head.config.action_head_type == "drifting"
    assert model.config.drifting_gen_per_label == 2
    assert model.config.drifting_temperatures == [0.1, 0.3]
    assert model.config.drifting_per_timestep_loss is False
    saved_config = json.loads((tmp_path / "final_model_config.json").read_text())
    assert saved_config["action_head_type"] == "drifting"
    assert (source / "config.json").read_bytes() == original_config


class _SyntheticDataset(torch.utils.data.Dataset):
    seed = 0

    def __init__(self, config):
        self.inputs = _make_action_input(config, batch_size=4).data
        self.inputs["test_vl"] = torch.randn(4, 8, config.backbone_embedding_dim)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"inputs": {key: value[index] for key, value in self.inputs.items()}}

    def reset_seed(self, seed):
        self.seed = seed


@pytest.mark.parametrize("head_type", ["flow_matching", "drifting"])
def test_trainer_save_resume_and_reject_objective_switch(tmp_path, tiny_full_model, head_type):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import TrainingArguments

    model = tiny_full_model
    model.config.action_head_type = head_type
    dataset = _SyntheticDataset(model.config)

    def make_trainer(steps):
        return Gr00tTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=torch.utils.data.default_collate,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=steps,
                per_device_train_batch_size=2,
                dataloader_num_workers=0,
                save_steps=1,
                report_to="none",
                remove_unused_columns=False,
                use_cpu=True,
            ),
        )

    before = model.action_head.action_decoder.layer2.W.detach().clone()
    trainer = make_trainer(2)
    trainer.train()
    assert trainer.state.global_step == 2
    assert not torch.equal(before, model.action_head.action_decoder.layer2.W)
    checkpoint = tmp_path / "checkpoint-2"
    for name in (
        "model.safetensors",
        "config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    ):
        assert (checkpoint / name).exists()
    # Legacy flow matching checkpoints have no selector field.
    if head_type == "flow_matching":
        config_path = checkpoint / "config.json"
        config_dict = json.loads(config_path.read_text())
        config_dict.pop("action_head_type", None)
        config_path.write_text(json.dumps(config_dict))
    resumed = make_trainer(3)
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == 3
    assert dataset.seed == 2
    model.config.action_head_type = "drifting" if head_type == "flow_matching" else "flow_matching"
    with pytest.raises(ValueError, match="Cannot resume"):
        make_trainer(4).train(resume_from_checkpoint=str(checkpoint))


@pytest.mark.parametrize("drifting", [False, True])
def test_finetune_cli_selects_objective(monkeypatch, drifting):
    from gr00t.experiment import experiment

    captured = []
    monkeypatch.setattr(experiment, "run", captured.append)
    args = [
        "launch_finetune.py",
        "--base-model-path",
        "/unused/model",
        "--dataset-path",
        "/unused/data",
        "--embodiment-tag",
        "NEW_EMBODIMENT",
    ]
    if drifting:
        args += [
            "--action-head-type",
            "drifting",
            "--drifting-gen-per-label",
            "3",
            "--drifting-temperatures",
            "0.1",
            "0.3",
            "--no-drifting-per-timestep-loss",
            "--drifting-lora-rank",
            "2",
            "--drifting-lora-alpha",
            "4",
            "--drifting-lora-dropout",
            "0.1",
        ]
    monkeypatch.setattr(sys, "argv", args)
    script = Path(__file__).resolve().parents[3] / "gr00t/experiment/launch_finetune.py"
    runpy.run_path(str(script), run_name="__main__")
    config = captured[0]
    assert config.model.action_head_type == ("drifting" if drifting else "flow_matching")
    assert config.training.resume_from_checkpoint is False
    if drifting:
        assert config.model.drifting_gen_per_label == 3
        assert config.model.drifting_temperatures == [0.1, 0.3]
        assert config.model.drifting_per_timestep_loss is False
        assert config.model.drifting_lora_rank == 2
        assert config.model.drifting_lora_alpha == 4
        assert config.model.drifting_lora_dropout == 0.1


@pytest.mark.parametrize("head_type", ["flow_matching", "drifting"])
def test_checkpoint_and_run_metadata_are_scoped_to_drifting(tmp_path, tiny_full_model, head_type):
    from gr00t.configs.base_config import Config
    from gr00t.experiment.experiment import save_run_config_artifacts
    from wandb.util import json_friendly_val
    import yaml

    config = tiny_full_model.config
    # Match explicit launcher overrides, including when using the default FM mode.
    config.action_head_type = head_type
    config.drifting_gen_per_label = 3
    config.drifting_temperatures = [0.1, 0.3]
    config.drifting_per_timestep_loss = False
    checkpoint = tmp_path / "checkpoint"
    tiny_full_model.save_pretrained(checkpoint)
    run_config = Config(model=config)
    config_dir = tmp_path / "experiment_cfg"
    save_run_config_artifacts(config_dir, tmp_path, run_config, "test")

    serialized = [
        json.loads((checkpoint / "config.json").read_text()),
        json.loads(config.to_filtered_json()),
        yaml.safe_load((config_dir / "config.yaml").read_text())["model"],
        yaml.safe_load((config_dir / "conf.yaml").read_text())["model"],
        json_friendly_val(run_config.to_dict())["model"],
    ]
    for model_config in serialized:
        if head_type == "flow_matching":
            assert "action_head_type" not in model_config
            assert not any(key.startswith("drifting_") for key in model_config)
        else:
            assert model_config["action_head_type"] == "drifting"
            assert model_config["drifting_gen_per_label"] == 3
            assert model_config["drifting_temperatures"] == [0.1, 0.3]
            assert model_config["drifting_per_timestep_loss"] is False
    restored = Config.from_pretrained(config_dir / "config.yaml")
    assert restored.model.action_head_type == head_type
