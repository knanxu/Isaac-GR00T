# SPDX-License-Identifier: Apache-2.0
"""Exercise actual Qwen3-VL + PEFT + DiT, including self-contained checkpoints."""

import copy
import json

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as model_module
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.modules import qwen3_backbone
from peft.tuners.lora import LoraLayer
import pytest
from tests.gr00t.model.test_action_head import _make_action_input, _small_config
import torch
from transformers import AutoModel, Qwen3VLConfig, Qwen3VLForConditionalGeneration


@pytest.fixture
def tiny_qwen(monkeypatch):
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=60,
        video_token_id=59,
        vision_start_token_id=61,
        vision_end_token_id=62,
    )
    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    monkeypatch.setattr(
        qwen3_backbone.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        lambda *args, **kwargs: Qwen3VLForConditionalGeneration(copy.deepcopy(config)),
    )
    monkeypatch.setattr(processing_gr00t_n1d7, "Gr00tN1d7DataCollator", lambda **kwargs: None)

    def create(**overrides):
        defaults = {"select_layer": 2, "use_flash_attention": False}
        return model_module.Gr00tN1d7(_small_config(**(defaults | overrides)))

    return create


def _inputs(config, batch_size=2):
    inputs = _make_action_input(config, batch_size).data
    inputs.update(
        input_ids=torch.tensor([[61, 60, 60, 60, 60, 62, 2]]).repeat(batch_size, 1),
        attention_mask=torch.ones(batch_size, 7, dtype=torch.long),
        pixel_values=torch.randn(batch_size * 16, 24),
        image_grid_thw=torch.tensor([[1, 4, 4]]).repeat(batch_size, 1),
    )
    return inputs


def _assert_backbone_frozen(model):
    for name, parameter in model.backbone.named_parameters():
        assert parameter.requires_grad == ("lora_" in name), name
    assert all(p.requires_grad for p in model.action_head.parameters())
    assert not any("lora_" in n for n, _ in model.action_head.named_parameters())


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("alternate", [False, True])
def test_actual_qwen_lora_gradients_and_dropout(tiny_qwen, bf16, alternate):
    model = tiny_qwen(
        action_head_type="drifting",
        drifting_lora_rank=2,
        drifting_lora_alpha=4,
        use_alternate_vl_dit=alternate,
    )
    _assert_backbone_frozen(model)
    model.train()
    inputs = _inputs(model.config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = {
        n: p.detach().clone() for n, p in model.backbone.named_parameters() if not p.requires_grad
    }
    for step in range(2):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            loss = model(copy.deepcopy(inputs))["loss"]
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.backbone.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                if step == 1:
                    assert parameter.grad.abs().sum() > 0, name
            else:
                assert parameter.grad is None, name
        optimizer.step()
    for name, parameter in model.backbone.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert not model.backbone.model.visual.training
    assert not model.backbone.model.language_model.training
    adapters = [m for m in model.backbone.modules() if isinstance(m, LoraLayer)]
    assert len(adapters) == 2 * 4 + 2 * 7
    assert all(m.lora_dropout.default.training for m in adapters)
    model.eval()
    assert all(not m.lora_dropout.default.training for m in adapters)


def test_plain_base_to_lora_and_full_checkpoint_reload(tmp_path, tiny_qwen):
    base = tiny_qwen()
    source = tmp_path / "base"
    base.save_pretrained(source)
    original_bytes = {p.name: p.read_bytes() for p in source.iterdir()}
    config = Config(
        model=_small_config(
            select_layer=2,
            use_flash_attention=False,
            action_head_type="drifting",
            drifting_lora_rank=2,
            drifting_lora_alpha=4,
        )
    )
    config.training.start_from_checkpoint = str(source)
    model = Gr00tN1d7Pipeline(config, tmp_path)._create_model()
    _assert_backbone_frozen(model)
    assert model.action_head.config is model.config
    assert model.config.drifting_lora_rank == 2
    for name, parameter in model.state_dict().items():
        if "lora_" not in name:
            torch.testing.assert_close(
                parameter, base.state_dict()[name.replace(".base_layer.", ".")], rtol=0, atol=0
            )
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            assert not parameter.any()  # Fresh adapters preserve the base output.
    inputs = _inputs(model.config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(copy.deepcopy(inputs))["loss"].backward()
    optimizer.step()
    checkpoint = tmp_path / "lora"
    model.save_pretrained(checkpoint)
    assert (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "adapter_model.safetensors").exists()
    restored, info = AutoModel.from_pretrained(checkpoint, output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not info[key]
    _assert_backbone_frozen(restored)
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, parameter in restored.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], rtol=0, atol=0)
    del inputs["action"]
    model.eval()
    restored.eval()
    torch.manual_seed(17)
    expected = model.get_action(copy.deepcopy(inputs)).action_pred
    torch.manual_seed(17)
    actual = restored.get_action(copy.deepcopy(inputs)).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored.bfloat16()
    assert torch.isfinite(restored.get_action(copy.deepcopy(inputs)).action_pred).all()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original_bytes
    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["drifting_lora_rank"] == 2
    config.training.start_from_checkpoint = str(checkpoint)
    from_pipeline = Gr00tN1d7Pipeline(config, tmp_path)._create_model()
    assert from_pipeline.state_dict().keys() == model.state_dict().keys()
    config.model.drifting_lora_rank = 0
    with pytest.raises(ValueError, match="LoRA checkpoint"):
        Gr00tN1d7Pipeline(config, tmp_path)._create_model()


@pytest.mark.parametrize(
    "overrides",
    [
        {"action_head_type": "flow_matching", "drifting_lora_rank": 2},
        {"drifting_lora_rank": -1},
        {"drifting_lora_rank": 1.5},
        {"drifting_lora_alpha": 0},
        {"drifting_lora_alpha": float("nan")},
        {"drifting_lora_dropout": 1},
        {"drifting_lora_dropout": float("nan")},
        {"drifting_lora_rank": 2, "tune_llm": True},
        {"drifting_lora_rank": 2, "tune_visual": True},
    ],
)
def test_lora_invalid_or_conflicting_config(overrides):
    with pytest.raises(ValueError):
        Gr00tN1d7Config(**({"action_head_type": "drifting"} | overrides))


class _QwenDataset(torch.utils.data.Dataset):
    seed = 0

    def __init__(self, config):
        self.samples = [_inputs(config, batch_size=1) for _ in range(4)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def reset_seed(self, seed):
        self.seed = seed


def _collate(samples):
    return {"inputs": {key: torch.cat([sample[key] for sample in samples]) for key in samples[0]}}


def test_lora_trainer_optimizer_checkpoint_and_resume(tmp_path, tiny_qwen):
    from gr00t.experiment.trainer import Gr00tTrainer
    from transformers import TrainingArguments

    model = tiny_qwen(action_head_type="drifting", drifting_lora_rank=2, drifting_lora_alpha=4)
    dataset = _QwenDataset(model.config)

    def trainer(model, steps):
        return Gr00tTrainer(
            model=model,
            train_dataset=dataset,
            data_collator=_collate,
            args=TrainingArguments(
                output_dir=str(tmp_path),
                max_steps=steps,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                dataloader_num_workers=0,
                save_steps=1,
                report_to="none",
                remove_unused_columns=False,
                use_cpu=True,
            ),
        )

    first = trainer(model, 2)
    first.train()
    checkpoint = tmp_path / "checkpoint-2"
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "scheduler.pt").exists()
    restored = AutoModel.from_pretrained(checkpoint)
    second = trainer(restored, 3)
    second.train(resume_from_checkpoint=str(checkpoint))
    assert second.state.global_step == 3
    _assert_backbone_frozen(restored)
    for parameter in restored.parameters():
        if parameter.requires_grad:
            assert second.optimizer.state[parameter]["step"] == 3
    incompatible = tiny_qwen(action_head_type="drifting", drifting_lora_rank=4)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer(incompatible, 4).train(resume_from_checkpoint=str(checkpoint))
