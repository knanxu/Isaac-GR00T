# SPDX-License-Identifier: Apache-2.0
"""Compare FM with pre-drifting source, rather than with another new-code branch.

The default reference captures the cloud FM data/processor settings before drifting.
GROOT_FM_BASELINE_DIR can point to a read-only snapshot of an existing cloud run's
source, including its uncommitted training/data configuration changes.
"""

import ast
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from gr00t.configs import base_config
from gr00t.configs.model import gr00t_n1d7 as config_module
from gr00t.model.gr00t_n1d7 import gr00t_n1d7 as model_module
from gr00t.model.modules import qwen3_backbone
from omegaconf import OmegaConf
import pytest
from tests.gr00t.model.test_action_head import (
    _make_action_input,
    _make_backbone_output,
    _small_config,
)
from tests.gr00t.model.test_drifting_lora import tiny_qwen as tiny_qwen
import torch


def _source(path):
    directory = os.environ.get("GROOT_FM_BASELINE_DIR")
    if directory:
        return (Path(directory) / path).read_text()
    repo = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "show", f"df26a4fc9d99962a6509e90f6186b05e2907551d:{path}"], cwd=repo, text=True
    )


def _class_from_source(path, name, namespace):
    tree = ast.parse(_source(path))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def legacy_classes():
    legacy_config = _class_from_source(
        "gr00t/configs/model/gr00t_n1d7.py", "Gr00tN1d7Config", vars(config_module).copy()
    )
    namespace = vars(model_module).copy()
    namespace["Gr00tN1d7Config"] = legacy_config
    legacy_head = _class_from_source(
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py", "Gr00tN1d7ActionHead", namespace
    )
    legacy_model = _class_from_source(
        "gr00t/model/gr00t_n1d7/gr00t_n1d7.py", "Gr00tN1d7", namespace
    )
    legacy_run_config = _class_from_source(
        "gr00t/configs/base_config.py", "Config", vars(base_config).copy()
    )
    return legacy_config, legacy_head, legacy_model, legacy_run_config


@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_fm_numerics_rng_trainables_and_inference_unchanged(legacy_classes, alternate, bf16):
    legacy_config_cls, legacy_head_cls, _, _ = legacy_classes
    config = _small_config(
        use_alternate_vl_dit=alternate,
        state_dropout_prob=0.2,
        state_history_length=2,
        attn_dropout=0.2,
    )
    legacy_config = legacy_config_cls(**config.to_filtered_dict(exclude_augment=False))
    torch.manual_seed(251)
    before = legacy_head_cls(legacy_config)
    initial_rng = torch.get_rng_state()
    torch.manual_seed(251)
    after = model_module.Gr00tN1d7ActionHead(config)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert before.state_dict().keys() == after.state_dict().keys()
    assert [(n, p.requires_grad) for n, p in before.named_parameters()] == [
        (n, p.requires_grad) for n, p in after.named_parameters()
    ]
    for name, value in before.state_dict().items():
        torch.testing.assert_close(value, after.state_dict()[name], rtol=0, atol=0)
    inputs = _make_action_input(config)
    backbone = _make_backbone_output(config)
    backbone["backbone_attention_mask"] = backbone.backbone_attention_mask.bool()
    backbone.image_mask[:, :4] = False
    backbone.backbone_attention_mask[:, -1] = 0
    outputs, states, gradients = [], [], []
    for head in (before, after):
        head.train()
        torch.manual_seed(153)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            output = head(copy.deepcopy(backbone), copy.deepcopy(inputs))
        output["loss"].backward()
        outputs.append(output)
        states.append(torch.get_rng_state())
        gradients.append({n: p.grad for n, p in head.named_parameters()})
    assert torch.equal(*states)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=0, atol=0)
    for rtc in (False, True):
        inference_input = copy.deepcopy(inputs)
        options = None
        if rtc:
            options = dict(
                action_horizon=4, rtc_overlap_steps=3, rtc_frozen_steps=1, rtc_ramp_rate=2
            )
        else:
            del inference_input["action"]
        predictions = []
        for head in (before, after):
            head.eval()
            torch.manual_seed(871)
            predictions.append(
                head.get_action(copy.deepcopy(backbone), copy.deepcopy(inference_input), options)
            )
        for key in predictions[0]:
            torch.testing.assert_close(predictions[0][key], predictions[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("image_override", [False, True])
def test_fm_serialized_configs_match_pre_drifting(tmp_path, legacy_classes, image_override):
    legacy_config_cls, _, _, legacy_run_cls = legacy_classes
    kwargs = {}
    if image_override:
        kwargs = dict(
            shortest_image_edge=224,
            crop_fraction=1.0,
            vlm_min_pixels=50176,
            image_crop_size=None,
            image_target_size=None,
        )
    before = legacy_config_cls(**kwargs)
    after = config_module.Gr00tN1d7Config(**kwargs)
    # Explicit default CLI values must also disappear from FM artifacts.
    after.action_head_type = "flow_matching"
    after.drifting_lora_rank = 0
    after.drifting_lora_alpha = 32.0
    after.drifting_lora_dropout = 0.05
    assert before.to_dict() == after.to_dict()
    assert before.to_json_string() == after.to_json_string()
    assert before.to_filtered_json() == after.to_filtered_json()
    legacy_run = legacy_run_cls(model=before)
    new_run = base_config.Config(model=after)
    assert asdict(legacy_run) == new_run.to_dict()
    legacy_run.save(tmp_path / "before.yaml")
    new_run.save(tmp_path / "after.yaml")
    assert (tmp_path / "before.yaml").read_bytes() == (tmp_path / "after.yaml").read_bytes()
    assert OmegaConf.to_yaml(OmegaConf.create(legacy_run.__dict__)) == OmegaConf.to_yaml(
        OmegaConf.create(new_run.to_dict())
    )


def test_fm_real_checkpoint_layout_and_reload_identical(tmp_path, legacy_classes, monkeypatch):
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7
    from tests.gr00t.model.test_drifting_action_head import _TinyBackbone
    from transformers import AutoModel

    monkeypatch.setenv("GROOT_SKIP_HF_MODEL_WEIGHTS", "0")
    monkeypatch.setattr(processing_gr00t_n1d7, "Gr00tN1d7DataCollator", lambda **kwargs: None)
    old_config_cls, _, old_model_cls, _ = legacy_classes
    old_model_cls.__init__.__globals__["get_backbone_cls"] = lambda config: _TinyBackbone
    monkeypatch.setattr(model_module, "get_backbone_cls", lambda config: _TinyBackbone)
    config = _small_config()
    torch.manual_seed(12)
    original_kwargs = {
        k: v
        for k, v in vars(config).items()
        if k != "action_head_type" and not k.startswith("drifting_")
    }
    old = old_model_cls(old_config_cls(**original_kwargs))
    torch.manual_seed(12)
    new = model_module.Gr00tN1d7(config)
    old.save_pretrained(tmp_path / "old")
    new.save_pretrained(tmp_path / "new")
    old_files = {p.name: p.read_bytes() for p in (tmp_path / "old").iterdir()}
    new_files = {p.name: p.read_bytes() for p in (tmp_path / "new").iterdir()}
    assert old_files == new_files
    loaded, info = AutoModel.from_pretrained(tmp_path / "old", output_loading_info=True)
    assert not any(info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    assert loaded.config.action_head_type == "flow_matching"
    assert not any("lora_" in k for k in loaded.state_dict())
    assert not any(k.startswith("drifting_") for k in json.loads(new_files["config.json"]))
    for name, parameter in loaded.state_dict().items():
        torch.testing.assert_close(parameter, old.state_dict()[name], rtol=0, atol=0)


@pytest.mark.usefixtures("tiny_qwen")
def test_fm_actual_qwen_trainable_parameters_and_gradients_unchanged(legacy_classes):
    from tests.gr00t.model.test_drifting_lora import _inputs

    old_config_cls, _, old_model_cls, _ = legacy_classes
    old_backbone_cls = _class_from_source(
        "gr00t/model/modules/qwen3_backbone.py", "Qwen3Backbone", vars(qwen3_backbone).copy()
    )
    old_model_cls.__init__.__globals__["get_backbone_cls"] = lambda config: old_backbone_cls
    config = _small_config(
        select_layer=2,
        use_flash_attention=False,
        tune_visual=True,
        tune_llm=False,
        state_dropout_prob=0.2,
        use_alternate_vl_dit=True,
    )
    kwargs = {
        k: v
        for k, v in vars(config).items()
        if k != "action_head_type" and not k.startswith("drifting_")
    }
    torch.manual_seed(512)
    before = old_model_cls(old_config_cls(**kwargs))
    torch.manual_seed(512)
    after = model_module.Gr00tN1d7(config)
    assert before.state_dict().keys() == after.state_dict().keys()
    assert [(n, p.requires_grad) for n, p in before.named_parameters()] == [
        (n, p.requires_grad) for n, p in after.named_parameters()
    ]
    for key in before.state_dict():
        torch.testing.assert_close(
            before.state_dict()[key], after.state_dict()[key], rtol=0, atol=0
        )
    inputs = _inputs(config)
    outputs = []
    for model in (before, after):
        model.train()
        torch.manual_seed(521)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(copy.deepcopy(inputs))
        output["loss"].backward()
        outputs.append(output)
    torch.testing.assert_close(outputs[0]["loss"], outputs[1]["loss"], rtol=0, atol=0)
    after_parameters = dict(after.named_parameters())
    for name, parameter in before.named_parameters():
        torch.testing.assert_close(parameter.grad, after_parameters[name].grad, rtol=0, atol=0)
    assert not hasattr(after.backbone, "_drifting_lora_dropouts")
