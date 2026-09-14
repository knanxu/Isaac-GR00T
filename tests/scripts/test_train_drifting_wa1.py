# SPDX-License-Identifier: Apache-2.0
"""Check recipe overrides preserve the actual FM dataset and preprocessing arguments."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from scripts.training import train_drifting_wa1
from scripts.training.train_drifting_wa1 import RECIPE, _options, build_command


def _reference():
    return {
        "argv": [
            "/old/.venv/bin/python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=4",
            "--master_port=29542",
            "gr00t/experiment/launch_finetune.py",
            "--base-model-path",
            "/base",
            "--dataset-path",
            "/train/a:/train/b",
            "--validation-dataset-path",
            "/validation/c",
            "--embodiment-tag",
            "naviai_wa1_head_lr_wf",
            "--modality-config-path",
            "examples/naviai_wa1_head_lr_wf/modality_config.py",
            "--output-dir",
            "/old-fm/train",
            "--num-gpus",
            "4",
            "--global-batch-size",
            "64",
            "--gradient-accumulation-steps",
            "1",
            "--tune-visual",
            "--no-tune-llm",
            "--state-dropout-prob",
            "0.2",
            "--shortest-image-edge",
            "224",
            "--crop-fraction",
            "1.0",
            "--vlm-min-pixels",
            "50176",
            "--color-jitter-params",
            "brightness",
            "0.3",
            "contrast",
            "0.4",
            "--shard-size",
            "256",
            "--episode-sampling-rate",
            "0.1",
            "--num-shards-per-epoch",
            "100000",
        ]
    }


@pytest.mark.parametrize("phase,steps,save", [("smoke", "20", "10"), ("train", "20000", "1000")])
def test_recipe_preserves_fm_inputs_and_effective_batch(phase, steps, save):
    reference = _reference()
    original = copy.deepcopy(reference)
    command = build_command(reference, Path("/new-drifting"), phase)
    entry = command.index("gr00t/experiment/launch_finetune.py")
    actual = _options(command[entry + 1 :])
    previous = _options(reference["argv"][entry + 1 :])
    for key in (
        "--base-model-path",
        "--dataset-path",
        "--validation-dataset-path",
        "--modality-config-path",
        "--shortest-image-edge",
        "--crop-fraction",
        "--vlm-min-pixels",
        "--color-jitter-params",
        "--shard-size",
        "--episode-sampling-rate",
        "--num-shards-per-epoch",
    ):
        assert actual[key] == previous[key]
    assert reference == original
    assert all(actual[key] == value for key, value in RECIPE.items())
    assert "--tune-visual" not in actual
    assert (
        int(actual["--global-batch-size"][0]) * int(actual["--gradient-accumulation-steps"][0])
        == 64
    )
    assert actual["--max-steps"] == [steps]
    assert actual["--save-steps"] == [save]
    assert actual["--output-dir"] == ["/new-drifting/train"]
    assert "--master_port=29543" in command


@pytest.mark.parametrize(
    "path", ["/old-fm", "/old-fm/train", "/old-fm/new", "/base", "/base/new", "/"]
)
def test_rejects_output_overlapping_fm_or_base(path):
    with pytest.raises(ValueError, match="separate"):
        build_command(_reference(), Path(path), "train")


@pytest.mark.parametrize("phase", ["smoke", "train"])
def test_full_fm_scope_has_no_adapters_or_accumulation_and_preserves_inputs(phase):
    reference = _reference()
    original = copy.deepcopy(reference)
    command = build_command(reference, Path("/new-drifting"), phase, "full-fm-scope")
    entry = command.index("gr00t/experiment/launch_finetune.py")
    actual = _options(command[entry + 1 :])
    baseline = _options(build_command(reference, Path("/new-drifting"), phase)[entry + 1 :])
    changed = {
        key for key in actual.keys() | baseline.keys() if actual.get(key) != baseline.get(key)
    }
    assert changed == {
        "--drifting-lora-rank",
        "--drifting-lora-dropout",
        "--global-batch-size",
        "--gradient-accumulation-steps",
        "--save-total-limit",
        "--no-tune-visual",
        "--tune-visual",
    }
    assert actual["--drifting-lora-rank"] == ["0"]
    assert actual["--gradient-accumulation-steps"] == ["1"]
    previous = _options(reference["argv"][entry + 1 :])
    assert actual["--global-batch-size"] == previous["--global-batch-size"] == ["64"]
    assert int(actual["--global-batch-size"][0]) // int(actual["--num-gpus"][0]) == 16
    assert actual["--save-total-limit"] == ["1"]
    assert "--no-save-only-model" in actual
    assert "--tune-visual" in actual
    assert "--no-tune-llm" in actual
    assert "--tune-llm" not in actual
    assert reference == original


def test_successful_lora_smoke_cannot_authorize_full_weight_run(tmp_path, monkeypatch):
    reference = _reference()
    source = tmp_path / "fm_command.json"
    source.write_text(json.dumps(reference))
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    (smoke / "train_status.json").write_text(
        json.dumps({"state": "complete", "checkpoint_reload_passed": True})
    )
    (smoke / "train_command.json").write_text(
        json.dumps(
            {
                "git_commit": "same-commit",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "argv": build_command(reference, smoke, "smoke", "lora16"),
            }
        )
    )
    run = tmp_path / "train"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_drifting_wa1.py",
            "--fm-command",
            str(source),
            "--run-dir",
            str(run),
            "--phase",
            "train",
            "--recipe",
            "full-fm-scope",
            "--smoke-run",
            str(smoke),
            "--execute",
        ],
    )
    monkeypatch.setattr(
        train_drifting_wa1.subprocess, "check_output", lambda *a, **kw: "same-commit"
    )
    with pytest.raises(ValueError, match="same recipe"):
        train_drifting_wa1.main()
    assert not run.exists()


def test_full_weight_dry_run_does_not_create_directory(tmp_path, monkeypatch, capsys):
    source = tmp_path / "fm_command.json"
    source.write_text(json.dumps(_reference()))
    run = tmp_path / "new-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_drifting_wa1.py",
            "--fm-command",
            str(source),
            "--run-dir",
            str(run),
            "--phase",
            "train",
            "--recipe",
            "full-fm-scope",
        ],
    )
    train_drifting_wa1.main()
    command = capsys.readouterr().out
    assert "--drifting-lora-rank 0" in command
    assert "--gradient-accumulation-steps 1" in command
    assert "--global-batch-size 64" in command
    assert "--save-total-limit 1" in command
    assert not run.exists()
