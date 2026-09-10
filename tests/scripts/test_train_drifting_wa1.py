# SPDX-License-Identifier: Apache-2.0
"""Check recipe overrides preserve the actual FM dataset and preprocessing arguments."""

import copy
from pathlib import Path

import pytest
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
