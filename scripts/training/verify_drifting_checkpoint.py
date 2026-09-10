#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reload the unmerged drift checkpoint and repeat the prior FM observation probes."""

import argparse
import json
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-reload", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    reference = json.loads(args.reference_reload.read_text())
    policy = Gr00tPolicy("naviai_wa1_head_lr_wf", str(args.checkpoint), device=args.device)
    config = policy.model.config
    expected = dict(
        action_head_type="drifting",
        drifting_lora_rank=16,
        drifting_lora_alpha=32,
        drifting_lora_dropout=0.05,
        drifting_gen_per_label=4,
        drifting_temperatures=[0.02, 0.05, 0.2],
        drifting_per_timestep_loss=True,
        action_horizon=40,
        state_dropout_prob=0.0,
    )
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise RuntimeError(f"Reloaded {key} differs from the confirmed recipe")
    if not any("lora_A" in name for name in policy.model.state_dict()):
        raise RuntimeError("Checkpoint has no unmerged LoRA weights")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    grid = policy.processor.processor.image_processor(images=[image], return_tensors="pt")[
        "image_grid_thw"
    ].tolist()
    collator_grid = policy.collate_fn.processor.image_processor(
        images=[image], return_tensors="pt"
    )["image_grid_thw"].tolist()
    if grid != [[1, 14, 14]] or collator_grid != grid or policy.processor.vlm_min_pixels != 50176:
        raise RuntimeError("Checkpoint processor did not retain 224x224 image processing")
    results = []
    for row in reference["robots"]:
        loader = LeRobotEpisodeLoader(
            Path(row["dataset"]), policy.modality_configs, decoder_kwargs={"num_ffmpeg_threads": 2}
        )
        episode, step = row["episode"], row["step"]
        table = loader._load_parquet_data(episode)
        video = loader._load_video_data(episode, np.array([step]))
        observation = {
            "video": {key: value[None] for key, value in video.items()},
            "state": {
                key: np.stack(table["state." + key].iloc[[step]])[None].astype(np.float32)
                for key in policy.modality_configs["state"].modality_keys
            },
            "language": {"annotation.human.task_description": [["Sort the parcels."]]},
        }
        calls = []
        handle = policy.model.action_head.model.register_forward_hook(
            lambda *unused: calls.append(1)
        )
        try:
            actions, _ = policy.get_action(observation)
        finally:
            handle.remove()
        if len(calls) != 1:
            raise RuntimeError(f"Expected one action-network call, got {len(calls)}")
        for name, shape in row["action_shapes"].items():
            if list(actions[name].shape) != shape or not np.isfinite(actions[name]).all():
                raise RuntimeError(f"Invalid {name} actions for {row['robot']}")
        results.append(
            {
                "robot": row["robot"],
                "episode": episode,
                "step": step,
                "all_finite": True,
                "action_network_calls": len(calls),
                "action_shapes": {key: list(value.shape) for key, value in actions.items()},
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "image_grid_thw": grid,
        "robots": results,
        "purpose": "Checkpoint reload and finite single-step actions; not robot task success",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
