#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the confirmed WA1 drift command from the recorded FM command.

Default: print a reviewable command without writing files or starting training.
--execute creates a new run directory, trains, and verifies the final checkpoint.
Run this script under nohup for a job independent of an SSH connection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


REPO = Path(__file__).resolve().parents[2]
RECIPE = {
    "--action-head-type": ["drifting"],
    "--drifting-gen-per-label": ["4"],
    "--drifting-temperatures": ["0.02", "0.05", "0.2"],
    "--drifting-per-timestep-loss": [],
    "--drifting-lora-rank": ["16"],
    "--drifting-lora-alpha": ["32"],
    "--drifting-lora-dropout": ["0.05"],
    "--global-batch-size": ["8"],
    "--gradient-accumulation-steps": ["8"],
    "--learning-rate": ["1e-4"],
    "--warmup-ratio": ["0.05"],
    "--weight-decay": ["1e-5"],
    "--state-dropout-prob": ["0"],
    "--no-tune-llm": [],
    "--no-tune-visual": [],
    "--tune-projector": [],
    "--tune-diffusion-model": [],
    "--no-save-only-model": [],
    "--no-use-wandb": [],
    "--no-resume-from-checkpoint": [],
    "--save-total-limit": ["3"],
}


def _options(argv):
    result = {}
    for token in argv:
        if token.startswith("--"):
            if token in result:
                raise ValueError(f"Duplicate option in FM command: {token}")
            result[token] = []
            current = token
        else:
            result[current].append(token)
    return result


def build_command(reference, run_dir, phase):
    argv = reference["argv"]
    entry = "gr00t/experiment/launch_finetune.py"
    index = argv.index(entry)
    launcher = argv[:index]
    if "torch.distributed.run" not in launcher or "--nproc_per_node=4" not in launcher:
        raise ValueError("Expected the recorded single-node four-GPU FM launcher")
    options = _options(argv[index + 1 :])
    if options.get("--action-head-type", ["flow_matching"]) != ["flow_matching"]:
        raise ValueError("The reference command must be a flow-matching run")
    if options.get("--embodiment-tag") != ["naviai_wa1_head_lr_wf"]:
        raise ValueError("This confirmed recipe is for naviai_wa1_head_lr_wf")
    if options.get("--num-gpus") != ["4"]:
        raise ValueError("The confirmed global batch requires four GPUs")
    old_output = Path(options["--output-dir"][0]).resolve()
    base = Path(options["--base-model-path"][0]).resolve()
    for protected in (old_output.parent, base):
        if run_dir.is_relative_to(protected) or protected.is_relative_to(run_dir):
            raise ValueError("Drifting output must be separate from the FM run and base weights")
    for flag in RECIPE:
        opposite = "--" + flag[5:] if flag.startswith("--no-") else "--no-" + flag[2:]
        options.pop(opposite, None)
    options.update(RECIPE)
    options["--output-dir"] = [str(run_dir / "train")]
    options["--max-steps"] = ["20" if phase == "smoke" else "20000"]
    options["--save-steps"] = ["10" if phase == "smoke" else "1000"]
    options["--eval-steps"] = ["10" if phase == "smoke" else "1000"]
    # Separate port from the original FM launcher. Smoke and full run are sequential.
    launcher = ["--master_port=29543" if a.startswith("--master_port=") else a for a in launcher]
    return (
        launcher + [entry] + [item for flag, values in options.items() for item in (flag, *values)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fm-command", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "train"], required=True)
    parser.add_argument("--smoke-run", type=Path, help="Required for executing the full run")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    reference = json.loads(args.fm_command.read_text())
    run_dir = args.run_dir.resolve()
    command = build_command(reference, run_dir, args.phase)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_hash = hashlib.sha256(args.fm_command.read_bytes()).hexdigest()
    if args.phase == "train":
        if args.smoke_run is None:
            raise ValueError(
                "Full training requires --smoke-run with a successful checkpoint reload"
            )
        smoke_status = json.loads((args.smoke_run / "train_status.json").read_text())
        smoke_command = json.loads((args.smoke_run / "train_command.json").read_text())
        if smoke_status.get("state") != "complete" or not smoke_status.get(
            "checkpoint_reload_passed"
        ):
            raise ValueError("The four-GPU smoke run has not passed checkpoint verification")
        if smoke_command["git_commit"] != commit or smoke_command["source_sha256"] != source_hash:
            raise ValueError("Smoke verification must use the same code and FM data command")

    options = _options(command[command.index("gr00t/experiment/launch_finetune.py") + 1 :])
    for key in ("--base-model-path", "--dataset-path", "--validation-dataset-path"):
        for path in options[key][0].split(os.pathsep):
            if not Path(path).is_dir():
                raise FileNotFoundError(path)
    reference_reload = args.fm_command.parent / "train_reload.json"
    if not reference_reload.is_file():
        raise FileNotFoundError(reference_reload)
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        PYTHONPATH=str(REPO),
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    )
    metadata = {
        "argv": command,
        "source_command": str(args.fm_command.resolve()),
        "source_sha256": source_hash,
        "base_revision": reference.get("base_revision"),
        "git_commit": commit,
        "environment": {
            k: environment[k]
            for k in (
                "PYTHONPATH",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTORCH_ALLOC_CONF",
                "OMP_NUM_THREADS",
            )
        },
    }
    (run_dir / "train_command.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    split = args.fm_command.parent / "data_split.json"
    if split.exists():
        shutil.copyfile(split, run_dir / "data_split.json")
    status = {"phase": args.phase, "state": "training", "started_unix": time.time()}

    def write_status():
        temporary = run_dir / "train_status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(run_dir / "train_status.json")

    write_status()
    try:
        with (run_dir / "train.log").open("w") as log:
            subprocess.run(
                command,
                cwd=REPO,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status["state"] = "verifying_checkpoint"
        write_status()
        checkpoint = run_dir / "train" / f"checkpoint-{options['--max-steps'][0]}"
        if not (checkpoint / "trainer_state.json").is_file():
            raise RuntimeError("Final resumable checkpoint is missing Trainer state")
        if not (checkpoint / "optimizer.pt").is_file() and not list(
            checkpoint.glob("**/*optim_states.pt")
        ):
            raise RuntimeError("Final checkpoint is missing optimizer state")
        with (run_dir / "reload.log").open("w") as log:
            subprocess.run(
                [
                    command[0],
                    "scripts/training/verify_drifting_checkpoint.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--reference-reload",
                    str(reference_reload),
                    "--report",
                    str(run_dir / "train_reload.json"),
                    "--device",
                    "cuda:3",
                ],
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status.update(state="complete", checkpoint=str(checkpoint), checkpoint_reload_passed=True)
    except BaseException as exc:
        status.update(state="failed", error=str(exc))
        raise
    finally:
        status["finished_unix"] = time.time()
        write_status()


if __name__ == "__main__":
    main()
