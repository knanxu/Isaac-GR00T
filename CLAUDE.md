# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.12 (dGPU, Thor, DGX Spark); Python 3.10 (Orin — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Drifting integration: preserve the existing flow matching workflow

User requirement (2026-09-09): adding or extending the drifting action head must
not change the existing flow matching (FM) training workflow. This is a hard
compatibility constraint, including the user's FM policies trained in the cloud.

- Existing FM commands must run as before without adding flags. FM stays the
  default, including when an old checkpoint has no action-head selector.
- Preserve FM losses, random sampling, trainable parameters, data processing,
  optimizer/scheduler, precision, distributed settings, save/resume behavior,
  checkpoint filenames/layout, weight keys/shapes, and serialized model/config
  fields. Do not write inactive drifting/LoRA metadata into FM artifacts.
- New drift/LoRA behavior must be explicitly enabled and scoped to that path.
  Do not copy the reference fork's global DDP, resume, or LoRA changes into FM.
- Never overwrite or convert the user's existing FM checkpoints. A drift run
  uses its own output directory. FM weights may initialize a new drift run;
  optimizer-state resume across objectives is a different operation and is rejected.
- LoRA resumable checkpoints and merged deployment weights must be distinguished;
  new drift artifacts do not justify changing the original FM artifact contract.
- Verify FM compatibility against the pre-drifting implementation (numerical
  behavior and serialized artifacts), alongside drift tests. CPU smoke tests
  do not establish multi-GPU or real-robot readiness.
- User-confirmed WA1 drift recipe (2026-09-10): official N1.7 base initialization;
  vision/text backbone LoRA rank 16, alpha 32; full action-head training; G=4,
  temperatures [0.02, 0.05, 0.2], per-timestep loss; four GPUs with microbatch 2
  and accumulation 8 (effective batch 64); state dropout 0; 20,000 steps,
  LR 1e-4, warmup 0.05, weight decay 1e-5. Reuse the recorded FM train/validation
  splits and processing. Final push/cloud training commands require user review
  before execution. This recipe does not change any FM defaults.

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** CUDA 13.0 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.
