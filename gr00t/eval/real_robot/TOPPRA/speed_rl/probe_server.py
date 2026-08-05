from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import time
from typing import Any

import numpy as np

from ..eval_toppra_bimanual import DEFAULT_TASK, GR00TPolicyClient
from .client import discover_server_contract
from .contract import extract_candidate_features


_KION_STATE_VALUES = {
    "left_tcp": np.array([0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "right_tcp": np.array([0.4, -0.2, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "left_wrist_force": np.zeros(3, dtype=np.float32),
    "right_wrist_force": np.zeros(3, dtype=np.float32),
    "left_finger_pressure": np.zeros(5, dtype=np.float32),
    "right_finger_pressure": np.zeros(5, dtype=np.float32),
}


def _modality_keys(config: Mapping[str, Any], name: str) -> list[str]:
    if name not in config:
        return []
    value = config[name]
    if hasattr(value, "modality_keys"):
        keys = value.modality_keys
    elif isinstance(value, Mapping):
        keys = value.get("modality_keys")
    else:
        raise ValueError(f"Server modality {name!r} has unsupported type {type(value)!r}")
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        raise ValueError(f"Server modality {name!r} has no valid modality_keys")
    return [str(key) for key in keys]


def build_kion_probe_observation(
    modality_config: Mapping[str, Any],
    *,
    task: str,
    batch_size: int,
    image_size: int,
) -> dict[str, dict[str, Any]]:
    unknown_states = [
        key for key in _modality_keys(modality_config, "state") if key not in _KION_STATE_VALUES
    ]
    if unknown_states:
        raise ValueError(
            "The server requests state keys that the Kion probe cannot safely synthesize: "
            f"{unknown_states}"
        )
    video = {
        key: np.zeros((batch_size, 1, image_size, image_size, 3), dtype=np.uint8)
        for key in _modality_keys(modality_config, "video")
    }
    state = {
        key: np.repeat(_KION_STATE_VALUES[key][None, None, :], batch_size, axis=0)
        for key in _modality_keys(modality_config, "state")
    }
    language_keys = _modality_keys(modality_config, "language")
    language = {key: [[task] for _ in range(batch_size)] for key in language_keys}
    return {"video": video, "state": state, "language": language}


def probe_server(
    *,
    host: str,
    port: int,
    timeout_ms: int,
    task: str,
    tts_samples: int,
    image_size: int,
) -> dict[str, Any]:
    contract = discover_server_contract(host, port, timeout_ms)
    client = GR00TPolicyClient(host, port, timeout_ms=timeout_ms)
    try:
        modality_config = client.get_modality_config()
        if not isinstance(modality_config, Mapping):
            raise ValueError("Server modality config must be a mapping")
        observation = build_kion_probe_observation(
            modality_config,
            task=task,
            batch_size=tts_samples,
            image_size=image_size,
        )
        started = time.perf_counter()
        response = client.get_action(observation)
        latency_s = time.perf_counter() - started
    finally:
        client.close()

    actions = response[0]
    if not isinstance(actions, Mapping) or not actions:
        raise ValueError("Server returned an empty or invalid action mapping")
    action_shapes = {key: tuple(np.asarray(value).shape) for key, value in actions.items()}
    first_shape = next(iter(action_shapes.values()))
    if len(first_shape) < 2:
        raise ValueError(f"Action arrays must have (K, H, ...) shape, got {first_shape}")
    candidate_count, horizon = first_shape[:2]
    if candidate_count != tts_samples:
        raise ValueError(f"Server returned {candidate_count} candidates, expected {tts_samples}")
    if any(shape[:2] != (candidate_count, horizon) for shape in action_shapes.values()):
        raise ValueError(f"Server action shapes are not candidate/horizon aligned: {action_shapes}")
    features = extract_candidate_features(
        response,
        contract,
        [horizon] * candidate_count,
    )
    return {
        "server": f"{host}:{port}",
        "latency_ms": round(latency_s * 1000.0, 3),
        "contract": contract.to_dict(),
        "action_shapes": {key: list(shape) for key, shape in action_shapes.items()},
        "feature_shapes": [list(feature.shape) for feature in features],
        "status": "ok",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a Speed-RL GR00T server with synthetic Kion observations."
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=47866)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--tts-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.tts_samples < 1 or args.image_size < 1 or args.timeout_ms < 1:
        parser.error("tts-samples, image-size, and timeout-ms must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = probe_server(
        host=args.server_host,
        port=args.server_port,
        timeout_ms=args.timeout_ms,
        task=args.task,
        tts_samples=args.tts_samples,
        image_size=args.image_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
