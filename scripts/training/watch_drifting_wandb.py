#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Backfill and follow drift Trainer metrics without restarting training.

Reads the existing train.log and train_status.json. Only this separate process
uses W&B; no model weights, training callbacks, or checkpoint files are changed.
"""

import argparse
import ast
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import select
import socket
import threading
import time


METRICS = {
    "loss": "train/loss",
    "grad_norm": "train/grad_norm",
    "learning_rate": "train/learning_rate",
    "eval_loss": "eval/loss",
    "eval_runtime": "eval/runtime",
    "eval_samples_per_second": "eval/samples_per_second",
    "eval_steps_per_second": "eval/steps_per_second",
}


def start_resolve_proxy(addresses):
    """Keep TLS verification with a process-local proxy when system DNS is down."""

    class Tunnel(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *unused):
            pass

        def do_CONNECT(self):
            host, separator, port = self.path.rpartition(":")
            if not separator or port != "443" or host not in addresses:
                self.send_error(403, "Host is not in the explicit HTTPS resolution map")
                return
            try:
                with socket.create_connection((addresses[host], 443), timeout=15) as upstream:
                    self.send_response(200, "Connection established")
                    self.end_headers()
                    self.wfile.flush()
                    self.close_connection = True
                    while True:
                        readable, _, _ = select.select([self.connection, upstream], [], [], 120)
                        if not readable:
                            return
                        for connection in readable:
                            data = connection.recv(65536)
                            if not data:
                                return
                            target = upstream if connection is self.connection else self.connection
                            target.sendall(data)
            except OSError:
                return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Tunnel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proxy = f"http://127.0.0.1:{server.server_port}"
    os.environ.update(
        HTTP_PROXY=proxy,
        HTTPS_PROXY=proxy,
        http_proxy=proxy,
        https_proxy=proxy,
        NO_PROXY="",
        no_proxy="",
    )
    return server


def parse_metrics(log, max_steps):
    """Use the training progress bar's step, excluding nested evaluation bars."""
    pattern = re.compile(rf"(?P<step>\d+)/{max_steps}\s+\[|(?P<row>\{{[^{{}}\r\n]*\}})")
    step = 0
    records = []
    for match in pattern.finditer(log):
        if match.group("step") is not None:
            step = int(match.group("step"))
            continue
        raw = match.group("row")
        if not re.search(r"['\"](?:loss|eval_loss)['\"]\s*:", raw):
            continue
        row = ast.literal_eval(raw)
        if step <= 0:
            raise ValueError("A metric has no preceding training step")
        metrics = {METRICS[k]: v for k, v in row.items() if k in METRICS}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values()):
            raise ValueError(f"Non-finite or invalid metrics at training step {step}")
        records.append({"train/global_step": step, **metrics})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="finetune-gr00t-n1d7")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument(
        "--resolve",
        action="append",
        default=[],
        metavar="HOST=IP",
        help="Optional process-local HTTPS DNS override; repeat for each host",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    root = args.run_dir.resolve()
    metadata = json.loads((root / "train_command.json").read_text())
    argv = metadata["argv"]

    def option(name):
        return argv[argv.index(name) + 1]

    if option("--action-head-type") != "drifting":
        raise ValueError("This monitor is only for explicitly enabled drifting runs")
    max_steps = int(option("--max-steps"))
    directory = root / "wandb-monitor"
    directory.mkdir(exist_ok=True)
    # Avoid two uploaders writing the same W&B run concurrently.
    lock = (directory / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = f"{root}:{metadata['git_commit']}"
    run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    parse_metrics((root / "train.log").read_text(errors="replace"), max_steps)
    addresses = {}
    for value in args.resolve:
        host, address = value.split("=", 1)
        addresses[host] = str(ipaddress.ip_address(address))
    proxy = start_resolve_proxy(addresses) if addresses else None

    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=run_id,
        resume="allow",
        name=root.name,
        dir=str(directory),
        mode="online",
        job_type="drifting-training",
        tags=["drifting", "lora16", "wa1", "log-monitor"],
        config={
            "training_git_commit": metadata["git_commit"],
            "logging_source": "existing Trainer stdout; independent monitoring process",
            "action_head_type": "drifting",
            "max_steps": max_steps,
            "learning_rate": float(option("--learning-rate")),
            "global_batch_size": int(option("--global-batch-size")),
            "gradient_accumulation_steps": int(option("--gradient-accumulation-steps")),
            "drifting_lora_rank": int(option("--drifting-lora-rank")),
            "drifting_lora_alpha": float(option("--drifting-lora-alpha")),
            "drifting_gen_per_label": int(option("--drifting-gen-per-label")),
        },
        settings=wandb.Settings(
            console="off", disable_code=True, x_disable_stats=True, init_timeout=45
        ),
    )
    run.define_metric("train/global_step", hidden=True)
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("eval/*", step_metric="train/global_step")
    # W&B's internal step counts log records. The chart x-axis is the optimizer
    # step, so train and eval records at the same optimizer step both survive.
    cursor = run.step
    print(f"W&B run: {run.url}; resuming at record {cursor}", flush=True)
    try:
        while True:
            status = json.loads((root / "train_status.json").read_text())
            records = parse_metrics((root / "train.log").read_text(errors="replace"), max_steps)
            if cursor > len(records):
                raise RuntimeError("Training log is shorter than the previously uploaded history")
            for index in range(cursor, len(records)):
                run.log(records[index], step=index)
            cursor = len(records)
            state = status["state"]
            run.summary["training_state"] = state
            run.summary["checkpoint_reload_passed"] = status.get("checkpoint_reload_passed", False)
            report = {
                "url": run.url,
                "run_id": run.id,
                "entity": run.entity,
                "project": run.project,
                "pid": os.getpid(),
                "training_state": state,
                "records_logged": cursor,
                "last_training_step": records[-1]["train/global_step"] if records else 0,
                "updated_unix": time.time(),
            }
            temporary = root / "wandb_monitor.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(root / "wandb_monitor.json")
            print(json.dumps(report), flush=True)
            if state in ("complete", "failed"):
                run.finish(exit_code=0 if state == "complete" else 1)
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        run.finish(exit_code=1)
        raise
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        lock.close()


if __name__ == "__main__":
    main()
