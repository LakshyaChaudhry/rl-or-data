"""Sequential job runner for the GPU box (tasks/04 §7). AGENT-OWNED.

``queue.yaml`` entries: ``{name, cmd, requires: [paths], produces: [marker paths], run_dir?, resume?}``.
Runs jobs in order, skips jobs whose markers already exist, stops on the first failure, syncs
``run_dir`` to the artifact store after each job, and prints cumulative GPU-hours and cost.
A job with ``resume: true`` gets ``--resume`` appended when its ``run_dir`` already holds a
``checkpoints/checkpoint-*`` directory but no ``budgets.json`` (a GRPO run that died mid-way).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from rlordata.envfile import gpu_rate_usd_per_hour
from rlordata.run_dir import format_cost
from rlordata.train.common import sync_run


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(jobs, list):
        raise SystemExit(f"{path}: expected a list of jobs or {{jobs: [...]}}")
    for j in jobs:
        if "cmd" not in j:
            raise SystemExit(f"{path}: job {j.get('name', '?')} has no cmd")
    return jobs


def resolve_cmd(job: dict[str, Any]) -> str:
    """The command to run, with ``--resume`` appended for a resumable job that has checkpoints."""
    cmd = str(job["cmd"])
    run_dir = job.get("run_dir")
    if job.get("resume") and run_dir:
        rd = Path(run_dir)
        has_ckpt = any((rd / "checkpoints").glob("checkpoint-*")) if rd.exists() else False
        if has_ckpt and not (rd / "budgets.json").exists() and "--resume" not in cmd:
            cmd += " --resume"
    return cmd


def run_queue(path: str | Path, *, dry_run: bool = False) -> int:
    jobs = load_queue(path)
    rate = gpu_rate_usd_per_hour()
    total_s = 0.0
    for i, job in enumerate(jobs, 1):
        name = job.get("name", f"job_{i}")
        requires = [Path(p) for p in job.get("requires") or []]
        produces = [Path(p) for p in job.get("produces") or []]
        if produces and all(p.exists() for p in produces):
            print(f"[queue] skip {name}: markers exist")
            continue
        missing = [str(p) for p in requires if not p.exists()]
        if missing and dry_run:
            print(f"[queue] (dry-run) {name} would STOP: missing requires {missing}")
        elif missing:
            print(f"[queue] STOP {name}: missing requires {missing}")
            return 1
        cmd = resolve_cmd(job)
        print(f"[queue] ({i}/{len(jobs)}) {name}: {cmd}")
        if dry_run:
            continue
        t0 = time.monotonic()
        rc = subprocess.call(cmd, shell=True)
        dt = time.monotonic() - t0
        total_s += dt
        print(
            f"[queue] {name} exit={rc} wall={dt / 3600:.3f} h; "
            f"cumulative {format_cost(total_s / 3600, rate)}"
        )
        if job.get("run_dir") and Path(str(job["run_dir"])).exists():
            sync_run(job["run_dir"])
        if rc != 0:
            print(f"[queue] STOP on failure at {name}")
            return rc
        for p in produces:
            if not p.exists():
                print(f"[queue] WARN {name}: expected marker missing {p}")
    print(f"[queue] done; cumulative {format_cost(total_s / 3600, rate)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/run_queue.py")
    p.add_argument("--queue", default="queue.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the commands; run nothing")
    args = p.parse_args(argv)
    return run_queue(args.queue, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
