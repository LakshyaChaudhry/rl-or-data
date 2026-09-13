"""Minimal sequential job runner (tasks/04 §7). AGENT-OWNED.

``queue.yaml`` entries: ``{name, cmd, requires: [paths], produces: [marker paths]}``.
Skips jobs whose markers already exist; stops on first failure; prints cumulative cost.
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


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    jobs = data.get("jobs") if isinstance(data, dict) else data
    if not isinstance(jobs, list):
        raise SystemExit(f"{path}: expected a list of jobs or {{jobs: [...]}}")
    return jobs


def run_queue(path: str | Path, *, dry_run: bool = False) -> int:
    jobs = load_queue(path)
    total_s = 0.0
    for i, job in enumerate(jobs, 1):
        name = job.get("name", f"job_{i}")
        cmd = job["cmd"]
        requires = [Path(p) for p in job.get("requires") or []]
        produces = [Path(p) for p in job.get("produces") or []]
        if produces and all(p.exists() for p in produces):
            print(f"[queue] skip {name}: markers exist")
            continue
        missing = [str(p) for p in requires if not p.exists()]
        if missing:
            print(f"[queue] STOP {name}: missing requires {missing}")
            return 1
        print(f"[queue] ({i}/{len(jobs)}) {name}: {cmd}")
        if dry_run:
            continue
        t0 = time.monotonic()
        rc = subprocess.call(cmd, shell=True)
        dt = time.monotonic() - t0
        total_s += dt
        print(f"[queue] {name} exit={rc} wall={dt / 3600:.3f} h")
        if rc != 0:
            print(
                f"[queue] STOP on failure; cumulative {format_cost(total_s / 3600, gpu_rate_usd_per_hour())}"
            )
            return rc
        for p in produces:
            if not p.exists():
                print(f"[queue] WARN {name}: expected marker missing {p}")
    print(f"[queue] done; cumulative {format_cost(total_s / 3600, gpu_rate_usd_per_hour())}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/run_queue.py")
    p.add_argument("--queue", default="queue.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    return run_queue(args.queue, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
