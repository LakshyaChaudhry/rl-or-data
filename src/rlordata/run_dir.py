"""Run-directory provenance (CLAUDE.md "Every run directory contains ..."). AGENT-OWNED; tasks/02a.

Every eval / tier / training run directory gets, via :func:`start_run`:

- ``config.yaml``       the fully resolved config the run actually used
- ``config_hash.txt``   sha256 of the canonical JSON of that config (``generator.config_hash``)
- ``meta.json``         run_id, config hash, git SHA, package versions, hostname, GPU type,
                        started_at / finished_at / wall_clock_s, cost estimate, free-form extras
- ``NOTES.md``          lab-notebook stub in the ``notebook/LAB_NOTEBOOK.md`` template

:func:`finish_run` fills in the timing fields at exit.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rlordata.data.generator import config_hash as _config_hash
from rlordata.env_check import versions as _versions
from rlordata.envfile import gpu_rate_usd_per_hour

PACKAGES = ("rlordata", "torch", "transformers", "peft", "trl", "vllm", "numpy", "reasoning-gym")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=5
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str]:
    return _versions(PACKAGES)


def gpu_type() -> str:
    """GPU name via torch if importable, else ``nvidia-smi``, else ``"none"``."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001 — torch missing or broken; fall through
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        if names:
            return ", ".join(names)
    except (OSError, subprocess.SubprocessError):
        pass
    return "none"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def config_hash(resolved: dict[str, Any]) -> str:
    """sha256 hex of the canonical JSON of ``resolved`` (same function the generator uses)."""
    return _config_hash(resolved)


def format_cost(gpu_hours: float, rate: float | None) -> str:
    if rate is None:
        return f"{gpu_hours:.3f} GPU-h × (RLORDATA_GPU_RATE_USD_PER_HOUR unset) = $?"
    return f"{gpu_hours:.3f} GPU-h × ${rate:.2f}/h = ${gpu_hours * rate:.2f}"


def notes_stub(run_id: str, cfg_hash: str, sha: str, gpu: str) -> str:
    return (
        f"### {datetime.now(UTC).date().isoformat()} — {run_id}\n"
        f"- Config hash: {cfg_hash[:12]} | git SHA: {sha[:12]} | GPU: {gpu} | wall-clock: | est. cost:\n"
        "- What I ran / decided:\n"
        "- Result (with n, seed, CI, truncation%):\n"
        "- What I learned (one sentence):\n"
        "- Next:\n"
    )


@dataclass
class RunHandle:
    """Open run directory; call :func:`finish_run` when done."""

    run_dir: Path
    run_id: str
    config_hash: str
    started_monotonic: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_monotonic


def start_run(
    run_dir: str | Path,
    resolved_config: dict[str, Any],
    *,
    run_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> RunHandle:
    """Create ``run_dir`` and write config.yaml, config_hash.txt, meta.json, NOTES.md."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_hash(resolved_config)
    sha = git_sha()
    gpu = gpu_type()
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=True, default_flow_style=False)
    (run_dir / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")
    meta: dict[str, Any] = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "git_sha": sha,
        "git_dirty": git_dirty(),
        "package_versions": package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "gpu_type": gpu,
        "gpu_rate_usd_per_hour": gpu_rate_usd_per_hour(),
        "started_at": now_iso(),
        "finished_at": None,
        "wall_clock_s": None,
        "status": "running",
    }
    if extra_meta:
        meta.update(extra_meta)
    _write_meta(run_dir, meta)
    notes = run_dir / "NOTES.md"
    if not notes.exists():
        notes.write_text(notes_stub(run_id, cfg_hash, sha, gpu), encoding="utf-8")
    return RunHandle(
        run_dir=run_dir,
        run_id=run_id,
        config_hash=cfg_hash,
        started_monotonic=time.monotonic(),
        meta=meta,
    )


def finish_run(handle: RunHandle, *, status: str = "finished", **updates: Any) -> dict[str, Any]:
    """Stamp finished_at / wall_clock_s / actual cost into meta.json."""
    wall = handle.elapsed_s
    rate = gpu_rate_usd_per_hour()
    handle.meta.update(
        {
            "finished_at": now_iso(),
            "wall_clock_s": round(wall, 3),
            "gpu_hours_actual": round(wall / 3600.0, 6),
            "cost_usd_actual": None if rate is None else round(wall / 3600.0 * rate, 4),
            "status": status,
        }
    )
    handle.meta.update(updates)
    _write_meta(handle.run_dir, handle.meta)
    return handle.meta


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def read_run_config(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_run_meta(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "meta.json").open(encoding="utf-8") as f:
        return json.load(f)


REQUIRED_RUN_FILES = ("config.yaml", "config_hash.txt", "meta.json", "NOTES.md")
