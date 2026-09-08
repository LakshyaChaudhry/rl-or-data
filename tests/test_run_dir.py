from __future__ import annotations

import json
from pathlib import Path

from rlordata.data.generator import config_hash
from rlordata.run_dir import (
    REQUIRED_RUN_FILES,
    finish_run,
    read_run_config,
    read_run_meta,
    start_run,
)


def test_start_and_finish_run_write_provenance(tmp_path: Path) -> None:
    resolved = {"kind": "eval", "seed": 1, "max_completion_tokens": 1024, "nested": {"a": [1, 2]}}
    h = start_run(tmp_path / "run", resolved, run_id="r1", extra_meta={"split": "test_300"})
    for name in REQUIRED_RUN_FILES:
        assert (tmp_path / "run" / name).exists(), name
    assert h.config_hash == config_hash(resolved)
    assert (tmp_path / "run" / "config_hash.txt").read_text().strip() == h.config_hash
    assert read_run_config(tmp_path / "run") == resolved
    meta = read_run_meta(tmp_path / "run")
    for key in ("git_sha", "package_versions", "gpu_type", "started_at", "hostname", "python"):
        assert key in meta, key
    assert meta["split"] == "test_300" and meta["status"] == "running"
    assert "rlordata" in meta["package_versions"]
    notes = (tmp_path / "run" / "NOTES.md").read_text()
    assert "r1" in notes and h.config_hash[:12] in notes

    finish_run(h, n_samples=10)
    meta = json.loads((tmp_path / "run" / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["finished_at"] is not None
    assert meta["wall_clock_s"] >= 0 and meta["n_samples"] == 10
