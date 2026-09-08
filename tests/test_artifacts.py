from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from rlordata import artifacts


def _make_run(root: Path) -> Path:
    run = root / "runs" / "eval" / "m" / "test_300" / "greedy"
    run.mkdir(parents=True)
    (run / "samples.jsonl").write_text('{"a": 1}\n')
    (run / "metrics.json").write_text("{}\n")
    (run / "sub").mkdir()
    (run / "sub" / "x.txt").write_text("x")
    return run


def test_sync_local_mirrors_repo_relative_layout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run = _make_run(repo)
    dest_root = tmp_path / "artifacts"
    dest = artifacts.sync_run_dir(run, str(dest_root), repo_root=repo)
    assert dest == str(dest_root / "runs" / "eval" / "m" / "test_300" / "greedy")
    assert (Path(dest) / "samples.jsonl").read_text() == '{"a": 1}\n'
    assert (Path(dest) / "sub" / "x.txt").exists()
    # idempotent
    assert artifacts.sync_run_dir(run, str(dest_root), repo_root=repo) == dest


def test_sync_single_file_and_missing_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    f = repo / "data" / "samples" / "tiering_pass8.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("{}\n")
    dest = artifacts.sync_run_dir(f, str(tmp_path / "a"), repo_root=repo)
    assert Path(dest) == tmp_path / "a" / "data" / "samples" / "tiering_pass8.jsonl"
    assert artifacts.sync_run_dir(repo / "nope", str(tmp_path / "a"), repo_root=repo) is None


def test_unset_env_is_a_warned_noop(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv(artifacts.ENV_VAR, raising=False)
    run = _make_run(tmp_path)
    assert artifacts.sync_run_dir(run) is None
    assert "NOT syncing" in capsys.readouterr().err


def test_env_var_is_default_destination(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    run = _make_run(repo)
    monkeypatch.setenv(artifacts.ENV_VAR, str(tmp_path / "store"))
    dest = artifacts.sync_run_dir(run, repo_root=repo)
    assert dest is not None and dest.startswith(str(tmp_path / "store"))


def test_sync_all_and_cli(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    _make_run(repo)
    (repo / "configs" / "locked").mkdir(parents=True)
    (repo / "configs" / "locked" / "cap.yaml").write_text("max_completion_tokens: 1024\n")
    monkeypatch.chdir(repo)
    monkeypatch.setenv(artifacts.ENV_VAR, str(tmp_path / "store"))
    done = artifacts.sync_all(repo_root=repo)
    assert len(done) == 2
    assert (tmp_path / "store" / "configs" / "locked" / "cap.yaml").exists()
    assert artifacts.main(["sync-all"]) == 0
    monkeypatch.delenv(artifacts.ENV_VAR)
    assert artifacts.main(["sync-all"]) == 3


def test_s3_uses_boto3_upload_file(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            calls.append((filename, bucket, key))

    fake = types.ModuleType("boto3")
    fake.client = lambda name: FakeClient()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake)
    repo = tmp_path / "repo"
    run = _make_run(repo)
    dest = artifacts.sync_run_dir(run, "s3://my-bucket/rlordata", repo_root=repo)
    assert dest == "s3://my-bucket/rlordata/runs/eval/m/test_300/greedy"
    keys = sorted(k for _, b, k in calls if b == "my-bucket")
    assert keys == [
        "rlordata/runs/eval/m/test_300/greedy/metrics.json",
        "rlordata/runs/eval/m/test_300/greedy/samples.jsonl",
        "rlordata/runs/eval/m/test_300/greedy/sub/x.txt",
    ]
    with pytest.raises(ValueError):
        artifacts._parse_s3("s3://")


def test_restore_copies_store_back_into_repo(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    run = _make_run(repo)
    store = tmp_path / "store"
    artifacts.sync_run_dir(run, str(store), repo_root=repo)
    (repo / "data" / "samples").mkdir(parents=True)
    (repo / "data" / "samples" / "tiering_pass8.jsonl").write_text("{}\n")
    artifacts.sync_run_dir(repo / "data" / "samples", str(store), repo_root=repo)
    # a fresh clone: nothing local
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    counts = artifacts.restore(["runs", "data/samples", "data/splits"], str(store), repo_root=fresh)
    assert counts == {"copied": 4, "skipped": 0}
    assert (
        fresh / "runs" / "eval" / "m" / "test_300" / "greedy" / "samples.jsonl"
    ).read_text() == '{"a": 1}\n'
    assert (fresh / "data" / "samples" / "tiering_pass8.jsonl").exists()
    # existing files are kept unless --overwrite
    (fresh / "data" / "samples" / "tiering_pass8.jsonl").write_text("local\n")
    counts = artifacts.restore(["data/samples"], str(store), repo_root=fresh)
    assert counts == {"copied": 0, "skipped": 1}
    assert (fresh / "data" / "samples" / "tiering_pass8.jsonl").read_text() == "local\n"
    counts = artifacts.restore(["data/samples"], str(store), repo_root=fresh, overwrite=True)
    assert counts == {"copied": 1, "skipped": 0}
    monkeypatch.chdir(fresh)
    monkeypatch.setenv(artifacts.ENV_VAR, str(store))
    assert artifacts.main(["restore", "runs"]) == 0
    monkeypatch.delenv(artifacts.ENV_VAR)
    import pytest

    with pytest.raises(RuntimeError):
        artifacts.restore(["runs"], None, repo_root=fresh)
