"""tasks/04 §7: the queue runner skips finished jobs, resumes crashed GRPO jobs, stops on failure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _load():
    spec = importlib.util.spec_from_file_location("run_queue", Path("scripts/run_queue.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_default_queue_parses_and_orders_h3_pair_first():
    rq = _load()
    jobs = rq.load_queue("queue.yaml")
    names = [j["name"] for j in jobs]
    # mixed s1 trains then evaluates first: the eval stage is exercised after one run, not eleven
    assert names[:4] == [
        "train_grpo_mixed_s1",
        "eval_grpo_mixed_s1",
        "train_grpo_curated_s1",
        "train_grpo_easy_s1",
    ]
    assert names[4:6] == ["train_grpo_random_reward_s1", "train_grpo_format_only_s1"]
    assert len([n for n in names if n.startswith("train_")]) == 11
    assert len([n for n in names if n.startswith("eval_")]) == 11
    assert all(n.startswith("train_") for n in names[2:12])
    for j in jobs:
        assert "--stage train" in j["cmd"] or "--stage eval" in j["cmd"]
        assert j["produces"]


def test_resume_appended_only_when_checkpoint_exists_and_run_unfinished(tmp_path: Path):
    rq = _load()
    rd = tmp_path / "grpo_mixed_s1"
    job = {"cmd": "echo train", "run_dir": str(rd), "resume": True}
    assert rq.resolve_cmd(job) == "echo train"
    (rd / "checkpoints" / "checkpoint-25").mkdir(parents=True)
    assert rq.resolve_cmd(job) == "echo train --resume"
    (rd / "budgets.json").write_text("{}")
    assert rq.resolve_cmd(job) == "echo train"


def test_run_queue_skips_markers_and_stops_on_failure(tmp_path: Path, capsys):
    rq = _load()
    marker = tmp_path / "done.txt"
    marker.write_text("x")
    q = tmp_path / "q.yaml"
    q.write_text(
        yaml.safe_dump(
            {
                "jobs": [
                    {"name": "a", "cmd": "echo a", "produces": [str(marker)]},
                    {"name": "b", "cmd": "false"},
                    {"name": "c", "cmd": "echo never"},
                ]
            }
        )
    )
    rc = rq.run_queue(q)
    out = capsys.readouterr().out
    assert rc != 0
    assert "skip a" in out and "STOP on failure at b" in out and "never" not in out
