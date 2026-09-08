"""Shared test scaffolding (infrastructure only; no expected values). AGENT-OWNED."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlordata.cli import main as cli_main
from rlordata.data.generator import generate_pool, write_jsonl
from rlordata.types import Problem

REPO = Path(__file__).resolve().parents[1]

TINY_SPLITS: dict[str, Any] = {
    "train_easy_100": {"easy": 20},
    "train_mixed_100": {"easy": 6, "medium": 6, "hard": 8},
    "val_mixed_100": {"easy": 6, "medium": 6, "hard": 8},
    "test_300": {"easy": 20, "medium": 20, "hard": 20},
    "train_curated": {"derived_from": "train_mixed_100", "pass8_min": 1, "pass8_max": 7},
}


def load_yaml(rel: str) -> dict[str, Any]:
    with (REPO / rel).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path


def small_pool(n: int = 800, seed: int = 20260909) -> list[Problem]:
    cfg = deepcopy(load_yaml("configs/data/pool.yaml"))
    cfg["n_problems"] = n
    cfg["seed"] = seed
    return generate_pool(cfg, seed=seed)


def small_ood(n: int = 30, seed: int = 20260909) -> list[Problem]:
    cfg = deepcopy(load_yaml("configs/data/ood.yaml"))
    cfg["n_problems"] = n
    cfg["seed"] = seed
    return generate_pool(cfg, seed=seed)


@dataclass
class World:
    """A tiny end-to-end world in tmp_path: pool, ood, tiering config, cap path (absent)."""

    root: Path
    pool_path: Path
    ood_path: Path
    cap_path: Path
    tier_config: Path
    splits_dir: Path
    samples_path: Path
    tier_run_dir: Path

    def tier_with_stub(self, extra_args: list[str] | None = None) -> int:
        return cli_main(["tier", "--config", str(self.tier_config), "--stub", *(extra_args or [])])

    def eval_config(self, **overrides: Any) -> Path:
        cfg = deepcopy(load_yaml("configs/eval/base.yaml"))
        cfg.update(
            {
                "splits_dir": str(self.splits_dir),
                "pool": str(self.pool_path),
                "ood_path": str(self.ood_path),
                "cap_yaml": str(self.cap_path),
                "output_dir": str(self.root / "runs" / "eval"),
            }
        )
        cfg.update(overrides)
        return dump_yaml(cfg, self.root / "configs" / "eval.yaml")


def make_world(tmp_path: Path, *, n_pool: int = 800, n_ood: int = 30) -> World:
    root = tmp_path
    pool_path = root / "data" / "pool" / "pool.jsonl"
    ood_path = root / "data" / "pool" / "ood_hard_200.jsonl"
    write_jsonl(small_pool(n_pool), pool_path)
    write_jsonl(small_ood(n_ood), ood_path)
    cap_path = root / "configs" / "locked" / "cap.yaml"  # deliberately absent
    splits_dir = root / "data" / "splits"
    samples_path = root / "data" / "samples" / "tiering_pass8.jsonl"
    tier_run_dir = root / "runs" / "tier"
    cfg = deepcopy(load_yaml("configs/data/tiering.yaml"))
    cfg.update(
        {
            "input": str(pool_path),
            "cap_yaml": str(cap_path),
            "splits": TINY_SPLITS,
            "post_hoc_tier_inputs": [str(ood_path)],
            "output_dir": str(splits_dir),
            "samples_output": str(samples_path),
            "run_dir": str(tier_run_dir),
        }
    )
    tier_config = dump_yaml(cfg, root / "configs" / "tiering.yaml")
    return World(
        root=root,
        pool_path=pool_path,
        ood_path=ood_path,
        cap_path=cap_path,
        tier_config=tier_config,
        splits_dir=splits_dir,
        samples_path=samples_path,
        tier_run_dir=tier_run_dir,
    )
