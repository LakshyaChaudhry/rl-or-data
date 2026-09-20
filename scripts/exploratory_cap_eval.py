"""EXPLORATORY greedy re-evaluation at 2 × the locked cap (tasks/06b C). AGENT-OWNED.

A logged deviation (PREREGISTRATION.md §4, 2026-09-20, approved by Laksh): SPEC §7 fixes one cap
for every evaluation and SPEC §10 evaluates test once. The primary, locked-cap numbers stay the
result. This script shows what greedy accuracy would have been without truncation, for base, the
H3 pair and iterated RFT, and it is built so its output cannot be mistaken for a result:

  * the cap is ``cap_multiple × locked`` by rule; ``sampling.cap.resolve_cap`` admits exactly that
    value and only while this script holds the deviation id in the environment;
  * it refuses to write under ``runs/eval`` or any ``eval/final`` directory;
  * every config.yaml and metrics.json is stamped ``exploratory: true`` with the deviation id;
  * the adapter is the one the primary eval read (path checked, sha256 recorded).

    uv run python scripts/exploratory_cap_eval.py --config configs/eval/exploratory_cap.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from rlordata.envfile import load_env
from rlordata.sampling.cap import (
    EXPLORATORY_CAP_DEVIATION_ID,
    EXPLORATORY_CAP_ENV,
    EXPLORATORY_CAP_MULTIPLE,
    load_locked_cap,
)
from rlordata.sampling.eval_runner import (
    DecodingSpec,
    EvalUnit,
    ModelSpec,
    load_split,
    make_sampler,
    resolved_unit_config,
    run_unit,
)
from rlordata.train.common import print_cost, read_json, release_cuda_cache, sync_run

NOTE = (
    "EXPLORATORY: re-generated at a cap of {cap} tokens (2 x the locked {locked}) to show what "
    "accuracy would have been without truncation. Run after the primary results were seen; the "
    "policies were trained with completions capped at {locked}. Not a SPEC §10 number."
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_result_dirs(out: Path) -> None:
    """Exploratory units may never sit where a result is looked for."""
    parts = out.resolve().parts
    pairs = list(zip(parts, parts[1:], strict=False))
    if ("runs", "eval") in pairs or ("eval", "final") in pairs or ("eval", "val") in pairs:
        raise SystemExit(f"refusing to write exploratory units under a result directory: {out}")
    if "exploratory" not in out.name:
        raise SystemExit(f"output_dir must be named exploratory_*, got {out.name!r}")


def primary_adapter(run_dir: Path) -> str:
    """The adapter path the primary test eval recorded; must equal budgets.final_adapter."""
    adapter = str(read_json(run_dir / "budgets.json")["final_adapter"])
    primary = run_dir / "eval" / "final" / "test_300" / "greedy" / "config.yaml"
    if primary.exists():
        with primary.open(encoding="utf-8") as f:
            recorded = (yaml.safe_load(f) or {}).get("adapter")
        if recorded != adapter:
            raise SystemExit(
                f"{run_dir}: primary eval read {recorded!r}, budgets.json says {adapter!r}"
            )
    return adapter


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval/exploratory_cap.yaml")
    ap.add_argument("--only", default=None, help="comma-separated model names (default: all)")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--n-problems", type=int, default=None, help="--stub only")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.n_problems is not None and not args.stub:
        raise SystemExit("--n-problems is only for --stub dry runs")
    with Path(args.config).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    locked = load_locked_cap(cfg["cap_yaml"])
    if locked is None:
        raise SystemExit(f"{cfg['cap_yaml']} missing: the locked cap defines the exploratory one")
    if int(cfg["cap_multiple"]) != EXPLORATORY_CAP_MULTIPLE:
        raise SystemExit(f"cap_multiple must be {EXPLORATORY_CAP_MULTIPLE} (PREREGISTRATION §4)")
    cap = EXPLORATORY_CAP_MULTIPLE * locked
    out_root = Path(args.output_dir or cfg["output_dir"])
    refuse_result_dirs(out_root)
    os.environ[EXPLORATORY_CAP_ENV] = EXPLORATORY_CAP_DEVIATION_ID  # the one sanctioned deviation
    note = NOTE.format(cap=cap, locked=locked)
    print(note)

    models: list[dict[str, Any]] = [{**cfg["base"], "adapter": None, "dir": None}]
    for run in cfg["runs"]:
        models.append({**run, "adapter": primary_adapter(Path(run["dir"]))})
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m["name"] in keep]
    splits_dir = Path(cfg["splits_dir"])
    greedy = DecodingSpec(name="greedy", temperature=0.0, n=1)
    n_units = len(models) * len(cfg["splits"])
    print_cost(
        f"exploratory start estimate ({n_units} greedy units, worst case every completion at {cap})",
        sum(500 * cap for _ in models) / 3000.0 / 3600.0,
    )
    for m in models:
        spec = ModelSpec(id=cfg["model_id"], kind="base", arm=m["name"], lora_path=m["adapter"])
        units = []
        for split in cfg["splits"]:
            problems = load_split(
                split,
                splits_dir=splits_dir,
                pool_path=Path(cfg["pool"]),
                ood_path=Path(cfg["ood_path"]),
            )
            if args.n_problems is not None:
                problems = problems[: args.n_problems]
            out = out_root / m["name"] / split / "greedy"
            if (out / "metrics.json").exists() and not args.force:
                print(f"  [skip] {m['name']} {split}: already evaluated")
                continue
            units.append((EvalUnit(spec, split, greedy, problems), out))
        if not units:
            continue
        weights = None if m["adapter"] is None else Path(m["adapter"]) / "adapter_model.safetensors"
        if weights is not None and not weights.exists() and not args.stub:
            raise SystemExit(f"{weights} missing: restore the evaluated adapter from the store")
        release_cuda_cache()
        sampler = make_sampler(
            spec,
            stub=args.stub,
            cap=cap,
            seed=int(m["seed"]),
            cap_path=cfg["cap_yaml"],
            allow_provisional_cap=False,
            problems=[p for u, _ in units for p in u.problems],
            dtype=str(cfg.get("dtype", "bfloat16")),
            gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
        )
        assert int(sampler.max_completion_tokens) == cap
        try:
            for unit, out in units:
                resolved = resolved_unit_config(
                    unit,
                    seed=int(m["seed"]),
                    cap=cap,
                    cap_path=cfg["cap_yaml"],
                    sampler_desc=sampler.describe(),
                    chat_kwargs=None,
                    source_config=str(args.config),
                )
                stamp = {
                    "exploratory": True,
                    "cap_deviation": EXPLORATORY_CAP_DEVIATION_ID,
                    "locked_cap": locked,
                    "note": note,
                }
                resolved.update(
                    {
                        **stamp,
                        "kind": "exploratory_cap_eval",
                        "train_run_dir": m["dir"],
                        "adapter": m["adapter"],
                        "adapter_sha256": None if weights is None else _sha256(weights),
                    }
                )
                run_id = f"exploratory_cap{cap}_{m['name']}_{unit.split}_greedy_seed{m['seed']}"
                resolved["run_id"] = run_id
                metrics = run_unit(
                    unit,
                    sampler,
                    out,
                    seed=int(m["seed"]),
                    resolved=resolved,
                    chat_kwargs=None,
                    run_id=run_id,
                )
                (out / "metrics.json").write_text(
                    json.dumps({**metrics, **stamp}, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"  [exploratory cap {cap}] {m['name']} {unit.split}: acc {metrics['accuracy']:.3f} "
                    f"[{metrics['ci_low']:.3f},{metrics['ci_high']:.3f}] n={metrics['n_problems']} "
                    f"seed {m['seed']} trunc {100 * metrics['truncation_rate']:.1f}%"
                )
        finally:
            if hasattr(sampler, "close"):
                sampler.close()
            sync_run(out_root / m["name"])
    (out_root / "README.md").write_text(f"# {out_root.name}\n\n{note}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
