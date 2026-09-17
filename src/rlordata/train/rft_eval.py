"""Evaluate a trained adapter through the single generation path (tasks/03 §5). AGENT-OWNED.

The adapter is applied by vLLM as a *native* LoRA (``VLLMSampler(lora_path=…)``: ``enable_lora`` +
``LoRARequest``) at the locked cap → the same ``eval_runner`` units, metrics and provenance as the
base-model evals → sync the run directory. It is never merged into the bf16 base weights: a bf16
merge rounds ~90 % of ΔW entries to exactly zero (per-entry |ΔW| ≪ bf16 half-ulp of W), so a merged
model is mostly the base model (notebook 2026-09-14; ``train/vllm_lora.py``).

Eval sets:
    val     ``val_mixed_100`` greedy only — what the sweep selects on (SPEC §10: val only)
    final   val + the primary and secondary sets from ``configs/eval/final.yaml`` (test_300 and
            ood_hard_200 greedy + mean@8, pass@k on the test_300 first-100 subset, the transfer
            sets from ``configs/locked/transfer.yaml``). Each final unit runs once; re-running
            needs ``--force``.

Sanity (``analysis/sanity.py``) gates the numbers: the adapter is non-trivial (‖ΔW‖ > 0), the
trainer and vLLM tokenizers hash identically, the cap and template equal the base eval's, the
adapter evaluated is the run's final one, and the merged model's greedy val outputs differ from
the base model's on at least 10 % of prompts. The last check runs right after the val greedy
unit and before any other unit; on failure that unit's metrics.json is removed and the eval stops.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import yaml

from rlordata.analysis.sanity import (
    check_adapter_nontrivial,
    check_final_checkpoint,
    check_outputs_differ,
    check_protocol_identical,
)
from rlordata.run_dir import read_run_config
from rlordata.sampling.eval_runner import (
    DecodingSpec,
    EvalUnit,
    ModelSpec,
    load_split,
    make_sampler,
    read_samples,
    resolved_unit_config,
    run_unit,
)
from rlordata.train.common import (
    load_yaml,
    print_cost,
    read_json,
    release_cuda_cache,
    sync_run,
    tokenizer_hash,
    write_json,
)
from rlordata.types import Problem

# Recorded in every adapter eval's resolved config: how vLLM applied the adapter.
ADAPTER_MODE = "vllm_native_lora"

DEFAULT_FINAL_EVAL_CONFIG = Path("configs/eval/final.yaml")
DEFAULT_TRANSFER_CONFIG = Path("configs/locked/transfer.yaml")
DEFAULT_BASE_EVAL_DIR = Path("runs/eval/Qwen__Qwen3-4B-Base")
VAL_SPLIT = "val_mixed_100"
MIN_DIFFERENT_FRACTION = 0.10  # tasks/03 §5


def transfer_splits(path: str | Path = DEFAULT_TRANSFER_CONFIG) -> list[str]:
    """The secondary transfer sets, from the locked file Laksh writes (SPEC §6.3 v1.7)."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{p} does not exist: the transfer sets are a locked protocol constant that Laksh writes "
            "(expected e.g. `splits: [gsm8k_500]` after SPEC v1.7). Final evals cannot run without it."
        )
    cfg = load_yaml(p)
    return [str(s) for s in cfg.get("splits", [])]


def plan_eval_units(
    *,
    eval_set: str,
    model_id: str,
    arm: str,
    splits_dir: Path,
    pool_path: Path,
    ood_path: Path | None,
    final_config: Path,
    transfer_config: Path,
    n_problems: int | None = None,
) -> list[EvalUnit]:
    """Val greedy first (the sanity gate), then the rest."""
    model = ModelSpec(id=model_id, kind="base", arm=arm)

    def get(name: str) -> list[Problem]:
        probs = load_split(name, splits_dir=splits_dir, pool_path=pool_path, ood_path=ood_path)
        return probs[: int(n_problems)] if n_problems is not None else probs

    greedy = DecodingSpec(name="greedy", temperature=0.0, n=1)
    units = [EvalUnit(model, VAL_SPLIT, greedy, get(VAL_SPLIT))]
    if eval_set == "val":
        return units
    if eval_set != "final":
        raise ValueError(f"eval_set must be 'val' or 'final', got {eval_set!r}")
    cfg = load_yaml(final_config)
    decodings = [DecodingSpec.from_config(n, e) for n, e in cfg["decoding"].items()]
    splits = [str(s) for s in cfg["splits"]] + transfer_splits(transfer_config)
    for split in splits:
        for dec in decodings:
            if dec.subset is None and not (split == VAL_SPLIT and dec.name == "greedy"):
                if split in cfg.get("greedy_only", []) and dec.name != "greedy":
                    continue
                units.append(EvalUnit(model, split, dec, get(split)))
    for dec in decodings:
        if dec.subset is not None:
            from rlordata.sampling.eval_runner import parse_subset

            parent, first_n = parse_subset(dec.subset)
            units.append(EvalUnit(model, parent, dec, get(parent)[:first_n]))
    return units


def evaluate_run(cfg: dict[str, Any], run_dir: Path, args: Any) -> int:
    """``rlordata rft --stage eval --run-dir <run> --eval-set val|final``."""
    eval_set = str(getattr(args, "eval_set", None) or "val")
    stub = bool(getattr(args, "stub", False))
    force = bool(getattr(args, "force", False))
    seed = int(args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 1))
    splits_dir = Path(getattr(args, "splits_dir", None) or cfg.get("splits_dir", "data/splits"))
    pool_path = Path(cfg.get("pool", "data/pool/pool.jsonl"))
    ood_path = Path(cfg.get("ood_path", "data/pool/ood_hard_200.jsonl"))
    base_eval_dir = Path(
        getattr(args, "base_eval_dir", None) or cfg.get("base_eval_dir", DEFAULT_BASE_EVAL_DIR)
    )
    final_config = Path(cfg.get("final_eval_config", DEFAULT_FINAL_EVAL_CONFIG))
    transfer_config = Path(cfg.get("transfer_yaml", DEFAULT_TRANSFER_CONFIG))
    n_problems = getattr(args, "n_problems", None)
    if n_problems is not None and not stub:
        raise SystemExit("--n-problems is only for --stub dry runs")

    train_cfg = read_run_config(run_dir)
    budgets = read_json(run_dir / "budgets.json")
    adapter = Path(budgets["final_adapter"])
    if not adapter.is_absolute():
        adapter = Path(adapter)
    issues = check_final_checkpoint(run_dir, eval_set=eval_set) + check_adapter_nontrivial(adapter)
    if issues:
        raise SystemExit("[sanity] refusing to evaluate:\n  - " + "\n  - ".join(issues))

    units = plan_eval_units(
        eval_set=eval_set,
        model_id=cfg["model_id"],  # replaced by the merged dir below for the real sampler
        arm=cfg["arm"],
        splits_dir=splits_dir,
        pool_path=pool_path,
        ood_path=ood_path,
        final_config=final_config,
        transfer_config=transfer_config,
        n_problems=n_problems,
    )
    eval_root = run_dir / "eval" / eval_set
    pending = []
    for u in units:
        out = eval_root / u.split / u.decoding.name
        if (out / "samples.jsonl").exists() and not force:
            print(f"  [skip] {u.split}/{u.decoding.name}: already evaluated (use --force)")
            continue
        pending.append((u, out))
    if not pending:
        print("[eval] nothing to do")
        return 0
    cap = int(cfg["max_completion_tokens"])
    print_cost(
        "eval start estimate (worst case, every completion at the cap)",
        sum(len(u.problems) * u.decoding.n for u, _ in pending) * cap / 3000.0 / 3600.0,
    )

    sampler = None
    try:
        if stub:
            print("STUB SAMPLER: scripted completions, no GPU, no model weights. Not a result.")
            model_for_sampler = ModelSpec(id=cfg["model_id"], kind="base", arm=cfg["arm"])
        else:
            # Native vLLM LoRA; never a bf16 merge (module docstring).
            release_cuda_cache()  # a training stage may have run in this process
            model_for_sampler = ModelSpec(
                id=cfg["model_id"], kind="base", arm=cfg["arm"], lora_path=str(adapter)
            )
        sampler = make_sampler(
            model_for_sampler,
            stub=stub,
            cap=cap,
            seed=seed,
            cap_path=cfg["cap_yaml"],
            allow_provisional_cap=False,
            problems=[p for u, _ in pending for p in u.problems],
            dtype="bfloat16",
            gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
        )
        vllm_tok_sha = (
            tokenizer_hash(sampler.tokenizer) if not stub else train_cfg["tokenizer_sha256"]
        )
        if vllm_tok_sha != train_cfg["tokenizer_sha256"]:
            raise SystemExit(
                "[sanity] tokenizer mismatch: trainer "
                f"{train_cfg['tokenizer_sha256'][:12]} vs vLLM {vllm_tok_sha[:12]} (tasks/03 §3)"
            )
        for unit, out in pending:
            resolved = resolved_unit_config(
                unit,
                seed=seed,
                cap=cap,
                cap_path=cfg["cap_yaml"],
                sampler_desc=sampler.describe(),
                chat_kwargs=None,
                source_config=str(cfg.get("source_config")),
            )
            resolved.update(
                {
                    "kind": "rft_eval",
                    "arm": cfg["arm"],
                    "train_run_dir": str(run_dir),
                    "train_config_hash": (run_dir / "config_hash.txt").read_text().strip(),
                    "adapter": str(adapter),
                    "merged_model_dir": None,
                    "adapter_mode": ADAPTER_MODE,
                    "tokenizer_sha256_trainer": train_cfg["tokenizer_sha256"],
                    "tokenizer_sha256_vllm": vllm_tok_sha,
                    "eval_set": eval_set,
                    "seed_train": train_cfg["seed"],
                }
            )
            # The model id in the resolved config is the base id + adapter, not the temp dir.
            resolved["model"] = {
                "id": cfg["model_id"],
                "kind": "base",
                "arm": cfg["arm"],
                "lora_path": str(adapter),
            }
            base_unit_dir = base_eval_dir / unit.split / unit.decoding.name
            if base_unit_dir.exists():
                protocol_issues = _protocol_issues(resolved, base_unit_dir)
                if protocol_issues:
                    raise SystemExit(
                        "[sanity] protocol differs from the base eval:\n  - "
                        + "\n  - ".join(protocol_issues)
                    )
            run_id = (
                f"rft_eval_{cfg['arm']}_{run_dir.name}_{unit.split}_{unit.decoding.name}_seed{seed}"
            )
            resolved["run_id"] = run_id
            metrics = run_unit(
                unit, sampler, out, seed=seed, resolved=resolved, chat_kwargs=None, run_id=run_id
            )
            if unit.split == VAL_SPLIT and unit.decoding.name == "greedy":
                base_samples_path = base_eval_dir / VAL_SPLIT / "greedy" / "samples.jsonl"
                if base_samples_path.exists():
                    diff_issues = check_outputs_differ(
                        read_samples(base_samples_path),
                        read_samples(out / "samples.jsonl"),
                        min_fraction=MIN_DIFFERENT_FRACTION,
                    )
                    if diff_issues:
                        (out / "metrics.json").unlink(missing_ok=True)
                        raise SystemExit(
                            "[sanity] adapter has no measurable effect; metrics withheld:\n  - "
                            + "\n  - ".join(diff_issues)
                        )
                elif not stub:
                    raise SystemExit(
                        f"[sanity] base val greedy samples not found at {base_samples_path}; "
                        "restore runs/eval from the store before evaluating adapters"
                    )
            write_json(
                out / "sanity.json",
                {"protocol_vs_base": "ok", "adapter": str(adapter), "eval_set": eval_set},
            )
            _ = metrics
    finally:
        if sampler is not None and hasattr(sampler, "close"):
            sampler.close()
        sync_run(run_dir)
    _write_eval_summary(run_dir, eval_set)
    return 0


def _protocol_issues(resolved: dict[str, Any], base_unit_dir: Path) -> list[str]:
    """Reuse :func:`check_protocol_identical` by writing the resolved config to a temp dir."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg"
        p.mkdir()
        with (p / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=True)
        return check_protocol_identical([p, base_unit_dir])


def _write_eval_summary(run_dir: Path, eval_set: str) -> None:
    root = run_dir / "eval" / eval_set
    rows = {}
    for m in sorted(root.glob("*/*/metrics.json")):
        d = read_json(m)
        rows[f"{d['split']}/{d['decoding']}"] = {
            k: d.get(k)
            for k in (
                "accuracy",
                "ci_low",
                "ci_high",
                "truncation_rate",
                "extraction_failure_rate",
                "answer_line_rate",
                "pass_at_k",
                "per_tier",
                "n_problems",
                "flags",
            )
        }
    write_json(root / "summary.json", rows)
    for k, r in rows.items():
        print(
            f"  [{eval_set}] {k}: acc {r['accuracy']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] "
            f"trunc {100 * r['truncation_rate']:.1f}% extract-fail {100 * r['extraction_failure_rate']:.1f}%"
        )


def val_greedy_accuracy(run_dir: Path) -> dict[str, Any]:
    """The number the sweep selects on: val_mixed_100 greedy of a run's final adapter."""
    m = run_dir / "eval" / "val" / VAL_SPLIT / "greedy" / "metrics.json"
    if not m.exists():
        raise FileNotFoundError(f"{m} (run `rlordata rft --stage eval --eval-set val` first)")
    return read_json(m)
