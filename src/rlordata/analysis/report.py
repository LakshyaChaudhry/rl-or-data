"""tasks/05: tables, paired contrasts, the 'how could this be wrong' list and the grading sheet.

    uv run python -m rlordata.analysis.report --config configs/analysis/default.yaml
    make analysis [RUN_ROOT=/path/to/runs] [OUT=outputs]

Reads the run root (never writes to it), runs the cross-run sanity checks and aborts on any
failure, then writes — deterministically, byte for byte, from the run dirs and the config —

    <out>/sanity.md, sanity.json
    <out>/tables/{results.md, units.csv, arms.csv, per_tier.csv, budgets.csv, truncation.csv}
    <out>/tables/{contrasts.md, contrasts.csv}
    <out>/how_could_this_be_wrong.md
    <out>/hypotheses.md                  (verdict lines blank: Laksh grades)
    <out>/figures/*.png                  (analysis.plots)
    <out>/analysis_config.yaml           (resolved config + hash)
    <out>/analysis_meta.json             (git SHA, package versions, wall-clock: provenance only,
                                          the one file that is not byte-reproducible)

Nothing here branches on a test_300 or ood_hard_200 number: every contrast, column and figure is
fixed in advance by ``CONTRASTS`` / the config, and flags only annotate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.analysis import loader, sanity, stats
from rlordata.analysis.loader import Dataset, Run, Unit
from rlordata.core.evaluate import bootstrap_ci
from rlordata.data.generator import read_jsonl

MISSING = "missing"
FLAG = "⚑"  # SPEC §7: truncation > 5 % on val/test — not a headline number
TIERS = ("easy", "medium", "hard")

# (metric key, split, decoding, pass@k or None, column title)
METRICS: tuple[tuple[str, str, str, int | None, str], ...] = (
    ("val_greedy", "val_mixed_100", "greedy", None, "val greedy"),
    ("test_greedy", "test_300", "greedy", None, "test greedy"),
    ("ood_greedy", "ood_hard_200", "greedy", None, "ood greedy"),
    ("test_mean8", "test_300", "mean_at_k", None, "test mean@8"),
    ("ood_mean8", "ood_hard_200", "mean_at_k", None, "ood mean@8"),
    ("test_pass8", "test_300", "pass_at_k", 8, "pass@8 (test first-100, 64 samples)"),
    ("test_pass64", "test_300", "pass_at_k", 64, "pass@64 (test first-100, 64 samples)"),
    ("gsm8k_greedy", "gsm8k_500", "greedy", None, "gsm8k greedy"),
    ("gsm8k_mean8", "gsm8k_500", "mean_at_k", None, "gsm8k mean@8"),
)
METRIC_BY_KEY = {m[0]: m for m in METRICS}
PRIMARY = ("test_greedy", "ood_greedy")  # SPEC §10
SECONDARY = ("val_greedy", "test_mean8", "ood_mean8", "test_pass8", "test_pass64", "gsm8k_greedy")


@dataclass(frozen=True)
class ContrastSpec:
    key: str
    hypothesis: str  # H1 | H2 | H3 | controls | aux
    a: str  # arm/control key
    b: str  # arm key or "base"
    headline: bool  # gets a 'how could this be wrong' list


# Fixed in advance by tasks/05 item 3 (plus one SPEC §8 auxiliary); never chosen from results.
CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec("H1_rft", "H1", "rft_mixed", "rft_easy", True),
    ContrastSpec("H1_grpo", "H1", "grpo_mixed", "grpo_easy", True),
    ContrastSpec("H2_num", "H2", "rft_curated", "rft_mixed", True),
    ContrastSpec("H2_den", "H2", "grpo_curated", "rft_mixed", True),
    ContrastSpec("H3", "H3", "grpo_curated", "rft_curated", True),
    ContrastSpec("C1", "controls", "c1_random_reward", "base", False),
    ContrastSpec("C2", "controls", "c2_format_only", "base", False),
    ContrastSpec("aux_grpo_curation", "aux", "grpo_curated", "grpo_mixed", False),
)

# tasks/06b: registered AFTER unblinding (PREREGISTRATION §6, 2026-09-20), so never part of the
# confirmatory H1–H3 analysis. Evaluated only when the arm is configured; appended after CONTRASTS so
# every confirmatory row keeps its place. H3 = S2 + S1 exactly.
SECONDARY_HYPOTHESIS = "secondary (registered after unblinding)"
SECONDARY_CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec("S1_iter_vs_rft", SECONDARY_HYPOTHESIS, "iter_rft_curated", "rft_curated", True),
    ContrastSpec("S2_grpo_vs_iter", SECONDARY_HYPOTHESIS, "grpo_curated", "iter_rft_curated", True),
)


def active_contrasts(ds: Dataset) -> tuple[ContrastSpec, ...]:
    """CONTRASTS plus the secondary contrasts whose arms are configured."""
    have = {"base", *ds.arms, *ds.controls}
    return (*CONTRASTS, *(c for c in SECONDARY_CONTRASTS if {c.a, c.b} <= have))


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------


def _vector(unit: Unit, k: int | None) -> np.ndarray:
    """[P] per-problem score for a metric: mean(correct), or unbiased pass@k."""
    return unit.scores if k is None else unit.pass_at(k)


def unit_cell(run: Run, metric: str, *, n_boot: int) -> dict[str, Any] | None:
    """One number with everything CLAUDE.md requires next to it; None when the unit is missing."""
    _, split, dec, k, _ = METRIC_BY_KEY[metric]
    u = run.units.get((split, dec))
    if u is None:
        return None
    m = u.metrics
    if k is None:
        value, lo, hi = m["accuracy"], m["ci_low"], m["ci_high"]
    else:
        value = m["pass_at_k"][str(k)]
        lo, hi = bootstrap_ci(u.pass_at(k), n_boot=n_boot, seed=u.seed)
    return {
        "value": float(value),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_problems": int(m["n_problems"]),
        "samples_per_problem": m["samples_per_problem"],
        "seed": run.seed,
        "truncation_rate": float(m["truncation_rate"]),
        "extraction_failure_rate": float(m["extraction_failure_rate"]),
        "answer_line_rate": float(m["answer_line_rate"]),
        "mean_completion_tokens": float(m["mean_completion_tokens"]),
        "frac_correct_over_2048": u.frac_correct_over_2048,
        "truncation_flag": bool(m["flags"]["truncation_gt_5pct"]),
        "config_hash": u.config_hash,
        "git_dirty": bool(u.meta.get("git_dirty")),
        "unit": u.rel,
    }


def arm_cell(runs: Sequence[Run], metric: str, *, n_boot: int, seed: int) -> dict[str, Any] | None:
    """mean ± std across seeds, every seed kept, CI over problems of the seed-averaged score."""
    _, split, dec, k, _ = METRIC_BY_KEY[metric]
    units = [r.units.get((split, dec)) for r in runs]
    if not units or any(u is None for u in units):
        return None
    assert len({u.problem_ids for u in units}) == 1, "seeds were scored on different problems"
    s = stats.summarize_arm(
        [r.seed for r in runs], [_vector(u, k) for u in units], n_boot=n_boot, seed=seed
    )
    trunc = [float(u.metrics["truncation_rate"]) for u in units]
    xfail = [float(u.metrics["extraction_failure_rate"]) for u in units]
    return {
        **asdict(s),
        "truncation_per_seed": trunc,
        "truncation_mean": float(np.mean(trunc)),
        "extraction_failure_per_seed": xfail,
        "extraction_failure_mean": float(np.mean(xfail)),
        "answer_line_rate_mean": float(np.mean([u.metrics["answer_line_rate"] for u in units])),
        "mean_completion_tokens": float(
            np.mean([u.metrics["mean_completion_tokens"] for u in units])
        ),
        "flagged_seeds": [
            r.seed
            for r, u in zip(runs, units, strict=True)
            if u.metrics["flags"]["truncation_gt_5pct"]
        ],
    }


def tier_cells(run: Run, *, n_boot: int) -> dict[str, dict[str, Any]]:
    """Per-tier greedy accuracy on test_300 with a bootstrap CI over that tier's problems."""
    u = run.units.get(("test_300", "greedy"))
    if u is None:
        return {}
    tiers = np.asarray(u.tiers)
    out = {}
    for t in TIERS:
        mask = tiers == t
        if not mask.any():
            continue
        lo, hi = bootstrap_ci(u.scores[mask], n_boot=n_boot, seed=u.seed)
        out[t] = {
            "value": float(u.scores[mask].mean()),
            "ci_low": lo,
            "ci_high": hi,
            "n_problems": int(mask.sum()),
            "seed": run.seed,
            "truncation_rate": float(u.metrics["per_tier_truncation_rate"][t]),
            "extraction_failure_rate": float(u.metrics["per_tier_extraction_failure_rate"][t]),
        }
    return out


def arm_tier_cells(runs: Sequence[Run], *, n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    units = [r.units.get(("test_300", "greedy")) for r in runs]
    if not units or any(u is None for u in units):
        return {}
    tiers = np.asarray(units[0].tiers)
    out = {}
    for t in TIERS:
        mask = tiers == t
        if not mask.any():
            continue
        s = stats.summarize_arm(
            [r.seed for r in runs], [u.scores[mask] for u in units], n_boot=n_boot, seed=seed
        )
        out[t] = {
            **asdict(s),
            "truncation_mean": float(
                np.mean([u.metrics["per_tier_truncation_rate"][t] for u in units])
            ),
        }
    return out


def run_budget(run: Run, ds: Dataset) -> dict[str, Any]:
    b = run.budgets or {}
    eval_h = sum(float(u.meta.get("gpu_hours_actual") or 0.0) for u in run.units.values())
    return {
        "who": f"{run.key}/seed{run.seed}",
        "label": run.label,
        "seed": run.seed,
        "data_condition": run.data_condition,
        "prompts": b.get("prompts"),
        "prompts_in_split": b.get("prompts_parent_split", b.get("prompts")),
        "completions_available": b.get("completions_available"),
        "completions_consumed": b.get("completions_consumed"),
        "training_tokens": b.get("training_tokens"),
        "optimizer_steps": b.get("optimizer_steps"),
        "learning_rate": b.get("learning_rate"),
        "epochs": b.get("epochs"),
        "per_tier_prompts": b.get("per_tier_prompts"),
        "per_tier_examples": b.get("per_tier_examples"),
        "gpu_hours_train": float((run.meta or {}).get("gpu_hours_actual") or 0.0),
        "gpu_hours_eval_final": eval_h,
        "config_hash": run.config_hash,
        "git_sha": str((run.meta or {}).get("git_sha", ""))[:7],
    }


def _arm_runs(ds: Dataset, key: str) -> list[Run]:
    if key == "base":
        return [ds.base]
    return ds.arms.get(key) or ds.controls[key]


def contrast_block(
    ds: Dataset, spec: ContrastSpec, metric: str, *, n_boot: int, seed: int
) -> dict[str, Any] | None:
    _, split, dec, k, _ = METRIC_BY_KEY[metric]
    ra, rb = _arm_runs(ds, spec.a), _arm_runs(ds, spec.b)
    ua = [r.units.get((split, dec)) for r in ra]
    ub = [r.units.get((split, dec)) for r in rb]
    if any(u is None for u in (*ua, *ub)):
        return None
    assert len({u.problem_ids for u in (*ua, *ub)}) == 1, f"{spec.key}: different problems"
    if len(rb) > 1:
        assert [r.seed for r in ra] == [r.seed for r in rb], f"{spec.key}: seeds are not paired"
    c = stats.paired_contrast(
        [r.seed for r in ra],
        [_vector(u, k) for u in ua],
        [_vector(u, k) for u in ub],
        n_boot=n_boot,
        seed=seed,
    )

    def rates(units: list[Unit], field: str) -> list[float]:
        return [float(u.metrics[field]) for u in units]

    ta, tb = rates(ua, "truncation_rate"), rates(ub, "truncation_rate")
    xa, xb = rates(ua, "extraction_failure_rate"), rates(ub, "extraction_failure_rate")
    return {
        **asdict(c),
        "metric": metric,
        "split": split,
        "truncation_a_per_seed": ta,
        "truncation_b_per_seed": tb,
        "truncation_gap": float(np.mean(ta) - np.mean(tb)),
        "n_truncated_a_per_seed": [
            int(round(t * u.metrics["n_samples"])) for t, u in zip(ta, ua, strict=True)
        ],
        "n_truncated_b_per_seed": [
            int(round(t * u.metrics["n_samples"])) for t, u in zip(tb, ub, strict=True)
        ],
        "n_samples_per_run": int(ua[0].metrics["n_samples"]),
        "extraction_failure_a_per_seed": xa,
        "extraction_failure_b_per_seed": xb,
        "extraction_failure_gap": float(np.mean(xa) - np.mean(xb)),
        "answer_line_gap": float(
            np.mean(rates(ua, "answer_line_rate")) - np.mean(rates(ub, "answer_line_rate"))
        ),
        "mean_tokens_a": float(np.mean(rates(ua, "mean_completion_tokens"))),
        "mean_tokens_b": float(np.mean(rates(ub, "mean_completion_tokens"))),
        "flagged_seeds_a": [
            r.seed for r, u in zip(ra, ua, strict=True) if u.metrics["flags"]["truncation_gt_5pct"]
        ],
        "flagged_seeds_b": [
            r.seed for r, u in zip(rb, ub, strict=True) if u.metrics["flags"]["truncation_gt_5pct"]
        ],
        "git_dirty_units": [u.rel for u in (*ua, *ub) if u.meta.get("git_dirty")],
    }


def _train_tier_counts(ds: Dataset, data_condition: str | None) -> dict[str, int] | None:
    if not data_condition:
        return None
    path = Path(ds.cfg["splits_dir"]) / f"{data_condition}.jsonl"
    if not path.exists():
        return None
    counts: dict[str, int] = {}
    for p in read_jsonl(path):
        counts[p.tier] = counts.get(p.tier, 0) + 1
    return dict(sorted(counts.items()))


def build_results(ds: Dataset) -> dict[str, Any]:
    """Every number the outputs show, as one JSON-serialisable tree."""
    cfg = ds.cfg
    n_boot, seed = int(cfg["n_boot"]), int(cfg["seed"])
    res: dict[str, Any] = {"runs": {}, "arms": {}, "budgets": {}, "contrasts": {}}
    for run in ds.all_runs():
        who = f"{run.key}/seed{run.seed}" if run.kind in ("arm", "control") else run.key
        res["runs"][who] = {
            "key": run.key,
            "label": run.label,
            "kind": run.kind,
            "method": run.method,
            "data_condition": run.data_condition,
            "seed": run.seed,
            "cells": {m[0]: unit_cell(run, m[0], n_boot=n_boot) for m in METRICS},
            "tiers": tier_cells(run, n_boot=n_boot),
            "pass_at_k": (
                run.units[("test_300", "pass_at_k")].metrics["pass_at_k"]
                if ("test_300", "pass_at_k") in run.units
                else None
            ),
        }
        if run.kind in ("arm", "control"):
            res["budgets"][who] = run_budget(run, ds)
    for key, runs in {**ds.arms, **ds.controls}.items():
        res["arms"][key] = {
            "label": runs[0].label,
            "kind": runs[0].kind,
            "method": runs[0].method,
            "data_condition": runs[0].data_condition,
            "train_tier_counts": _train_tier_counts(ds, runs[0].data_condition),
            "seeds": [r.seed for r in runs],
            "cells": {m[0]: arm_cell(runs, m[0], n_boot=n_boot, seed=seed) for m in METRICS},
            "tiers": arm_tier_cells(runs, n_boot=n_boot, seed=seed),
            "selection": ds.selection.get(key),
        }
    for spec in active_contrasts(ds):
        blocks = {
            m: contrast_block(ds, spec, m, n_boot=n_boot, seed=seed) for m in (*PRIMARY, *SECONDARY)
        }
        crit = None
        if blocks["test_greedy"] and blocks["ood_greedy"]:
            crit = asdict(
                stats.evaluate_criterion(
                    stats.Contrast(**_contrast_fields(blocks["test_greedy"])),
                    stats.Contrast(**_contrast_fields(blocks["ood_greedy"])),
                )
            )
        res["contrasts"][spec.key] = {"spec": asdict(spec), "metrics": blocks, "criterion": crit}
    res["vs_base"] = {
        key: contrast_block(
            ds,
            ContrastSpec(f"{key}_vs_base", "aux", key, "base", False),
            "test_greedy",
            n_boot=n_boot,
            seed=seed,
        )
        for key in (*ds.arms, *ds.controls)
    }
    # H2 ratio: (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed)
    res["h2_ratio"] = {}
    for metric in PRIMARY:
        _, split, dec, k, _ = METRIC_BY_KEY[metric]
        vecs = {
            a: [_vector(r.units[(split, dec)], k) for r in ds.arms[a]]
            for a in ("rft_curated", "grpo_curated", "rft_mixed")
        }
        res["h2_ratio"][metric] = asdict(
            stats.ratio_of_contrasts(
                vecs["rft_curated"],
                vecs["grpo_curated"],
                vecs["rft_mixed"],
                n_boot=n_boot,
                seed=seed,
            )
        )
    return res


def _contrast_fields(block: dict[str, Any]) -> dict[str, Any]:
    names = stats.Contrast.__dataclass_fields__
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in block.items() if k in names}


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def f3(x: float | None) -> str:
    return MISSING if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def sgn(x: float | None) -> str:
    return MISSING if x is None or math.isnan(x) else f"{x:+.3f}"


def pct(x: float | None) -> str:
    return MISSING if x is None else f"{100 * x:.1f}%"


def fmt_cell(c: dict[str, Any] | None) -> str:
    """'0.793 [0.747, 0.840] · tr 5.3% ⚑' — value, bootstrap 95 % CI, truncation, §7 flag."""
    if c is None:
        return MISSING
    flag = f" {FLAG}" if c.get("truncation_flag") else ""
    return f"{c['value']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] · tr {pct(c['truncation_rate'])}{flag}"


def fmt_arm_cell(c: dict[str, Any] | None) -> str:
    """'0.778 ± 0.017 [0.741, 0.812] · tr 7.6%' — mean ± seed std, CI of the seed mean, mean truncation."""
    if c is None:
        return MISSING
    flag = f" {FLAG}×{len(c['flagged_seeds'])}" if c["flagged_seeds"] else ""
    return (
        f"{c['mean']:.3f} ± {c['std']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] · "
        f"tr {pct(c['truncation_mean'])}{flag}"
    )


def fmt_seeds(values: Iterable[float], signed: bool = False) -> str:
    return " / ".join((sgn(v) if signed else f3(v)) for v in values)


def fmt_pcts(values: Iterable[float]) -> str:
    return " / ".join(pct(v) for v in values)


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _n_of(results: dict[str, Any], metric: str) -> str:
    for run in results["runs"].values():
        c = run["cells"].get(metric)
        if c:
            return str(c["n_problems"])
    return "?"


LEGEND = (
    "Cell format: `value [bootstrap 95 % CI over problems, 10,000 resamples] · tr = share of "
    f"completions cut at the 4,352-token cap`; {FLAG} = truncation > 5 % on val/test (SPEC §7: flagged, "
    "**not a headline number as it stands**; on ood truncation is reported, never flagged). Arm rows: "
    "`mean ± sample std across seeds [CI of the seed-averaged score] · mean truncation`, "
    f"{FLAG}×k = k seeds flagged. `{MISSING}` = the unit was never evaluated (not zero). A truncated "
    "completion is scored wrong by rule (SPEC §5 v1.6). Seeds 1–3 unless stated; base and reference "
    "models are single evaluations at seed 1."
)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def results_markdown(ds: Dataset, results: dict[str, Any], cfg_hash: str) -> str:
    runs, arms = results["runs"], results["arms"]
    out = [
        "# Results tables (tasks/05 item 2)",
        "",
        f"Generated by `rlordata.analysis.report` from `{ds.cfg['run_root']}`; analysis config hash "
        f"`{cfg_hash[:12]}`; bootstrap seed {ds.cfg['seed']}, {ds.cfg['n_boot']:,} resamples. No "
        "arm-vs-arm claim is made in this file; contrasts are in `contrasts.md`, grading in "
        "`../hypotheses.md`.",
        "",
        LEGEND,
        "",
        "## 1. All arms: mean ± std across seeds, with base, controls and reference models",
        "",
    ]
    cols = ("val_greedy", "test_greedy", "ood_greedy", "test_mean8", "test_pass8", "test_pass64")
    header = ["arm", "seeds", *[f"{METRIC_BY_KEY[m][4]} (n={_n_of(results, m)})" for m in cols]]
    rows = []
    rows.append(["Base", "1 eval (seed 1)", *[fmt_cell(runs["base"]["cells"][m]) for m in cols]])
    for key, a in arms.items():
        if len(a["seeds"]) > 1:
            rows.append(
                [
                    a["label"],
                    ",".join(map(str, a["seeds"])),
                    *[fmt_arm_cell(a["cells"][m]) for m in cols],
                ]
            )
        else:
            who = f"{key}/seed{a['seeds'][0]}"
            rows.append(
                [
                    a["label"],
                    f"{a['seeds'][0]} (single seed)",
                    *[fmt_cell(runs[who]["cells"][m]) for m in cols],
                ]
            )
    for ref in ds.references:
        rows.append(
            [
                f"ref: {ref.label}",
                "1 eval (seed 1)",
                *[fmt_cell(runs[ref.key]["cells"][m]) for m in cols],
            ]
        )
    out += [md_table(header, rows), ""]

    out += ["## 2. Per arm: every seed as a row, then mean ± std", ""]
    for key, a in arms.items():
        sel = a.get("selection")
        note = ""
        if sel:
            ch = sel["chosen"]
            note = (
                f" — config lr={ch['learning_rate']:g}, epochs={ch['epochs']}: best of "
                f"{sel['n_configs_tried']} on val_mixed_100 (seed 1, n=100, val {ch['val_accuracy']:.3f}); "
                "'best on val', not 'optimal'"
            )
        elif a["method"] == "grpo":
            note = " — one fixed recipe (SPEC §9), nothing tuned"
        elif a["method"] == "iter_rft":
            b0 = results["budgets"][f"{key}/seed{a['seeds'][0]}"]
            note = (
                " — SECONDARY arm, registered after unblinding (PREREGISTRATION §6, 2026-09-20): 3 rounds × "
                f"64 samples per prompt, each round continues the previous adapter; no sweep, lr={b0['learning_rate']:g}, "
                f"epochs={b0['epochs']} reused from RFT-Curated; final = after round 3"
            )
        out += [f"### {a['label']} ({a['data_condition']}){note}", ""]
        header = [
            "seed",
            *[f"{METRIC_BY_KEY[m][4]} (n={_n_of(results, m)})" for m in cols],
            "config hash",
        ]
        rows = []
        for s in a["seeds"]:
            r = runs[f"{key}/seed{s}"]
            rows.append(
                [
                    s,
                    *[fmt_cell(r["cells"][m]) for m in cols],
                    str(results["budgets"][f"{key}/seed{s}"]["config_hash"])[:12],
                ]
            )
        if len(a["seeds"]) > 1:
            rows.append(["**μ ± σ**", *[fmt_arm_cell(a["cells"][m]) for m in cols], ""])
        out += [md_table(header, rows), ""]
        # secondary block: ood mean@8, gsm8k, rates on the primary unit
        header = [
            "seed",
            f"ood mean@8 (n={_n_of(results, 'ood_mean8')})",
            f"gsm8k greedy (n={_n_of(results, 'gsm8k_greedy')})",
            f"gsm8k mean@8 (n={_n_of(results, 'gsm8k_mean8')})",
            "test greedy: extraction-fail",
            "answer-line rate",
            "correct completions > 2048 tok",
            "mean tokens",
            "ood greedy: extraction-fail",
            "mean tokens (ood)",
        ]
        rows = []
        for s in a["seeds"]:
            c = runs[f"{key}/seed{s}"]["cells"]
            t, o = c["test_greedy"], c["ood_greedy"]
            rows.append(
                [
                    s,
                    fmt_cell(c["ood_mean8"]),
                    fmt_cell(c["gsm8k_greedy"]),
                    fmt_cell(c["gsm8k_mean8"]),
                    pct(t["extraction_failure_rate"]),
                    pct(t["answer_line_rate"]),
                    pct(t["frac_correct_over_2048"]),
                    f"{t['mean_completion_tokens']:.0f}",
                    pct(o["extraction_failure_rate"]),
                    f"{o['mean_completion_tokens']:.0f}",
                ]
            )
        out += [md_table(header, rows), ""]

    out += ["## 3. Base and reference models (single evaluation each, seed 1)", ""]
    allm = [m[0] for m in METRICS]
    header = ["model", *[f"{METRIC_BY_KEY[m][4]} (n={_n_of(results, m)})" for m in allm]]
    rows = [["Base", *[fmt_cell(runs["base"]["cells"][m]) for m in allm]]]
    for ref in ds.references:
        rows.append([f"ref: {ref.label}", *[fmt_cell(runs[ref.key]["cells"][m]) for m in allm]])
    out += [
        md_table(header, rows),
        "",
        "Reference models were evaluated on the counting splits only. `google/gemma-4-E4B-it` has never "
        "been evaluated for real (stub-sampler units only) and is excluded. Base gsm8k_500 mean@8 was "
        "never run.",
        "",
    ]

    out += [
        "## 4. Per-tier greedy accuracy on test_300 (tiers by the base model's pass@8; n = 100 per tier)",
        "",
    ]
    header = ["arm", "seed", *[f"{t}" for t in TIERS]]
    rows = [["Base", 1, *[fmt_cell(runs["base"]["tiers"].get(t)) for t in TIERS]]]
    for key, a in arms.items():
        for s in a["seeds"]:
            rows.append(
                [a["label"], s, *[fmt_cell(runs[f"{key}/seed{s}"]["tiers"].get(t)) for t in TIERS]]
            )
        if len(a["seeds"]) > 1:
            rows.append(
                [
                    a["label"],
                    "**μ ± σ**",
                    *[
                        (
                            f"{a['tiers'][t]['mean']:.3f} ± {a['tiers'][t]['std']:.3f} "
                            f"[{a['tiers'][t]['ci_low']:.3f}, {a['tiers'][t]['ci_high']:.3f}] · tr {pct(a['tiers'][t]['truncation_mean'])}"
                        )
                        for t in TIERS
                    ],
                ]
            )
    for ref in ds.references:
        rows.append(
            [f"ref: {ref.label}", 1, *[fmt_cell(runs[ref.key]["tiers"].get(t)) for t in TIERS]]
        )
    out += [md_table(header, rows), ""]

    out += [
        "## 5. Truncation by arm, split and decoding (per seed; the systematic RFT-vs-GRPO gap)",
        "",
    ]
    tcols = ("val_greedy", "test_greedy", "test_mean8", "test_pass8", "ood_greedy", "ood_mean8")
    names = {"test_pass8": "test pass@k unit"}
    header = ["arm", *[names.get(m, METRIC_BY_KEY[m][4]) for m in tcols]]
    rows = [
        ["Base", *[pct((runs["base"]["cells"][m] or {}).get("truncation_rate")) for m in tcols]]
    ]
    for a in arms.values():
        rows.append(
            [
                f"{a['label']} (seeds {','.join(map(str, a['seeds']))})",
                *[
                    fmt_pcts(a["cells"][m]["truncation_per_seed"]) if a["cells"][m] else MISSING
                    for m in tcols
                ],
            ]
        )
    out += [
        md_table(header, rows),
        "",
        "Every completion counted here was scored wrong regardless of content (SPEC §5 v1.6). n per cell "
        "is the unit's sample count: val 100, test greedy 300, test mean@8 2,400, pass@k 6,400, ood greedy "
        "200, ood mean@8 1,600.",
        "",
    ]

    out += _budget_section(ds, results)
    return "\n".join(out).rstrip() + "\n"


def _curated_available_note(results: dict[str, Any]) -> list[str]:
    """RFT-Curated's recorded 'available' budget vs the samples that belong to its prompts."""
    b = next(
        (
            b
            for b in results["budgets"].values()
            if b["data_condition"] == "train_curated" and b["epochs"] is not None
        ),
        None,
    )
    if b is None or b["completions_available"] == b["prompts"] * 192:
        return []
    return [
        f"RFT-Curated's budgets.json records {b['completions_available']:,} completions available (the whole "
        f"train_mixed_100 draw); the samples belonging to its {b['prompts']} prompts number "
        f"{b['prompts'] * 192:,} (SPEC §8.2: 192 per prompt). RFT 'prompts used' counts prompts with at "
        "least one verified-correct sample.",
        "",
    ]


def _budget_section(ds: Dataset, results: dict[str, Any]) -> list[str]:
    rate = ds.cfg["gpu_rate"]
    billed, recorded = float(rate["billed_usd_per_hour"]), float(rate["recorded_usd_per_hour"])
    out = [
        "## 6. Budgets (SPEC §8: prompts, completions available/consumed, gradient/token) and GPU-hours",
        "",
    ]
    header = [
        "run", "prompts used / in split", "completions available", "consumed", "training tokens",
        "optimizer steps", "lr", "epochs", "train GPU-h", "final-eval GPU-h", "git SHA",
    ]  # fmt: skip
    rows = []
    for b in results["budgets"].values():
        rows.append(
            [
                b["who"],
                f"{b['prompts']} / {b['prompts_in_split']}",
                f"{b['completions_available']:,}",
                f"{b['completions_consumed']:,}",
                f"{b['training_tokens']:,}",
                f"{b['optimizer_steps']:,}",
                "fixed 5e-05" if b["learning_rate"] is None else f"{b['learning_rate']:g}",
                "—" if b["epochs"] is None else b["epochs"],
                f"{b['gpu_hours_train']:.2f}",
                f"{b['gpu_hours_eval_final']:.2f}",
                b["git_sha"],
            ]
        )
    out += [md_table(header, rows), ""]
    header = [
        "arm", "configs tried", "training tokens (mean/seed)", "optimizer steps (mean/seed)",
        "completions consumed (mean/seed)", "GPU-h, result-bearing runs (train + final eval)",
        "GPU-h, the other sweep configs (train + val eval)",
        f"test greedy gain vs base per 100 prompts / per 1,000 completions consumed (n={_n_of(results, 'test_greedy')})",
    ]  # fmt: skip
    rows = []
    total = 0.0
    for key, a in results["arms"].items():
        bs = [results["budgets"][f"{key}/seed{s}"] for s in a["seeds"]]
        sel = a.get("selection")
        own = sum(b["gpu_hours_train"] + b["gpu_hours_eval_final"] for b in bs)
        other = (
            0.0
            if not sel
            else sel["other_configs_train_gpu_hours"] + sel["other_configs_val_eval_gpu_hours"]
        )
        total += own + other
        cell = a["cells"]["test_greedy"]
        vb = results["vs_base"][key]
        gain = vb["mean_delta"]
        prompts = float(np.mean([b["prompts"] for b in bs]))
        consumed = float(np.mean([b["completions_consumed"] for b in bs]))
        rows.append(
            [
                a["label"],
                sel["n_configs_tried"] if sel else 1,
                f"{np.mean([b['training_tokens'] for b in bs]):,.0f}",
                f"{np.mean([b['optimizer_steps'] for b in bs]):,.0f}",
                f"{consumed:,.0f}",
                f"{own:.2f}",
                f"{other:.2f}" if sel else "—",
                f"{sgn(100 * gain / prompts)} / {sgn(1000 * gain / consumed)} "
                f"(gain {sgn(gain)} [{sgn(vb['ci_low'])}, {sgn(vb['ci_high'])}], seeds {','.join(map(str, a['seeds']))}, tr {pct(cell['truncation_mean'])}"
                f"{' ' + FLAG if cell['flagged_seeds'] else ''})",
            ]
        )
    out += [
        md_table(header, rows),
        "",
        f"Total for the rows above: **{total:.2f} GPU-hours** on 1× H100 PCIe (the shared 192-sample RFT "
        "draw, 0.58 GPU-h, and GRPO checkpoint val evals are not included). GPU-hours are the primary "
        f"quantity. In dollars: ${total * billed:,.0f} at the ${billed:g}/h Lambda billed; the "
        f"`cost_usd_actual` fields in meta.json use ${recorded:g}/h and would give ${total * recorded:,.0f}.",
        "",
        *_curated_available_note(results),
        "Budget asymmetries that are part of the design (SPEC §8.3) and still have to be read next to "
        "every RFT-vs-GRPO number: RFT discards incorrect completions (consumed < available) and takes "
        "many more optimizer steps over several epochs; GRPO trains once on all 19,200. RFT tried 9 "
        "configurations per arm on val; GRPO ran one fixed recipe (PREREGISTRATION §5.2).",
        "",
    ]
    return out


def write_csvs(ds: Dataset, results: dict[str, Any], tables: Path) -> None:
    def dump(name: str, header: list[str], rows: list[list[Any]]) -> None:
        with (tables / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    unit_fields = [
        "value", "ci_low", "ci_high", "n_problems", "samples_per_problem", "truncation_rate",
        "extraction_failure_rate", "answer_line_rate", "frac_correct_over_2048",
        "mean_completion_tokens", "truncation_flag", "config_hash", "git_dirty", "unit",
    ]  # fmt: skip
    rows = []
    for who, r in results["runs"].items():
        for m in METRICS:
            c = r["cells"][m[0]]
            base = [
                who,
                r["label"],
                r["kind"],
                r["method"],
                r["data_condition"],
                r["seed"],
                m[0],
                m[1],
                m[2],
            ]
            rows.append(
                base + ([MISSING] * len(unit_fields) if c is None else [c[k] for k in unit_fields])
            )
    dump(
        "units.csv",
        [
            "who",
            "label",
            "kind",
            "method",
            "data_condition",
            "seed",
            "metric",
            "split",
            "decoding",
            *unit_fields,
        ],
        rows,
    )

    arm_fields = [
        "mean", "std", "ci_low", "ci_high", "n_problems", "seed_min", "seed_max",
        "truncation_mean", "extraction_failure_mean", "answer_line_rate_mean", "mean_completion_tokens",
    ]  # fmt: skip
    rows = []
    for key, a in results["arms"].items():
        for m in METRICS:
            c = a["cells"][m[0]]
            if c is None:
                continue
            rows.append(
                [key, a["label"], m[0], len(c["seeds"]), " ".join(map(str, c["seeds"])),
                 " ".join(repr(v) for v in c["per_seed"]), *[c[k] for k in arm_fields],
                 " ".join(repr(v) for v in c["truncation_per_seed"]), " ".join(map(str, c["flagged_seeds"]))]
            )  # fmt: skip
    dump(
        "arms.csv",
        [
            "arm",
            "label",
            "metric",
            "n_seeds",
            "seeds",
            "per_seed",
            *arm_fields,
            "truncation_per_seed",
            "flagged_seeds",
        ],
        rows,
    )

    rows = []
    for who, r in results["runs"].items():
        for t, c in r["tiers"].items():
            rows.append(
                [
                    who,
                    r["label"],
                    r["seed"],
                    t,
                    c["value"],
                    c["ci_low"],
                    c["ci_high"],
                    c["n_problems"],
                    c["truncation_rate"],
                    c["extraction_failure_rate"],
                ]
            )
    dump(
        "per_tier.csv",
        [
            "who",
            "label",
            "seed",
            "tier",
            "test_greedy",
            "ci_low",
            "ci_high",
            "n_problems",
            "truncation_rate",
            "extraction_failure_rate",
        ],
        rows,
    )

    keys = [
        "who", "label", "seed", "data_condition", "prompts", "prompts_in_split", "completions_available",
        "completions_consumed", "training_tokens", "optimizer_steps", "learning_rate", "epochs",
        "gpu_hours_train", "gpu_hours_eval_final", "config_hash", "git_sha",
    ]  # fmt: skip
    dump("budgets.csv", keys, [[b[k] for k in keys] for b in results["budgets"].values()])

    rows = []
    for who, r in results["runs"].items():
        for m in METRICS:
            c = r["cells"][m[0]]
            if c is not None and m[3] in (None, 8):
                rows.append(
                    [
                        who,
                        r["seed"],
                        m[1],
                        m[2],
                        c["n_problems"],
                        c["truncation_rate"],
                        c["extraction_failure_rate"],
                        c["truncation_flag"],
                    ]
                )
    dump(
        "truncation.csv",
        [
            "who",
            "seed",
            "split",
            "decoding",
            "n_problems",
            "truncation_rate",
            "extraction_failure_rate",
            "spec7_flag",
        ],
        rows,
    )

    cfields = [
        "mean_delta", "std_delta", "pooled_seed_std", "ci_low", "ci_high", "n_problems",
        "seed_ranges_overlap", "truncation_gap", "extraction_failure_gap", "answer_line_gap",
    ]  # fmt: skip
    rows = []
    for key, c in results["contrasts"].items():
        for metric, b in c["metrics"].items():
            if b is None:
                continue
            crit = c["criterion"] if metric == "test_greedy" and c["criterion"] else {}
            rows.append(
                [key, c["spec"]["hypothesis"], c["spec"]["a"], c["spec"]["b"], metric,
                 " ".join(map(str, b["seeds"])), " ".join(repr(v) for v in b["delta_per_seed"]),
                 *[b[k] for k in cfields],
                 " ".join(repr(v) for v in b["truncation_a_per_seed"]),
                 " ".join(repr(v) for v in b["truncation_b_per_seed"]),
                 " ".join(map(str, b["flagged_seeds_a"])), " ".join(map(str, b["flagged_seeds_b"])),
                 crit.get("threshold", ""), crit.get("magnitude_ok", ""), crit.get("same_sign_on_ood", ""), crit.get("met", "")]
            )  # fmt: skip
    dump(
        "contrasts.csv",
        ["contrast", "hypothesis", "a", "b", "metric", "seeds", "delta_per_seed", *cfields,
         "truncation_a_per_seed", "truncation_b_per_seed", "spec7_flagged_seeds_a", "spec7_flagged_seeds_b",
         "criterion_threshold_2x_pooled_std", "criterion_a_magnitude", "criterion_b_same_sign_ood", "criterion_met"],
        rows,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# contrasts, wrong-list, grading sheet
# ---------------------------------------------------------------------------


def _label(results: dict[str, Any], key: str) -> str:
    return "Base" if key == "base" else results["arms"][key]["label"]


def _crit_text(crit: dict[str, Any] | None) -> str:
    if crit is None:
        return "not evaluable (a unit is missing)"
    if crit["met"] is None:
        return (
            "(a) **not evaluable**: a pooled seed std needs ≥ 2 seeds in both arms (this contrast has "
            f"one); (b) same sign on ood: **{'yes' if crit['same_sign_on_ood'] else 'no'}**"
        )
    return (
        f"(a) |Δ test| > 2 × pooled seed std = {crit['threshold']:.3f}: "
        f"**{'yes' if crit['magnitude_ok'] else 'no'}**; (b) same sign on ood: "
        f"**{'yes' if crit['same_sign_on_ood'] else 'no'}** → criterion "
        f"**{'MET' if crit['met'] else 'NOT MET'}** (mechanical evaluation, not a verdict)"
    )


def _flag_text(results: dict[str, Any], spec: dict[str, Any], block: dict[str, Any]) -> str:
    parts = []
    for side, key in (("a", spec["a"]), ("b", spec["b"])):
        seeds = block[f"flagged_seeds_{side}"]
        if seeds:
            parts.append(f"{_label(results, key)} seeds {','.join(map(str, seeds))}")
    if not parts:
        return "no run in this contrast exceeds 5 % truncation on test_300 greedy"
    return (
        f"{FLAG} SPEC §7: " + " and ".join(parts) + " exceed 5 % truncation on test_300 greedy — "
        "**not a headline number as it stands**"
    )


def _contrast_rows(results: dict[str, Any], key: str, metrics: Sequence[str]) -> list[list[str]]:
    c = results["contrasts"][key]
    rows = []
    for m in metrics:
        b = c["metrics"][m]
        if b is None:
            rows.append([METRIC_BY_KEY[m][4], MISSING, "", "", "", "", "", ""])
            continue
        single = len(b["seeds"]) < 2 or len(b["b_per_seed"]) < 2
        spread = "n/a (1 seed)" if len(b["seeds"]) < 2 else f3(b["std_delta"])
        rows.append(
            [
                f"{METRIC_BY_KEY[m][4]} (n={b['n_problems']})",
                ",".join(map(str, b["seeds"])),
                fmt_seeds(b["delta_per_seed"], signed=True),
                f"{sgn(b['mean_delta'])} ± {spread} [{sgn(b['ci_low'])}, {sgn(b['ci_high'])}]",
                "n/a (1 seed)" if single else f3(b["pooled_seed_std"]),
                f"{fmt_pcts(b['truncation_a_per_seed'])} vs {fmt_pcts(b['truncation_b_per_seed'])}",
                f"{100 * b['truncation_gap']:+.1f} pp",
                "n/a (1 seed)" if single else ("yes" if b["seed_ranges_overlap"] else "no"),
            ]
        )
    return rows


CONTRAST_HEADER = [
    "metric", "seeds", "Δ per seed (seed i − seed i)", "mean Δ ± seed std of Δ [paired bootstrap 95 % CI]",
    "pooled seed std", "truncation A vs B (per seed)", "truncation gap A − B", "seed ranges overlap",
]  # fmt: skip


def contrasts_markdown(ds: Dataset, results: dict[str, Any], cfg_hash: str) -> str:
    out = [
        "# Paired contrasts (tasks/05 item 3)",
        "",
        f"Analysis config hash `{cfg_hash[:12]}`; bootstrap seed {ds.cfg['seed']}, {ds.cfg['n_boot']:,} "
        "resamples. Δ = A − B, paired seed-wise. The CI is a paired per-problem bootstrap of the "
        "seed-averaged difference (both arms are scored on the same problems). Pooled seed std = "
        "sqrt of the pooled sample variance (ddof = 1) of the two arms' per-seed accuracies. The "
        "SPEC §10 criterion is evaluated mechanically on test_300 greedy and ood_hard_200 greedy; "
        "nothing here is a verdict. **Truncation is printed next to every Δ because it differs "
        "systematically between RFT and GRPO and a truncated completion is scored wrong by rule.**",
        "",
    ]
    for key, c in results["contrasts"].items():
        spec = c["spec"]
        out += [
            f"## {key} ({spec['hypothesis']}): {_label(results, spec['a'])} − {_label(results, spec['b'])}",
            "",
            md_table(CONTRAST_HEADER, _contrast_rows(results, key, (*PRIMARY, *SECONDARY))),
            "",
            f"- SPEC §10 criterion: {_crit_text(c['criterion'])}",
        ]
        if c["metrics"]["test_greedy"]:
            out.append(f"- {_flag_text(results, spec, c['metrics']['test_greedy'])}")
        out.append("")
    out += [
        "## H2 ratio: (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed)",
        "",
        _ratio_table(results),
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


def _ratio_table(results: dict[str, Any]) -> str:
    rows = []
    for metric, r in results["h2_ratio"].items():
        den = results["contrasts"]["H2_den"]["metrics"][metric]
        num = results["contrasts"]["H2_num"]["metrics"][metric]
        ci = (
            f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
            if r["ci_is_bounded"]
            else f"unbounded: the denominator's bootstrap interval includes 0 ({pct(r['frac_boot_den_nonpositive'])} of resamples ≤ 0); percentile values [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] are not interpretable"
        )
        rows.append(
            [
                f"{METRIC_BY_KEY[metric][4]} (n={r['n_problems']})",
                "1,2,3",
                f"{sgn(r['numerator'])} (tr {fmt_pcts(num['truncation_a_per_seed'])} vs {fmt_pcts(num['truncation_b_per_seed'])})",
                f"{sgn(r['denominator'])} (tr {fmt_pcts(den['truncation_a_per_seed'])} vs {fmt_pcts(den['truncation_b_per_seed'])})",
                f"{r['value']:+.2f}",
                " / ".join(f"{v:+.2f}" for v in r["per_seed"]),
                ci,
            ]
        )
    return md_table(
        ["metric", "seeds", "numerator Δ (truncation A vs B)", "denominator Δ (truncation A vs B)", "ratio", "ratio per seed", "joint problem-bootstrap 95 % CI"],
        rows,
    )  # fmt: skip


def wrong_list(ds: Dataset, results: dict[str, Any], key: str) -> list[str]:
    """Auto-generated 'how could this be wrong' bullets for one contrast (tasks/05 item 6)."""
    c = results["contrasts"][key]
    spec = c["spec"]
    la, lb = _label(results, spec["a"]), _label(results, spec["b"])
    t, o = c["metrics"]["test_greedy"], c["metrics"]["ood_greedy"]
    arm_a, arm_b = results["arms"][spec["a"]], results["arms"][spec["b"]]
    out = []
    # 1. truncation
    out.append(
        f"**Truncation gap.** test_300 greedy: {la} {fmt_pcts(t['truncation_a_per_seed'])} vs {lb} "
        f"{fmt_pcts(t['truncation_b_per_seed'])} (gap {100 * t['truncation_gap']:+.1f} pp; "
        f"{'/'.join(map(str, t['n_truncated_a_per_seed']))} vs {'/'.join(map(str, t['n_truncated_b_per_seed']))} "
        f"of {t['n_samples_per_run']} completions cut at the cap and scored wrong by rule), against an "
        f"accuracy Δ of {100 * t['mean_delta']:+.1f} pp. ood_hard_200 greedy: {fmt_pcts(o['truncation_a_per_seed'])} "
        f"vs {fmt_pcts(o['truncation_b_per_seed'])} (gap {100 * o['truncation_gap']:+.1f} pp; "
        f"{'/'.join(map(str, o['n_truncated_a_per_seed']))} vs {'/'.join(map(str, o['n_truncated_b_per_seed']))} of "
        f"{o['n_samples_per_run']}), against Δ {100 * o['mean_delta']:+.1f} pp. {_flag_text(results, spec, t)}. "
        "Truncated completions are scored wrong whatever they contain, so each arm's accuracy is a lower "
        "bound on what it would be without the cap, and the bound is looser for the arm with more truncation; how much looser is not estimated here (no correction has been approved)."
    )
    # 2. extraction failures beyond truncation
    out.append(
        f"**Extraction-failure gap.** test greedy {100 * t['extraction_failure_gap']:+.1f} pp, ood greedy "
        f"{100 * o['extraction_failure_gap']:+.1f} pp (A − B). Under v1.6 every truncated completion is "
        "an extraction failure, so this mostly restates the truncation gap; the part not explained by "
        f"truncation is {100 * (t['extraction_failure_gap'] - t['truncation_gap']):+.1f} pp on test and "
        f"{100 * (o['extraction_failure_gap'] - o['truncation_gap']):+.1f} pp on ood. Answer-line rate gap "
        f"(share of answers read from an explicit answer line rather than the last-integer fallback): "
        f"{100 * t['answer_line_gap']:+.1f} pp on test, {100 * o['answer_line_gap']:+.1f} pp on ood. Mean "
        f"completion length on test: {t['mean_tokens_a']:.0f} vs {t['mean_tokens_b']:.0f} tokens; on ood: "
        f"{o['mean_tokens_a']:.0f} vs {o['mean_tokens_b']:.0f}."
    )

    # 3. budgets
    def bud(k: str, arm_key: str, arm: dict[str, Any]) -> str:
        vals = [results["budgets"][f"{arm_key}/seed{s}"][k] for s in arm["seeds"]]
        return f"{np.mean(vals):,.0f}"

    sel_a, sel_b = arm_a.get("selection"), arm_b.get("selection")
    out.append(
        f"**Budget gap** (mean per seed, {la} vs {lb}). Prompts contributing examples: "
        f"{bud('prompts', spec['a'], arm_a)} vs {bud('prompts', spec['b'], arm_b)}; completions consumed: "
        f"{bud('completions_consumed', spec['a'], arm_a)} vs {bud('completions_consumed', spec['b'], arm_b)} "
        f"(of {bud('completions_available', spec['a'], arm_a)} vs {bud('completions_available', spec['b'], arm_b)} "
        f"available); training tokens: {bud('training_tokens', spec['a'], arm_a)} vs "
        f"{bud('training_tokens', spec['b'], arm_b)}; optimizer steps: {bud('optimizer_steps', spec['a'], arm_a)} vs "
        f"{bud('optimizer_steps', spec['b'], arm_b)}. Configurations tried on val: "
        f"{sel_a['n_configs_tried'] if sel_a else 1} vs {sel_b['n_configs_tried'] if sel_b else 1}"
        + (
            _iter_rft_config_note(results, arm_a, arm_b)
            if "iter_rft" in (arm_a["method"], arm_b["method"])
            else " — RFT's config is the best of 9 on a 100-problem val set (partly noise; 'best on val', not "
            "'optimal'), GRPO ran one fixed recipe; the asymmetry tilts toward RFT (PREREGISTRATION §5)."
            if bool(sel_a) != bool(sel_b)
            else (
                " — both arms' configs are each the best of 9 on a 100-problem val set, chosen separately, so the "
                "two arms differ in lr/epochs as well as in data: "
                f"{la} lr={sel_a['chosen']['learning_rate']:g}/ep={sel_a['chosen']['epochs']}, "
                f"{lb} lr={sel_b['chosen']['learning_rate']:g}/ep={sel_b['chosen']['epochs']}."
                if sel_a and sel_b
                else " — same fixed recipe on both sides."
            )
        )
    )

    # 4. tier composition
    def tiers_of(arm_key: str, arm: dict[str, Any]) -> str:
        b = results["budgets"][f"{arm_key}/seed{arm['seeds'][0]}"]
        split = arm["train_tier_counts"]
        txt = f"split {arm['data_condition']} {split}"
        if b["per_tier_examples"]:
            txt += f", prompts with ≥ 1 correct sample {b['per_tier_prompts']}, SFT examples {b['per_tier_examples']}"
        return txt

    out.append(
        f"**Tier-composition gap (training side).** {la}: {tiers_of(spec['a'], arm_a)}. {lb}: "
        f"{tiers_of(spec['b'], arm_b)}. RFT trains only on verified-correct completions, so its examples "
        "are weighted toward easy prompts even on the mixed and curated sets; GRPO samples every prompt "
        "equally. Evaluation side: both arms are scored on the identical problems (asserted), so there "
        "is no tier gap in test_300 (100/100/100) or ood_hard_200."
    )
    if "iter_rft" in (arm_a["method"], arm_b["method"]):
        out.append(SECONDARY_NOTE)
    # 5. seeds
    out.append(
        f"**Seed range overlap.** test greedy per seed: {la} {fmt_seeds(t['a_per_seed'])} vs {lb} "
        f"{fmt_seeds(t['b_per_seed'])} — ranges {'overlap' if t['seed_ranges_overlap'] else 'do not overlap'}; "
        f"ood greedy: {fmt_seeds(o['a_per_seed'])} vs {fmt_seeds(o['b_per_seed'])} — ranges "
        f"{'overlap' if o['seed_ranges_overlap'] else 'do not overlap'}. Three seeds per arm: the pooled seed std has "
        "4 degrees of freedom, so under normality a 95 % interval for the true seed std runs from about "
        "0.60× to 2.87× the estimate; the 2 × pooled-std threshold is a coarse yardstick. Paired problem-bootstrap CI "
        f"of Δ: test [{sgn(t['ci_low'])}, {sgn(t['ci_high'])}], ood [{sgn(o['ci_low'])}, {sgn(o['ci_high'])}]."
    )
    # 6. provenance
    dirty = sorted(set(t["git_dirty_units"]) | set(o["git_dirty_units"]))
    if dirty:
        out.append(f"**Provenance.** Units recorded with a dirty git tree: {', '.join(dirty)}.")
    return out


SECONDARY_NOTE = (
    "**Registered after unblinding; unequal seed variance.** This contrast was registered on 2026-09-20, "
    "after the test_300 and ood_hard_200 results of all six arms had been seen (PREREGISTRATION §6): it is "
    "secondary, not part of the confirmatory H1–H3 analysis. RFT-Curated's three seeds train on one shared "
    "base-model draw, so its seed std omits sampling variance; IterRFT's rounds 2–3 are sampled per seed "
    "from the current policy, so its seed std includes it (round 1 is the first 64 per prompt of the same "
    "shared draw for every seed). The pooled seed std in the criterion mixes the two."
)


def _iter_rft_config_note(
    results: dict[str, Any], arm_a: dict[str, Any], arm_b: dict[str, Any]
) -> str:
    """'Configurations tried' text for a contrast with the iterated-RFT arm on one side."""
    other = arm_b if arm_a["method"] == "iter_rft" else arm_a
    it = arm_a if arm_a["method"] == "iter_rft" else arm_b
    key = next(k for k, a in results["arms"].items() if a is it)
    b = results["budgets"][f"{key}/seed{it['seeds'][0]}"]
    txt = (
        f" — {it['label']} ran no sweep: it reuses lr={b['learning_rate']:g}/ep={b['epochs']}, the config "
        "RFT-Curated selected as best of 9 on val for single-round RFT, not tuned for this arm "
        "(PREREGISTRATION §4 deviation, §6); generation budget 3 rounds × 64 per prompt = 14,016 rollouts "
        "(4,672 from the shared base draw, 9,344 from the current policy)"
    )
    if other.get("selection"):
        return (
            txt
            + f"; {other['label']} uses the same val-selected config on its 14,016 draw samples."
        )
    return txt + f"; {other['label']} ran one fixed recipe on 19,200 on-policy rollouts (SPEC §9)."


def wrong_markdown(ds: Dataset, results: dict[str, Any], cfg_hash: str) -> str:
    out = [
        "# How could this be wrong? (tasks/05 item 6, auto-generated per headline contrast)",
        "",
        f"Analysis config hash `{cfg_hash[:12]}`. Every number below is from the final units of seeds "
        "1–3 (n = 300 test, 200 ood, greedy); CIs are paired problem bootstraps.",
        "",
    ]
    for spec in active_contrasts(ds):
        if not spec.headline:
            continue
        out += [f"## {spec.key}: {_label(results, spec.a)} − {_label(results, spec.b)}", ""]
        out += [f"{i}. {line}" for i, line in enumerate(wrong_list(ds, results, spec.key), 1)]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _h1_truncation_note(results: dict[str, Any]) -> str:
    g = results["contrasts"]["H1_grpo"]["metrics"]
    r = results["contrasts"]["H1_rft"]["metrics"]
    return (
        "Both contrasts are within one method, but truncation still differs inside GRPO: GRPO-Mixed "
        f"minus GRPO-Easy is {100 * g['test_greedy']['truncation_gap']:+.1f} pp on test_300 and "
        f"{100 * g['ood_greedy']['truncation_gap']:+.1f} pp on ood_hard_200, the split criterion (b) reads "
        f"its sign from (RFT: {100 * r['test_greedy']['truncation_gap']:+.1f} pp and "
        f"{100 * r['ood_greedy']['truncation_gap']:+.1f} pp)."
    )


VERDICT = "**Verdict (Laksh):** ______________________________________________"


def _contrast_section(results: dict[str, Any], key: str) -> list[str]:
    c = results["contrasts"][key]
    spec = c["spec"]
    out = [
        f"**{_label(results, spec['a'])} − {_label(results, spec['b'])}** (`{key}`)",
        "",
        md_table(CONTRAST_HEADER, _contrast_rows(results, key, PRIMARY)),
        "",
        f"- SPEC §10 criterion: {_crit_text(c['criterion'])}",
    ]
    if c["metrics"]["test_greedy"]:
        out.append(f"- {_flag_text(results, spec, c['metrics']['test_greedy'])}")
    out.append("")
    return out


def hypotheses_markdown(ds: Dataset, results: dict[str, Any], cfg_hash: str) -> str:
    # The confirmatory sections read the confirmatory arms only: a secondary arm (tasks/06b, registered
    # after unblinding) gets its own section at the end and must not change a word above it.
    secondary = {k for k, spec in ds.cfg["arms"].items() if spec.get("secondary")}
    arms = {k: a for k, a in results["arms"].items() if k not in secondary}
    flagged = {
        a["label"]: a["cells"]["test_greedy"]["flagged_seeds"]
        for a in arms.values()
        if a["cells"]["test_greedy"] and a["cells"]["test_greedy"]["flagged_seeds"]
    }
    out = [
        "# Hypothesis grading sheet (tasks/05 item 5)",
        "",
        "Generated by `rlordata.analysis.report`; regenerating overwrites this file, so grade a copy "
        "(or commit the graded file elsewhere). **The agent computes and evaluates the pre-registered "
        "criterion mechanically; every verdict line is blank for Laksh.**",
        "",
        f"Inputs: final units of seeds 1–3 under `{ds.cfg['run_root']}`; analysis config hash "
        f"`{cfg_hash[:12]}`; bootstrap seed {ds.cfg['seed']}, {ds.cfg['n_boot']:,} resamples. Δ = A − B "
        "on greedy accuracy, paired seed-wise; CI = paired per-problem bootstrap; pooled seed std uses "
        "sample variances (ddof = 1). Full tables: `tables/results.md`, `tables/contrasts.md`; "
        "per-contrast caveats: `how_could_this_be_wrong.md`; checks: `sanity.md`.",
        "",
        "## Criterion (SPEC §10, PREREGISTRATION §2)",
        "",
        "An arm-vs-arm difference counts if (a) |Δ greedy accuracy on test_300| > 2 × the pooled seed "
        "std of the two arms, **and** (b) the sign of Δ is the same on ood_hard_200.",
        "",
        "## Standing caveats — read before any line below",
        "",
        "1. **Truncation differs systematically between methods.** SPEC §7: a run with truncation > 5 % "
        "on test_300 is flagged and its numbers are not headline numbers. Flagged on test_300 greedy: "
        + (
            "; ".join(f"{k} seeds {','.join(map(str, v))}" for k, v in flagged.items())
            if flagged
            else "none"
        )
        + ". No RFT run is flagged. A truncated completion is scored wrong whatever it contains, so every "
        "accuracy is a lower bound on its uncapped value and the bound is looser on the GRPO side of "
        "every RFT-vs-GRPO Δ; on ood_hard_200 (GRPO greedy truncation up to "
        f"{pct(max(max(a['cells']['ood_greedy']['truncation_per_seed']) for a in arms.values() if a['method'] == 'grpo' and a['cells']['ood_greedy']))}) "
        "a large share of the ood result is 'ran out of tokens'. This lands directly on H3 and on the H2 "
        "denominator. **No correction is applied anywhere in this sheet**; whether one is warranted is a "
        "protocol decision for Laksh.",
        "2. **Tuning asymmetry (PREREGISTRATION §5).** RFT tried 9 configurations per arm and reports the "
        "best on a 100-problem val set ('best on val', not 'optimal'; partly noise); GRPO ran one fixed "
        "recipe. The asymmetry tilts toward RFT, so an H3 result in GRPO's favour is the more convincing "
        "direction. Test and ood were never used to choose anything (`sanity.md`, 'model selection').",
        "3. **Three seeds per arm.** The seed std in the criterion is estimated from 3 values per arm.",
        "4. Base and the controls C1/C2 have one run each, so criterion (a) cannot be evaluated for them.",
        "",
    ]

    out += [
        "## H1 — data effect",
        "",
        "> SPEC §1: Mixed-difficulty training data improves held-out performance relative to easy-only "
        "data *for both* RFT and GRPO. Test: (RFT-Mixed − RFT-Easy) and (GRPO-Mixed − GRPO-Easy), each vs. "
        "seed spread.",
        ">",
        "> Falsifier (PREREGISTRATION §1): an asymmetry — GRPO-Mixed > GRPO-Easy by more than seed spread, "
        "but RFT-Mixed ≈ RFT-Easy (or worse).",
        "",
        *_contrast_section(results, "H1_rft"),
        *_contrast_section(results, "H1_grpo"),
        _h1_truncation_note(results),
        "",
        VERDICT,
        "",
    ]

    r_t, r_o = results["h2_ratio"]["test_greedy"], results["h2_ratio"]["ood_greedy"]
    out += [
        "## H2 — implicit filtering",
        "",
        "> SPEC §1: RFT trained on prompts selected by the same current-policy criterion that makes GRPO "
        "groups informative (1–7 correct of 8) recovers a substantial fraction of GRPO's advantage. Test: "
        "(RFT-Curated − RFT-Mixed) relative to (GRPO-Curated − RFT-Mixed).",
        ">",
        "> Falsifier (PREREGISTRATION §1): RFT-Curated ≈ RFT-Mixed while GRPO-Curated stays well above "
        "both; also falsified if RFT-Curated is worse than RFT-Mixed.",
        "",
        "**Threshold.** tasks/05 asks for 'the pre-registered threshold' on this ratio. No numeric "
        "threshold exists in SPEC.md, PREREGISTRATION.md or any task file (SPEC says 'a substantial "
        "fraction', the pre-registration 'much of the gap'). None is invented here: the sheet reports "
        "the ratio, its interval and the two falsifier conditions evaluated mechanically.",
        "",
        "Threshold Laksh applies to the ratio: ______",
        "",
        _ratio_table(results),
        "",
        f"- Falsifier 'RFT-Curated worse than RFT-Mixed': the numerator's sign on test is "
        f"**{'negative' if r_t['numerator'] < 0 else 'positive'}** ({sgn(r_t['numerator'])}; on ood "
        f"{sgn(r_o['numerator'])}); its CI and the §10 evaluation are in the `H2_num` table below. Whether "
        "that amounts to 'worse' is part of the verdict.",
        "",
        *_contrast_section(results, "H2_num"),
        *_contrast_section(results, "H2_den"),
        "The denominator is an RFT-vs-GRPO contrast and all three GRPO-Curated seeds carry the §7 flag: "
        "GRPO-Curated's accuracy is the looser lower bound of the two, so the denominator — and with it "
        "the ratio — is sensitive to how truncated completions are treated. Not estimated here.",
        "",
        VERDICT,
        "",
    ]

    out += [
        "## H3 — on-policy / negative-feedback effect (**the headline comparison**)",
        "",
        "> SPEC §1: GRPO-Curated outperforms RFT-Curated by more than run-to-run variability despite "
        "identical prompt and sampling budgets.",
        ">",
        "> Falsifier (PREREGISTRATION §1): the difference on test_300 is within 2 × the pooled seed std, or "
        "it flips sign on ood_hard_200, or RFT-Curated ≥ GRPO-Curated.",
        "",
        *_contrast_section(results, "H3"),
        "Stated next to H3 as required: **RFT tried 9 configurations per arm (best on val, n = 100, seed "
        "1) and GRPO ran one fixed recipe** (PREREGISTRATION §5.2); the asymmetry tilts toward RFT. "
        "**Every GRPO-Curated seed exceeds 5 % truncation on test_300 greedy and no RFT-Curated seed "
        "does**, so under SPEC §7 this Δ is not a headline number as it stands; GRPO-Curated's accuracy "
        "is the looser lower bound of the two. Same prompts (train_curated, 73) and the same 19,200-completion generation budget on "
        "both sides; consumed completions, training tokens and optimizer steps differ by design (see "
        "`how_could_this_be_wrong.md`, H3).",
        "",
        VERDICT,
        "",
    ]

    h1 = results["contrasts"]["H1_rft"]
    h3 = results["contrasts"]["H3"]
    t1, o1 = h1["metrics"]["test_greedy"], h1["metrics"]["ood_greedy"]
    t3, o3 = h3["metrics"]["test_greedy"], h3["metrics"]["ood_greedy"]
    out += [
        "## Pre-registered predictions (PREREGISTRATION §3, written before training)",
        "",
        "### P1 — 'RFT-Mixed > RFT-Easy on test_300 (same direction on ood_hard_200), gap > seed noise'",
        "",
        f"- Δ test = {sgn(t1['mean_delta'])} (seeds 1–3, n = {t1['n_problems']}, CI [{sgn(t1['ci_low'])}, {sgn(t1['ci_high'])}], "
        f"truncation {fmt_pcts(t1['truncation_a_per_seed'])} vs {fmt_pcts(t1['truncation_b_per_seed'])}): sign "
        f"**{'as predicted' if t1['mean_delta'] > 0 else 'opposite to the prediction'}**.",
        f"- Δ ood = {sgn(o1['mean_delta'])} (n = {o1['n_problems']}, CI [{sgn(o1['ci_low'])}, {sgn(o1['ci_high'])}], "
        f"truncation {fmt_pcts(o1['truncation_a_per_seed'])} vs {fmt_pcts(o1['truncation_b_per_seed'])}).",
        f"- 'gap > seed noise' read as SPEC §10: {_crit_text(h1['criterion'])}",
        "",
        VERDICT,
        "",
        "### P2 — 'GRPO-Curated > RFT-Curated on test_300 with the same sign on ood, but the residual is "
        "modest; most of GRPO's edge vs RFT-Mixed is recovered by curation'",
        "",
        f"- Δ test = {sgn(t3['mean_delta'])} (seeds 1–3, n = {t3['n_problems']}, CI [{sgn(t3['ci_low'])}, {sgn(t3['ci_high'])}], "
        f"truncation {fmt_pcts(t3['truncation_a_per_seed'])} vs {fmt_pcts(t3['truncation_b_per_seed'])} {FLAG}): sign "
        f"**{'as predicted' if t3['mean_delta'] > 0 else 'opposite to the prediction'}**; Δ ood = {sgn(o3['mean_delta'])} "
        f"(n = {o3['n_problems']}, CI [{sgn(o3['ci_low'])}, {sgn(o3['ci_high'])}], truncation "
        f"{fmt_pcts(o3['truncation_a_per_seed'])} vs {fmt_pcts(o3['truncation_b_per_seed'])}).",
        f"- SPEC §10 on this contrast: {_crit_text(h3['criterion'])}",
        f"- 'most of GRPO's edge … recovered by curation' is the H2 ratio: {r_t['value']:+.2f} on test "
        f"(per seed {' / '.join(f'{v:+.2f}' for v in r_t['per_seed'])}), {r_o['value']:+.2f} on ood. 'Most' and "
        "'modest' have no pre-registered number; if 'most' is read as a ratio > 0.5 the condition is "
        f"**{'met' if r_t['value'] > 0.5 else 'not met'}** on test — that reading is the agent's, not the protocol's.",
        "",
        VERDICT,
        "",
    ]

    out += ["## Controls (SPEC §8): C1 random reward and C2 format-only vs Base", ""]
    for key in ("C1", "C2"):
        out += _contrast_section(results, key)
    out += [
        "One training seed per control and one base evaluation: Δ and its paired problem-bootstrap CI "
        "are reported, the seed-std criterion is not evaluable. For scale, the seed std of GRPO-Mixed "
        f"(same prompts, same recipe) on test greedy is {f3(arms['grpo_mixed']['cells']['test_greedy']['std'])}.",
        "",
        VERDICT,
        "",
    ]
    if "S1_iter_vs_rft" in results["contrasts"] and "S2_grpo_vs_iter" in results["contrasts"]:
        out += _secondary_section(results)
    return "\n".join(out).rstrip() + "\n"


def _secondary_section(results: dict[str, Any]) -> list[str]:
    """tasks/06b B: the iterated-RFT contrasts, after every confirmatory section, verdict blank."""
    c = results["contrasts"]
    s1, s2, h3 = (c[k]["metrics"] for k in ("S1_iter_vs_rft", "S2_grpo_vs_iter", "H3"))
    t1, t2, t3 = s1["test_greedy"], s2["test_greedy"], h3["test_greedy"]
    o1, o2 = s1["ood_greedy"], s2["ood_greedy"]
    crit2 = c["S2_grpo_vs_iter"]["criterion"]
    share = (
        f"{100 * t1['mean_delta'] / t3['mean_delta']:.0f} % / {100 * t2['mean_delta'] / t3['mean_delta']:.0f} %"
        if t3["mean_delta"]
        else "n/a"
    )

    def signs(block: dict[str, Any]) -> str:
        return " / ".join(
            "+" if d > 0 else ("−" if d < 0 else "0") for d in block["delta_per_seed"]
        )

    return [
        "## Secondary — iterated RFT (PREREGISTRATION §6; **registered after unblinding, not confirmatory**)",
        "",
        "> §6 (2026-09-20), written after the test_300 and ood_hard_200 results of all six arms, both controls "
        "and the reference models had been seen: (i) IterRFT − RFT-Curated: what three policy refreshes of the "
        "sampler add within a positives-only SFT objective. (ii) GRPO-Curated − IterRFT: what remains. "
        "H3 = (ii) + (i) exactly. SPEC §10 criterion unchanged. No threshold is set on any 'fraction of the "
        "GRPO advantage recovered'; if such a fraction is reported it is descriptive only.",
        ">",
        "> Predictions (Laksh, before running): (i) 'small positive — a few points, maybe not past seed noise'; "
        "(ii) 'still clearly positive; most of the H3 gap remains with GRPO'; same signs on ood_hard_200: 'yes "
        "on both contrasts'. What would change his mind about H3: IterRFT closes most of the gap to "
        "GRPO-Curated (Δ within ~2× pooled seed std of zero, or flips); also if IterRFT ≤ RFT-Curated on test.",
        "",
        "**Seed variance is not comparable between the arms.** RFT-Curated's three seeds train on one shared "
        "base-model draw, so its seed std omits sampling variance; IterRFT's rounds 2–3 are sampled per seed "
        "from the current policy, so its seed std includes it. The pooled seed std below mixes the two. "
        "**What (ii) does not isolate** (§6): besides negative samples and group-relative advantages, GRPO "
        "still differs in 300 policy refreshes vs 3, one pass per sample vs 4 epochs, lr 5e-5 vs 1e-5, 19,200 vs "
        "14,016 rollouts, and adaptive zero-advantage filtering. IterRFT ran no sweep: lr/epochs are reused "
        "from RFT-Curated (PREREGISTRATION §4 deviation). The SPEC §7 truncation flag applies as everywhere.",
        "",
        *_contrast_section(results, "S1_iter_vs_rft"),
        *_contrast_section(results, "S2_grpo_vs_iter"),
        f"- Decomposition on test_300 greedy (exact by construction): H3 Δ {sgn(t3['mean_delta'])} = (ii) "
        f"{sgn(t2['mean_delta'])} + (i) {sgn(t1['mean_delta'])}; shares of the H3 Δ (i) / (ii): {share} — "
        "descriptive only, no threshold is registered.",
        f"- Signs per seed (seeds {','.join(map(str, t1['seeds']))}): (i) test {signs(t1)}, ood {signs(o1)}; "
        f"(ii) test {signs(t2)}, ood {signs(o2)}. Mean Δ on ood: (i) {sgn(o1['mean_delta'])} "
        f"[{sgn(o1['ci_low'])}, {sgn(o1['ci_high'])}], (ii) {sgn(o2['mean_delta'])} [{sgn(o2['ci_low'])}, "
        f"{sgn(o2['ci_high'])}].",
        "- 'Change my mind' conditions, evaluated mechanically: (ii) within 2 × pooled seed std of zero on "
        f"test: **{'no' if crit2 and crit2['magnitude_ok'] else 'yes'}** (|Δ| {f3(abs(t2['mean_delta']))} vs "
        f"{f3(crit2['threshold']) if crit2 else MISSING}); (ii) flips sign: **{'yes' if t2['mean_delta'] <= 0 else 'no'}**; "
        f"IterRFT ≤ RFT-Curated on test: **{'yes' if t1['mean_delta'] <= 0 else 'no'}** (Δ {sgn(t1['mean_delta'])}).",
        "",
        VERDICT,
        "",
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def write_appendices(
    ds: Dataset, results: dict[str, Any], out: Path, cfg_hash: str, cache: Path | None
) -> None:
    """tasks/06b: truncation bounds and the exploratory larger-cap table, in their own directory."""
    from rlordata.analysis import appendix

    app = out / "appendix"
    app.mkdir(parents=True, exist_ok=True)
    rows = appendix.bounds_rows(ds)
    (app / "truncation_bounds.md").write_text(
        appendix.bounds_markdown(ds, results, rows, cfg_hash), encoding="utf-8"
    )
    with (app / "truncation_bounds.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["contrast", "a", "b", "split", "scenario", "n_problems", "seeds", "a_per_seed",
                    "b_per_seed", "delta_per_seed", "mean_delta", "ci_low", "ci_high",
                    "two_pooled_seed_std", "abs_delta_above_it"])  # fmt: skip
        for r in rows:
            w.writerow([r["contrast"], r["a"], r["b"], r["split"], r["scenario"], r["n_problems"],
                        "/".join(map(str, r["seeds"])), "/".join(map(repr, r["a_per_seed"])),
                        "/".join(map(repr, r["b_per_seed"])), "/".join(map(repr, r["delta_per_seed"])),
                        r["mean_delta"], r["ci_low"], r["ci_high"], r["two_pooled_seed_std"], r["exceeds"]])  # fmt: skip
    if not ds.cfg.get("exploratory"):
        return
    hashes = sanity.adapter_hashes(
        ds, cache_path=None if cache is None else cache / "adapter_sha256.json"
    )
    erows = appendix.exploratory_rows(ds, {k: v["sha256"] for k, v in hashes.items()})
    if erows is None:
        return
    (app / "exploratory_cap.md").write_text(
        appendix.exploratory_markdown(ds, erows, cfg_hash), encoding="utf-8"
    )
    with (app / "exploratory_cap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        cols = ["model", "who", "seed", "split", "n_problems", "locked_cap", "exploratory_cap",
                "primary_accuracy", "primary_truncation", "accuracy", "truncation", "mean_tokens",
                "n_cut_primary", "cut_now_right", "cut_still_cut", "n_uncut", "uncut_same_text",
                "uncut_right_to_wrong", "uncut_wrong_to_right", "same_host", "hosts", "config_hash",
                "adapter_checked"]  # fmt: skip
        w.writerow([*cols, "ci_low", "ci_high", "exploratory"])
        for r in erows:
            w.writerow([*(r[c] for c in cols), r["ci"][0], r["ci"][1], True])


def write_outputs(
    ds: Dataset, sanity_report: sanity.CrossRunReport, out: Path, *, cache: Path | None = None
) -> dict[str, Any]:
    cfg_hash = loader.analysis_config_hash(ds.cfg)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (out / "sanity.md").write_text(sanity_report.to_markdown(), encoding="utf-8")
    (out / "sanity.json").write_text(
        json.dumps(sanity_report.to_json(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    resolved = {
        k: v for k, v in ds.cfg.items() if k != "out_dir"
    }  # where this file is, not an input
    (out / "analysis_config.yaml").write_text(
        f"# resolved analysis config; hash {cfg_hash}\n" + yaml.safe_dump(resolved, sort_keys=True),
        encoding="utf-8",
    )
    results = build_results(ds)
    (tables / "results.md").write_text(results_markdown(ds, results, cfg_hash), encoding="utf-8")
    (tables / "contrasts.md").write_text(
        contrasts_markdown(ds, results, cfg_hash), encoding="utf-8"
    )
    write_csvs(ds, results, tables)
    (out / "how_could_this_be_wrong.md").write_text(
        wrong_markdown(ds, results, cfg_hash), encoding="utf-8"
    )
    (out / "hypotheses.md").write_text(hypotheses_markdown(ds, results, cfg_hash), encoding="utf-8")
    write_appendices(ds, results, out, cfg_hash, cache)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rlordata.analysis.report")
    ap.add_argument("--config", default=str(loader.DEFAULT_CONFIG))
    ap.add_argument(
        "--run-root",
        default=None,
        help="override run_root (store-style runs/ directory; read-only)",
    )
    ap.add_argument("--out", default=None, help="override out_dir")
    ap.add_argument("--jobs", type=int, default=4, help="processes for parsing samples.jsonl")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write <out>/.cache")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    cfg = loader.load_config(args.config, run_root=args.run_root, out_dir=args.out)
    out = Path(cfg["out_dir"])
    if out.resolve().is_relative_to(Path(cfg["run_root"]).resolve()):
        raise SystemExit(f"refusing to write into the run root: {out}")
    cache = None if args.no_cache else out / ".cache"
    ds = loader.load_dataset(
        cfg, cache_dir=None if cache is None else cache / "units", jobs=args.jobs
    )
    rep = sanity.cross_run_checks(ds, splits_dir=cfg["splits_dir"], cache_dir=cache)
    out.mkdir(parents=True, exist_ok=True)
    if rep.failures:
        (out / "sanity.md").write_text(rep.to_markdown(), encoding="utf-8")
        print(rep.to_markdown())
        print(
            f"[analysis] {len(rep.failures)} sanity failure(s): no table, contrast or figure was written."
        )
        return 1
    results = write_outputs(ds, rep, out, cache=cache)
    if not args.no_figures:
        from rlordata.analysis import plots

        plots.write_figures(ds, results, out / "figures")
    from rlordata.run_dir import git_dirty, git_sha, package_versions

    (out / "analysis_meta.json").write_text(
        json.dumps(
            {
                "analysis_config_hash": loader.analysis_config_hash(cfg),
                "git_sha": git_sha(),
                "git_dirty": git_dirty(),
                "package_versions": package_versions(),
                "gpu_type": "none (CPU-only analysis)",
                "wall_clock_s": round(time.monotonic() - t0, 1),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    n_notes = sum(len(c.notes) for c in rep.checks)
    print(
        f"[analysis] sanity passed ({len(rep.checks)} checks, {n_notes} notes) — wrote {out}/ in {time.monotonic() - t0:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
