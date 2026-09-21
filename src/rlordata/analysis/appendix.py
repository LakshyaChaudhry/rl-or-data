"""tasks/06b appendices. Neither is a result table and neither touches a SPEC §10 criterion line.

1. **Post-hoc truncation bounds** (approved by Laksh 2026-09-20). A completion cut at the cap is scored
   wrong whatever it contains, so an arm's greedy accuracy lies between the observed value (every
   truncated completion wrong) and observed + truncation rate (every truncated completion right). For
   each contrast A − B this gives an identification interval for Δ; the 2 × pooled-seed-std yardstick is
   recomputed inside each scenario from that scenario's per-seed accuracies. Nothing is estimated.
2. **Exploratory larger-cap re-evaluation** (PREREGISTRATION §4 deviation, 2026-09-20): greedy units
   re-generated at 2 × the locked cap after the primary results were seen. The results loader refuses
   these paths (`never_results`); this module is the only reader, and everything it prints is
   asterisked and kept out of `tables/` and `hypotheses.md`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.analysis import report, stats
from rlordata.analysis.loader import Dataset, Run, Unit
from rlordata.analysis.report import f3, fmt_pcts, fmt_seeds, md_table, pct, sgn

SPLITS = (("test_300", "test greedy"), ("ood_hard_200", "ood greedy"))
EXPLORATORY_NOTE = (
    "\\* re-generated at a cap of {cap:,} tokens to show what accuracy would have been without truncation; "
    "exploratory, run after the primary results were seen; the policies were trained with completions "
    "capped at {locked:,}. Not a SPEC §10 number: nothing here enters the pre-registered criterion, the "
    "primary tables or a headline."
)
# (label, A uses its upper bound, B uses its upper bound)
SCENARIOS = (
    ("observed: every truncated completion wrong (the primary tables)", False, False),
    ("lower bound of Δ: B's truncated completions all right, A's all wrong", False, True),
    ("upper bound of Δ: A's truncated completions all right, B's all wrong", True, False),
    ("both arms' truncated completions all right", True, True),
)


def _arm_units(ds: Dataset, key: str, split: str) -> list[Unit]:
    runs = [ds.base] if key == "base" else (ds.arms.get(key) or ds.controls[key])
    return [r.units[(split, "greedy")] for r in runs]


def _scores(u: Unit, upper: bool) -> np.ndarray:
    """[P] greedy per-problem score; with ``upper`` every truncated completion counts as right."""
    assert u.scores.ndim == 1 and u.scores.shape == u.truncated.shape  # [P]
    assert float((u.scores * u.truncated).max(initial=0.0)) == 0.0, (
        "a truncated completion is scored right"
    )
    return u.scores + u.truncated if upper else u.scores


def bounds_rows(ds: Dataset) -> list[dict[str, Any]]:
    """One row per (contrast, split, scenario) for every headline and secondary contrast."""
    n_boot, seed = int(ds.cfg["n_boot"]), int(ds.cfg["seed"])
    rows = []
    for spec in report.active_contrasts(ds):
        if not spec.headline:
            continue
        for split, split_label in SPLITS:
            ua, ub = _arm_units(ds, spec.a, split), _arm_units(ds, spec.b, split)
            seeds = [u.seed for u in ua]
            for label, up_a, up_b in SCENARIOS:
                c = stats.paired_contrast(
                    seeds,
                    [_scores(u, up_a) for u in ua],
                    [_scores(u, up_b) for u in ub],
                    n_boot=n_boot,
                    seed=seed,
                )
                thr = stats.CRITERION_MULTIPLIER * c.pooled_seed_std
                rows.append(
                    {
                        "contrast": spec.key,
                        "a": spec.a,
                        "b": spec.b,
                        "split": split,
                        "split_label": split_label,
                        "scenario": label,
                        "n_problems": c.n_problems,
                        "seeds": seeds,
                        "a_per_seed": c.a_per_seed,
                        "b_per_seed": c.b_per_seed,
                        "delta_per_seed": c.delta_per_seed,
                        "mean_delta": c.mean_delta,
                        "ci_low": c.ci_low,
                        "ci_high": c.ci_high,
                        "two_pooled_seed_std": thr,
                        "exceeds": bool(abs(c.mean_delta) > thr),
                        "truncation_a": [float(u.metrics["truncation_rate"]) for u in ua],
                        "truncation_b": [float(u.metrics["truncation_rate"]) for u in ub],
                    }
                )
    return rows


def bounds_markdown(
    ds: Dataset, results: dict[str, Any], rows: list[dict[str, Any]], cfg_hash: str
) -> str:
    out = [
        "# Appendix — post-hoc truncation bounds (tasks/06b B; approved by Laksh 2026-09-20)",
        "",
        f"Analysis config hash `{cfg_hash[:12]}`; seeds 1–3, greedy, cap 4,352; paired problem bootstrap, "
        f"seed {ds.cfg['seed']}, {int(ds.cfg['n_boot']):,} resamples. **Post hoc: defined after the primary results "
        "were seen. Not a correction, not an estimate, and not an input to any SPEC §10 criterion line or to "
        "`tables/results.md`.** A completion cut at the cap is scored wrong by rule (SPEC §5 v1.6), so each "
        "run's accuracy lies in [observed, observed + truncation rate]. The rows give Δ = A − B at the corners "
        "of that region; the 2 × pooled-seed-std yardstick is recomputed inside each scenario from that "
        "scenario's per-seed accuracies. 'All truncated completions right' is an extreme, not a likely value: "
        "in the primary units no truncated greedy completion contains an answer before the cut (notebook "
        "2026-09-20).",
        "",
    ]
    for key in dict.fromkeys(r["contrast"] for r in rows):
        mine = [r for r in rows if r["contrast"] == key]
        la, lb = (results["arms"].get(mine[0][k], {}).get("label", mine[0][k]) for k in ("a", "b"))
        out += [f"## {key}: {la} − {lb}", ""]
        table = []
        for r in mine:
            table.append(
                [
                    f"{r['split_label']} (n={r['n_problems']})",
                    r["scenario"],
                    fmt_seeds(r["a_per_seed"]),
                    fmt_seeds(r["b_per_seed"]),
                    fmt_seeds(r["delta_per_seed"], signed=True),
                    f"{sgn(r['mean_delta'])} [{sgn(r['ci_low'])}, {sgn(r['ci_high'])}]",
                    f3(r["two_pooled_seed_std"]),
                    "yes" if r["exceeds"] else "no",
                ]
            )
        out += [
            md_table(
                ["split", "scenario", "A per seed", "B per seed", "Δ per seed", "mean Δ [paired bootstrap 95 % CI]",
                 "2 × pooled seed std (this scenario)", "abs(Δ) above it"],
                table,
            ),
            "",
        ]  # fmt: skip
        for split, split_label in SPLITS:
            rs = {r["scenario"]: r for r in mine if r["split"] == split}
            lo, hi = rs[SCENARIOS[1][0]], rs[SCENARIOS[2][0]]
            same = (lo["mean_delta"] > 0) == (hi["mean_delta"] > 0) and lo["mean_delta"] != 0
            out.append(
                f"- {split_label}: truncation A {fmt_pcts(lo['truncation_a'])} vs B {fmt_pcts(lo['truncation_b'])}; "
                f"Δ lies in [{sgn(lo['mean_delta'])}, {sgn(hi['mean_delta'])}] — "
                + (
                    f"the sign of Δ is the same at both ends ({'positive' if lo['mean_delta'] > 0 else 'negative'})."
                    if same
                    else "the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**."
                )
            )
        out.append("")
    ood = [
        t
        for r in rows
        if r["split"] == "ood_hard_200"
        for t in (*r["truncation_a"], *r["truncation_b"])
    ]
    width = [
        hi["mean_delta"] - lo["mean_delta"]
        for key in dict.fromkeys(r["contrast"] for r in rows)
        for lo, hi in [
            tuple(
                next(
                    r
                    for r in rows
                    if r["contrast"] == key
                    and r["split"] == "ood_hard_200"
                    and r["scenario"] == SCENARIOS[i][0]
                )
                for i in (1, 2)
            )
        ]
    ]
    unidentified = sum(
        1
        for key in dict.fromkeys(r["contrast"] for r in rows)
        if any(
            r["mean_delta"] <= 0
            for r in rows
            if r["contrast"] == key
            and r["split"] == "ood_hard_200"
            and r["scenario"] == SCENARIOS[1][0]
        )
        and any(
            r["mean_delta"] >= 0
            for r in rows
            if r["contrast"] == key
            and r["split"] == "ood_hard_200"
            and r["scenario"] == SCENARIOS[2][0]
        )
    )
    out += [
        f"On ood_hard_200 truncation is {pct(min(ood))}–{pct(max(ood))} per run in these contrasts, the interval "
        f"for Δ is {100 * min(width):.0f}–{100 * max(width):.0f} points wide, and it contains 0 for {unidentified} of "
        f"{len(width)} contrasts: **the ood bounds are uninformative**, including for "
        "criterion (b), which reads a sign on that split. They are printed so that nobody has to take that on "
        "trust.",
        "",
    ]
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# exploratory larger-cap re-evaluation
# ---------------------------------------------------------------------------


def _read_exploratory_unit(unit_dir: Path, locked_cap: int, multiple: int) -> dict[str, Any]:
    """Read one exploratory unit, refusing anything that is not stamped as such."""
    with (unit_dir / "metrics.json").open(encoding="utf-8") as f:
        m = json.load(f)
    with (unit_dir / "config.yaml").open(encoding="utf-8") as f:
        c = yaml.safe_load(f)
    for d, name in ((m, "metrics.json"), (c, "config.yaml")):
        if d.get("exploratory") is not True or not d.get("cap_deviation"):
            raise ValueError(f"{unit_dir}/{name} is not stamped exploratory + cap_deviation")
        if int(d["max_completion_tokens"]) != multiple * locked_cap:
            raise ValueError(
                f"{unit_dir}/{name}: cap {d['max_completion_tokens']} != {multiple} × {locked_cap}"
            )
    per_problem: dict[str, tuple[float, float, str]] = {}
    with (unit_dir / "samples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            per_problem[r["problem_id"]] = (
                float(bool(r["correct"])),
                float(bool(r["truncated"])),
                hashlib.sha256(str(r["completion"]).encode("utf-8")).hexdigest(),
            )
    ids = tuple(sorted(per_problem))
    meta_path = unit_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "metrics": m,
        "config": c,
        "problem_ids": ids,
        "scores": np.asarray([per_problem[i][0] for i in ids]),  # [P]
        "truncated": np.asarray([per_problem[i][1] for i in ids]),  # [P]
        "completion_sha": tuple(per_problem[i][2] for i in ids),  # [P]
        "hostname": str(meta.get("hostname", "unknown")),
    }


def exploratory_rows(ds: Dataset, adapter_sha: dict[str, str]) -> list[dict[str, Any]] | None:
    """One row per (model, split); None when the analysis config names no exploratory tree."""
    spec = ds.cfg.get("exploratory")
    if not spec:
        return None
    root = ds.run_root / spec["dir"]
    if not root.is_dir():
        return None
    locked = int(next(iter(ds.base.units.values())).config["max_completion_tokens"])
    rows = []
    for model in spec["models"]:
        run: Run = (
            ds.base
            if model["arm"] == "base"
            else next(r for r in ds.arms[model["arm"]] if r.seed == int(model["seed"]))
        )
        who = "base" if model["arm"] == "base" else f"{model['arm']}/seed{model['seed']}"
        for split, split_label in SPLITS:
            u = run.units[(split, "greedy")]
            e = _read_exploratory_unit(
                root / model["name"] / split / "greedy", locked, int(spec["cap_multiple"])
            )
            if e["problem_ids"] != u.problem_ids:
                raise ValueError(
                    f"{model['name']} {split}: exploratory and primary units score different problems"
                )
            if int(e["metrics"]["seed"]) != u.seed:
                raise ValueError(
                    f"{model['name']} {split}: seed {e['metrics']['seed']} != primary {u.seed}"
                )
            cut = u.truncated > 0  # [P] cut at the locked cap in the primary unit
            same_text = np.asarray(
                [a == b for a, b in zip(u.completion_sha, e["completion_sha"], strict=True)]
            )  # [P]
            sha = e["config"].get("adapter_sha256")
            rows.append(
                {
                    "model": model["name"], "who": who, "arm": model["arm"], "label": run.label, "seed": run.seed,
                    "split": split, "split_label": split_label, "n_problems": int(u.metrics["n_problems"]),
                    "locked_cap": locked, "exploratory_cap": int(e["metrics"]["max_completion_tokens"]),
                    "primary_accuracy": float(u.metrics["accuracy"]),
                    "primary_ci": (float(u.metrics["ci_low"]), float(u.metrics["ci_high"])),
                    "primary_truncation": float(u.metrics["truncation_rate"]),
                    "accuracy": float(e["metrics"]["accuracy"]),
                    "ci": (float(e["metrics"]["ci_low"]), float(e["metrics"]["ci_high"])),
                    "truncation": float(e["metrics"]["truncation_rate"]),
                    "mean_tokens": float(e["metrics"]["mean_completion_tokens"]),
                    "n_cut_primary": int(cut.sum()),
                    "cut_now_right": int((e["scores"][cut] == 1).sum()),
                    "cut_still_cut": int((e["truncated"][cut] == 1).sum()),
                    "n_uncut": int((~cut).sum()),
                    "uncut_same_text": int(same_text[~cut].sum()),
                    "uncut_right_to_wrong": int(((u.scores == 1) & (e["scores"] == 0) & ~cut).sum()),
                    "uncut_wrong_to_right": int(((u.scores == 0) & (e["scores"] == 1) & ~cut).sum()),
                    "same_host": str(u.meta.get("hostname", "?")) == e["hostname"],
                    "hosts": f"{u.meta.get('hostname', '?')} → {e['hostname']}",
                    "config_hash": str(e["metrics"].get("config_hash", ""))[:12],
                    "adapter_checked": None if sha is None else adapter_sha.get(who) == sha,
                    "scores": e["scores"],
                }
            )  # fmt: skip
    return rows


def _reproducibility_note(rows: list[dict[str, Any]]) -> str:
    """Greedy re-generation of completions the primary run did NOT truncate: a free reproducibility check."""
    parts = []
    for same in (True, False):
        rs = [r for r in rows if r["same_host"] is same]
        if not rs:
            continue
        n = sum(r["n_uncut"] for r in rs)
        ident = sum(r["uncut_same_text"] for r in rs)
        flips = [
            (r["uncut_right_to_wrong"] + r["uncut_wrong_to_right"]) / r["n_problems"] for r in rs
        ]
        net = [
            abs(r["uncut_wrong_to_right"] - r["uncut_right_to_wrong"]) / r["n_problems"] for r in rs
        ]
        parts.append(
            f"{'same host' if same else 'different host'} as the primary run ({len(rs)} units: "
            f"{', '.join(sorted({r['label'] for r in rs}))}): {ident:,} of {n:,} untruncated completions are "
            f"byte-identical ({100 * ident / n:.1f} %); correctness flips on {100 * min(flips):.1f}–{100 * max(flips):.1f} % "
            f"of problems per unit, net accuracy change from those flips up to {100 * max(net):.1f} points"
        )
    return (
        "**Reproducibility of greedy decoding, measured here as a by-product.** A completion that the primary "
        "run did not truncate does not depend on the cap, so re-generating it should give the same text. "
        + "; ".join(parts)
        + ". Sampler settings, package versions, sampling code and GPU model are identical between the two runs "
        "of every model; what differs is the cap and, where stated, the host. **Consequence for reading this "
        "table:** where the host differs, the 'change' column mixes the effect of the larger cap with re-run "
        "differences of untruncated completions; only the 'cut → right' counts are attributable to the cap. "
        "**Consequence for the primary tables:** a single greedy evaluation carries machine-level re-run noise "
        "of this size that the seed std of an arm evaluated on one host does not contain; it is comparable to the "
        "smaller contrasts and small next to the larger ones."
    )


def exploratory_markdown(ds: Dataset, rows: list[dict[str, Any]], cfg_hash: str) -> str:
    cap, locked = rows[0]["exploratory_cap"], rows[0]["locked_cap"]
    note = EXPLORATORY_NOTE.format(cap=cap, locked=locked)
    checked = [
        r["adapter_checked"]
        for r in rows
        if r["adapter_checked"] is not None and r["split"] == "test_300"
    ]
    out = [
        f"# Appendix — EXPLORATORY re-evaluation at a cap of {cap:,} tokens\\* (tasks/06b C)",
        "",
        note,
        "",
        f"Analysis config hash `{cfg_hash[:12]}`. A deviation from SPEC §7 and §10, approved by Laksh and logged in "
        f"PREREGISTRATION §4 (2026-09-20), decided after the primary test results were seen. The cap was fixed by "
        f"rule before running ({cap:,} = {cap // locked} × the locked {locked:,}), never chosen from results. Greedy "
        "only; same prompts, extractor, seeds and vLLM settings as the primary units; the results loader refuses "
        "these paths. Adapter "
        f"identity (sha256 in the exploratory config = the adapter hashed by the cross-run sanity): "
        f"{sum(bool(x) for x in checked)} of {len(checked)} trained models match.",
        "",
        _reproducibility_note(rows),
        "",
        "## Per model\\*",
        "",
    ]
    table = []
    for r in rows:
        table.append(
            [
                r["label"] if r["arm"] == "base" else f"{r['label']} s{r['seed']}",
                f"{r['split_label']} (n={r['n_problems']})",
                f"{f3(r['primary_accuracy'])} [{f3(r['primary_ci'][0])}, {f3(r['primary_ci'][1])}] · tr {pct(r['primary_truncation'])}",
                f"{f3(r['accuracy'])}\\* [{f3(r['ci'][0])}, {f3(r['ci'][1])}] · tr {pct(r['truncation'])}",
                sgn(r["accuracy"] - r["primary_accuracy"]) + "\\*",
                f"{r['n_cut_primary']} → {r['cut_now_right']} right, {r['cut_still_cut']} still cut",
                f"{r['uncut_same_text']} of {r['n_uncut']}; {r['uncut_right_to_wrong']} / {r['uncut_wrong_to_right']}",
                "same" if r["same_host"] else "different",
                f"{r['mean_tokens']:.0f}",
                r["config_hash"],
            ]
        )
    out += [
        md_table(
            ["model (seed)", "split", f"primary, cap {locked:,} [95 % CI] · truncation",
             f"cap {cap:,}\\* [95 % CI] · truncation", "change\\*", f"cut at {locked:,} → at {cap:,}\\*",
             "not cut in primary: identical text; right→wrong / wrong→right\\*", "host of the two runs",
             "mean tokens\\*", "config hash"],
            table,
        ),
        "",
        "## Per arm and between arms\\* (seed i vs seed i; no SPEC §10 criterion is evaluated on these)",
        "",
    ]  # fmt: skip
    arms = list(dict.fromkeys(r["arm"] for r in rows if r["arm"] != "base"))
    n_boot, seed = int(ds.cfg["n_boot"]), int(ds.cfg["seed"])
    table = []
    for split, split_label in SPLITS:
        by = {
            a: sorted(
                (r for r in rows if r["arm"] == a and r["split"] == split), key=lambda r: r["seed"]
            )
            for a in arms
        }
        for a in arms:
            acc = [r["accuracy"] for r in by[a]]
            table.append([
                by[a][0]["label"], split_label, fmt_seeds(acc) + "\\*",
                f"{f3(float(np.mean(acc)))} ± {f3(float(np.std(acc, ddof=1)))}\\*",
                f"{f3(float(np.mean([r['primary_accuracy'] for r in by[a]])))}",
                fmt_pcts([r["truncation"] for r in by[a]]) + "\\*", "",
            ])  # fmt: skip
        for a, b in ((x.a, x.b) for x in report.active_contrasts(ds) if x.a in by and x.b in by):
            c = stats.paired_contrast(
                [r["seed"] for r in by[a]], [r["scores"] for r in by[a]], [r["scores"] for r in by[b]],
                n_boot=n_boot, seed=seed,
            )  # fmt: skip
            prim = float(
                np.mean([r["primary_accuracy"] for r in by[a]])
                - np.mean([r["primary_accuracy"] for r in by[b]])
            )
            table.append([
                f"{by[a][0]['label']} − {by[b][0]['label']}", split_label, fmt_seeds(c.delta_per_seed, signed=True) + "\\*",
                f"{sgn(c.mean_delta)} ± {f3(c.std_delta)}\\* [{sgn(c.ci_low)}, {sgn(c.ci_high)}]",
                sgn(prim), "", f"primary Δ at cap {locked:,} for comparison",
            ])  # fmt: skip
    out += [
        md_table(
            ["arm or contrast", "split", f"per seed at cap {cap:,}\\*", "mean ± seed std\\* [paired bootstrap 95 % CI for Δ]",
             f"primary mean at cap {locked:,}", "truncation per seed\\*", "note"],
            table,
        ),
        "",
        note,
        "",
    ]  # fmt: skip
    return "\n".join(out).rstrip() + "\n"
