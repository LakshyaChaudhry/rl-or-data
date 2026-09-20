"""Results packet: one ≤ 300-line markdown file of tables plus a flat CSV, from the tasks/05 outputs.

``python -m rlordata.analysis.packet`` (``make packet``) loads the same allow-listed run root as
``rlordata.analysis.report``, re-runs the cross-run sanity checks, and writes

- ``<out>/results_packet.md`` — sections 0–12, tables only, no verdicts, ``hypotheses.md`` verbatim;
- ``<out>/results_flat.csv`` — one row per (arm, seed, split, metric, value, ci_lo, ci_hi, n).

Nothing is trained, sampled or evaluated, the run root is only read, and nothing here reads a
test/ood number to choose anything: every row and column is fixed in advance.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.analysis import loader, report, sanity, stats
from rlordata.analysis.loader import Dataset, Run, Unit
from rlordata.analysis.report import FLAG, MISSING, TIERS
from rlordata.core.evaluate import bootstrap_ci
from rlordata.data.generator import read_jsonl

MAX_LINES = 300
DEFAULT_NOTES = Path("configs/analysis/packet_notes.yaml")
LOCKED_DIR = Path("configs/locked")
DYNAMICS_STEPS = (0, 50, 100, 150, 200, 250, 300)
WINDOW = 10  # training-log numbers are means over the 10 steps ending at the step (640 completions)
TIER_ARMS = ("base", "rft_mixed", "rft_curated", "grpo_mixed", "grpo_curated")
PAIRS = (("rft_easy", "grpo_easy"), ("rft_mixed", "grpo_mixed"), ("rft_curated", "grpo_curated"))
# (section-5 label, key in results["contrasts"])
CONTRAST_ROWS = (
    ("H1a RFT-Mixed − RFT-Easy", "H1_rft"),
    ("H1b GRPO-Mixed − GRPO-Easy", "H1_grpo"),
    ("H3 GRPO-Curated − RFT-Curated", "H3"),
    ("C1 − Base", "C1"),
    ("C2 − Base", "C2"),
)
# section-10 rows: the H2 row carries its numerator and denominator contrasts side by side
GAP_ROWS = (
    ("H1a RFT-Mixed − RFT-Easy", ("H1_rft",)),
    ("H1b GRPO-Mixed − GRPO-Easy", ("H1_grpo",)),
    (
        "H2 numerator RFT-Curated − RFT-Mixed ‖ denominator GRPO-Curated − RFT-Mixed",
        ("H2_num", "H2_den"),
    ),
    ("H3 GRPO-Curated − RFT-Curated", ("H3",)),
)
FLAT_HEADER = ["arm", "seed", "split", "metric", "value", "ci_lo", "ci_hi", "n"]


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def md_table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    """report.md_table with every cell made pipe-safe: a literal '|' would split the column."""
    assert all(len(r) == len(header) for r in rows), "ragged table"

    def safe(row: Sequence[Any]) -> list[str]:
        return [str(c).replace("|", "¦") for c in row]

    return report.md_table(safe(header), [safe(r) for r in rows])


def a3(x: float | None) -> str:
    return MISSING if x is None or math.isnan(x) else f"{x:.3f}"


def s3(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:+.3f}"


def p1(x: float | None) -> str:
    return MISSING if x is None or math.isnan(x) else f"{100 * x:.1f}"


def ci(lo: float | None, hi: float | None, signed: bool = False) -> str:
    f = s3 if signed else a3
    return f"[{f(lo)},{f(hi)}]"


def slash(values: Sequence[float | None], f: Any = a3) -> str:
    return " / ".join(f(v) for v in values)


def _who(run: Run) -> str:
    return f"{run.key}/seed{run.seed}" if run.kind in ("arm", "control") else run.key


def _groups(ds: Dataset) -> dict[str, list[Run]]:
    return {**ds.arms, **ds.controls}


# ---------------------------------------------------------------------------
# 0. provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def locked_hashes(locked_dir: Path = LOCKED_DIR) -> dict[str, str]:
    """sha256 of every file under configs/locked (read, never written)."""
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(locked_dir.glob("*.yaml"))
    }


def locked_mismatches(ds: Dataset, locked_dir: Path = LOCKED_DIR) -> list[str]:
    """Runs/units whose recorded protocol values or git-SHA copies of configs/locked differ from HEAD.

    Run dirs do not store a hash of the locked files, so two things are compared: (i) the values a
    resolved config copies out of them (cap, prompt template, GRPO block, LoRA shape); (ii) the git
    blobs of configs/locked/* at each run's recorded SHA against the blobs at HEAD.
    """
    out: list[str] = []
    cap = yaml.safe_load((locked_dir / "cap.yaml").read_text(encoding="utf-8"))
    prompt = yaml.safe_load((locked_dir / "prompt.yaml").read_text(encoding="utf-8"))
    training = yaml.safe_load((locked_dir / "training.yaml").read_text(encoding="utf-8"))
    n_units = 0
    for run in ds.all_runs():
        for u in run.units.values():
            n_units += 1
            if int(u.config.get("max_completion_tokens", -1)) != int(cap["max_completion_tokens"]):
                out.append(f"{u.rel}: cap {u.config.get('max_completion_tokens')}")
            if str(u.config.get("prompt_template", "")).strip() != str(prompt["template"]).strip():
                out.append(f"{u.rel}: prompt template")
    lora = training["lora"]
    for run in ds.trained_runs():
        path = run.run_dir / "config.yaml"
        if not path.exists():
            continue
        c = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if run.method == "grpo" and "grpo_locked" in c and c["grpo_locked"] != training["grpo"]:
            out.append(f"{run.rel}: grpo block")
        rl = c.get("lora") or (c.get("training") or {}).get("lora") or {}
        got = (
            rl.get("r"),
            rl.get("lora_alpha", rl.get("alpha")),
            sorted(rl.get("target_modules", [])),
        )
        if rl and got != (lora["r"], lora["alpha"], sorted(lora["target_modules"])):
            out.append(f"{run.rel}: LoRA shape")
    shas: dict[str, list[str]] = {}
    for run in ds.all_runs():
        for sha, who in [((run.meta or {}).get("git_sha"), run.rel)] + [
            (u.meta.get("git_sha"), u.rel) for u in run.units.values()
        ]:
            if sha:
                shas.setdefault(str(sha), []).append(who)
    n_shas = 0
    for sha, whos in sorted(shas.items()):
        differs = []
        for name in sorted(locked_hashes(locked_dir)):
            head = _git("rev-parse", f"HEAD:{locked_dir.as_posix()}/{name}")
            then = _git("rev-parse", f"{sha}:{locked_dir.as_posix()}/{name}")
            if head is None:
                continue
            if then != head:
                differs.append(name if then else f"{name} (absent)")
        n_shas += 1
        if differs:
            out.append(
                f"git {sha[:7]} ({len(whos)} run/unit records, e.g. {whos[0]}): {', '.join(differs)}"
            )
    out.insert(
        0, f"{n_units} units, {len(ds.trained_runs())} training runs, {n_shas} git SHAs compared"
    )
    return out


def _generation_path(ds: Dataset) -> list[str]:
    units = [u for r in ds.all_runs() for u in r.units.values()]
    samp = [u.config.get("sampler") or {} for u in units]

    def distinct(values: list[Any]) -> str:
        return ", ".join(sorted({str(v) for v in values}))

    dec: dict[str, set[str]] = {}
    for u in units:
        d = u.config.get("decoding") or {}
        dec.setdefault(u.decoding, set()).add(
            f"T={d.get('temperature')} top_p={d.get('top_p')} n={d.get('n')} rep={d.get('repetition_penalty')}"
        )
    return [
        f"sampler {distinct([s.get('sampler') for s in samp])}; vLLM {distinct([s.get('vllm_version') for s in samp])}; "
        f"dtype {distinct([s.get('dtype') for s in samp])}; batch_invariant {distinct([s.get('batch_invariant') for s in samp])}; "
        f"prefix caching {distinct([s.get('enable_prefix_caching') for s in samp])}; thinking {distinct([u.config.get('thinking') for u in units])}",
        "; ".join(f"{k}: {' | '.join(sorted(v))}" for k, v in sorted(dec.items()))
        + f"; seed scheme: {distinct([s.get('seed_scheme') for s in samp])}",
    ]


def section_provenance(ds: Dataset, date: str, meta: dict[str, Any]) -> list[str]:
    spec = Path("SPEC.md")
    m = re.search(r"LOCKED (v[\d.]+)", spec.read_text(encoding="utf-8")) if spec.exists() else None
    cap = yaml.safe_load((LOCKED_DIR / "cap.yaml").read_text(encoding="utf-8"))
    path = _generation_path(ds)
    mism = locked_mismatches(ds)
    sha = meta.get("git_sha") or _git("rev-parse", "HEAD") or "unknown"
    rows = [
        ["date · git commit · SPEC", f"{date}; tables and figures written by `make analysis` at git {sha} (dirty: {meta.get('git_dirty', 'unknown')}), the packet generator is committed together with this file; SPEC {m.group(1) if m else 'unknown'}; analysis config hash {loader.analysis_config_hash(ds.cfg)[:12]}; run root `{ds.cfg['run_root']}` (read-only); bootstrap seed {ds.cfg['seed']}, {int(ds.cfg['n_boot']):,} resamples"],
        ["configs/locked/*.yaml sha256[:12] · cap", "; ".join(f"{k} {v[:12]}" for k, v in locked_hashes().items()) + f"; cap: max_completion_tokens = {cap['max_completion_tokens']} (binding term: {cap.get('binding_term')}; anchor {cap.get('anchor_cap')})"],
        ["eval generation path · sampling params", f"{path[0]}; {path[1]}"],
        ["runs whose configs/locked differ", f"{mism[0]}. " + ("Differences: " + " · ".join(mism[1:]) if mism[1:] else "Differences: none")],
    ]  # fmt: skip
    return ["## 0. Provenance", md_table(["item", "value"], rows)]


# ---------------------------------------------------------------------------
# 1–2. sanity, manifest
# ---------------------------------------------------------------------------


def section_sanity(rep: sanity.CrossRunReport) -> list[str]:
    rows = [
        [i, c.name, "FAIL" if c.failures else "PASS", " · ".join(c.failures) or "—", len(c.notes), c.detail]
        for i, c in enumerate(rep.checks, 1)
    ]  # fmt: skip
    return [
        f"## 1. Sanity (analysis/sanity.py cross-run checks: {len(rep.checks)} checks, {len(rep.failures)} failures; notes in `sanity.md`)",
        md_table(["#", "check", "result", "failures", "notes", "what was checked"], rows),
    ]


def _steps(run: Run) -> list[dict[str, Any]]:
    return [r for r in run.train_log if not r.get("summary") and r.get("step") is not None]


def _planned_steps(run: Run) -> int | None:
    b = run.budgets or {}
    if run.method == "grpo":
        return b.get("max_steps")
    if b.get("completions_consumed") and b.get("batch_size") and b.get("epochs"):
        return math.ceil(b["completions_consumed"] / b["batch_size"]) * int(b["epochs"])
    return None


def restart_flags(run: Run, run_root: Path) -> list[str]:
    """Mechanical traces of a crash/restart/resume: attempt leftovers on disk and log discontinuities."""
    flags = []
    stale = sorted(p.name for p in run.run_dir.glob("*.stale-*"))
    if stale:
        flags.append(
            f"RESTARTED from step 0 (earlier attempt's logs kept as {len(stale)} *.stale-* files)"
        )
    failed = sorted(p.name for p in (run.run_dir.parent / "_failed").glob(f"{run.run_dir.name}_*"))
    if failed:
        flags.append(f"RE-RUN after failed attempt ({', '.join(failed)}; not read)")
    steps = [int(r["step"]) for r in _steps(run)]
    wall = [r.get("wall_clock_s") for r in _steps(run)]
    if len(steps) != len(set(steps)) or any(
        a is not None and b is not None and b < a for a, b in zip(wall, wall[1:], strict=False)
    ):
        flags.append("RESUMED (duplicate steps or wall-clock reset in train_log)")
    return flags


def _window(run: Run, step: int, field: str, tier: str | None = None) -> float | None:
    """Mean of a train-log field over the WINDOW steps ending at ``step`` (step 0: steps 1..WINDOW)."""
    lo, hi = (1, WINDOW) if step == 0 else (step - WINDOW + 1, step)
    rows = [r for r in _steps(run) if lo <= int(r["step"]) <= hi]
    if tier is None:
        vals = [(float(r[field]), 1.0) for r in rows if r.get(field) is not None]
    else:
        cells = [(r.get("per_tier") or {}).get(tier) for r in rows]
        vals = [
            (float(c[field]), float(c.get("n", 0))) for c in cells if c and c.get(field) is not None
        ]
    den = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / den if den > 0 else None


def section_manifest(ds: Dataset, hashes: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for run in ds.trained_runs():
        meta, b = run.meta or {}, run.budgets or {}
        if run.method == "grpo":
            last = [r for r in _steps(run) if r.get("reward_mean") is not None]
            what = f"final train reward ({b.get('reward', 'reward')}) {a3(_window(run, DYNAMICS_STEPS[-1], 'reward_mean'))} (mean of last {WINDOW} steps); last step {a3(last[-1]['reward_mean']) if last else MISSING}"
        else:
            sel = ds.selection.get(run.key, {})
            what = f"selected lr={b.get('learning_rate'):g}, epochs={b.get('epochs')} (best of {sel.get('n_configs_tried', '?')} on val_mixed_100, seed 1)"
        flags = restart_flags(run, ds.run_root)
        wall = meta.get("wall_clock_s")
        rows.append([
            run.rel, run.label, run.seed, meta.get("status", MISSING),
            f"{b.get('optimizer_steps', MISSING)} / {_planned_steps(run) or MISSING}",
            MISSING if wall is None else f"{float(wall) / 3600:.2f}",
            meta.get("gpu_type", MISSING),
            str(hashes.get(_who(run), {}).get("sha256", MISSING))[:8],
            (run.config_hash or MISSING)[:12], what, "; ".join(flags) or "none",
        ])  # fmt: skip
    return [
        "## 2. Run manifest (result-bearing runs; one row per arm × seed)",
        md_table(
            ["run", "arm", "seed", "status", "optimizer steps done / planned", "wall-clock h", "GPU",
             "adapter sha256[:8]", "config hash", "final train reward (GRPO) / selected sweep config (RFT)",
             "crashed / restarted / resumed"],
            rows,
        ),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# 3–6. accuracy tables
# ---------------------------------------------------------------------------


def _cell(c: dict[str, Any] | None, rates: bool = True) -> str:
    if c is None:
        return MISSING
    s = f"{a3(c['value'])} {ci(c['ci_low'], c['ci_high'])}"
    if rates:
        s += f" tr {p1(c['truncation_rate'])} xf {p1(c['extraction_failure_rate'])}"
    return s + (f" {FLAG}" if c.get("truncation_flag") else "")


def _arm_cell(c: dict[str, Any] | None) -> str:
    if c is None:
        return MISSING
    return f"{a3(c['mean'])} ± {a3(c['std'])} {ci(c['ci_low'], c['ci_high'])}"


def section_base_refs(ds: Dataset, results: dict[str, Any]) -> list[str]:
    keys = (
        "val_greedy",
        "test_greedy",
        "ood_greedy",
        "test_mean8",
        "ood_mean8",
        "test_pass8",
        "test_pass64",
    )
    rows = [
        [run.label, run.seed, *(_cell(results["runs"][_who(run)]["cells"][k]) for k in keys)]
        for run in (ds.base, *ds.references)
    ]
    return [
        f"## 3. Base + reference models (one evaluation each; cell = value [bootstrap 95 % CI] tr = truncation %, xf = extraction-fail %; {FLAG} = truncation > 5 % on val/test)",
        md_table(
            ["model", "seed", "val greedy (n=100)", "test greedy (n=300)", "ood greedy (n=200)",
             "test mean@8 (n=300)", "ood mean@8 (n=200)", "pass@8 (test first-100, 64 samples)",
             "pass@64 (test first-100, 64 samples)"],
            rows,
        ),
    ]  # fmt: skip


def section_main(ds: Dataset, results: dict[str, Any]) -> list[str]:
    acc = ("val_greedy", "test_greedy", "ood_greedy", "test_mean8")
    rows = []
    for key, runs in _groups(ds).items():
        for run in runs:
            cells = results["runs"][_who(run)]["cells"]
            rows.append([
                run.label, run.seed, *(_cell(cells[k], rates=False) for k in acc),
                slash([cells[k]["truncation_rate"] if cells[k] else None for k in acc], p1),
                slash([cells[k]["extraction_failure_rate"] if cells[k] else None for k in acc], p1),
                (run.config_hash or MISSING)[:12],
            ])  # fmt: skip
        if len(runs) > 1:
            arm = results["arms"][key]["cells"]
            flagged = arm["test_greedy"]["flagged_seeds"] if arm["test_greedy"] else []
            rows.append([
                f"**{runs[0].label}**", "**μ ± σ**",
                *(_arm_cell(arm[k]) + (f" {FLAG}×{len(arm[k]['flagged_seeds'])}" if arm[k] and arm[k]["flagged_seeds"] else "") for k in acc),
                slash([arm[k]["truncation_mean"] if arm[k] else None for k in acc], p1),
                slash([arm[k]["extraction_failure_mean"] if arm[k] else None for k in acc], p1),
                f"test trunc > 5 %: seeds {flagged or 'none'}",
            ])  # fmt: skip
    return [
        f"## 4. Main table (greedy accuracy [bootstrap 95 % CI over problems]; μ ± σ = mean ± sample std across seeds [CI of the seed-averaged score]; {FLAG} = truncation > 5 % on val/test, SPEC §7)",
        md_table(
            ["arm", "seed", "val greedy (n=100)", "test greedy (n=300)", "ood greedy (n=200)", "test mean@8 (n=300)",
             "trunc % (val / test / ood greedy / test mean@8)", "extraction-fail % (same order)", "config hash"],
            rows,
        ),
    ]  # fmt: skip


def _crit(c: dict[str, Any] | None) -> tuple[str, str, str, str]:
    """(2 × pooled seed std, (a), (b), PASS/FAIL) of SPEC §10, as evaluated by stats.evaluate_criterion."""
    if c is None:
        return ("n/a",) * 4  # type: ignore[return-value]
    b = "yes" if c["same_sign_on_ood"] else "no"
    if c["met"] is None:
        return "n/a (1 seed)", "n/a", b, "n/a (1 seed)"
    return (
        a3(c["threshold"]),
        "yes" if c["magnitude_ok"] else "no",
        b,
        "PASS" if c["met"] else "FAIL",
    )


def section_contrasts(results: dict[str, Any], notes: dict[str, Any]) -> list[str]:
    rows = []
    for label, key in CONTRAST_ROWS:
        block = results["contrasts"][key]
        cells = []
        for metric in ("test_greedy", "ood_greedy"):
            m = block["metrics"][metric]
            cells += [
                slash(m["delta_per_seed"], s3),
                f"{s3(m['mean_delta'])} ± {a3(m['std_delta']) if len(m['seeds']) > 1 else 'n/a'} {ci(m['ci_low'], m['ci_high'], signed=True)}",
            ]
        rows.append([label, *cells, *_crit(block["criterion"])])
        if key == "H1_grpo":
            thr = notes.get("h2_threshold")
            ratio = []
            for metric in ("test_greedy", "ood_greedy"):
                r = results["h2_ratio"][metric]
                interval = (
                    ci(r["ci_low"], r["ci_high"], signed=True)
                    if r["ci_is_bounded"]
                    else (
                        f"unbounded ({p1(r['frac_boot_den_nonpositive'])} % of resamples have denominator ≤ 0)"
                    )
                )
                ratio += [
                    slash(r["per_seed"], s3),
                    f"{s3(r['numerator'])} / {s3(r['denominator'])} = {s3(r['value'])} (on the seed means) {interval}",
                ]
            test = results["h2_ratio"]["test_greedy"]["value"]
            source = " ".join(str(notes.get("h2_threshold_source", "")).split())
            verdict = (
                f"n/a — no threshold registered ({source})"
                if thr is None
                else ("PASS" if test >= float(thr) else "FAIL")
            )
            rows.append([
                "H2 recovery fraction (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed)", *ratio,
                "none registered" if thr is None else f"threshold {thr}", "—", "—", verdict,
            ])  # fmt: skip
    return [
        "## 5. Paired contrasts, seed-wise (Δ = A − B, greedy accuracy; seeds 1 / 2 / 3; [paired problem-bootstrap 95 % CI]; SPEC §10: (a) |Δ test| > 2 × pooled seed std and (b) same sign on ood; H2 numerator and denominator contrasts: section 11)",
        md_table(
            ["contrast", "test Δ per seed (n=300)", "test mean Δ ± seed std of Δ [CI]", "ood Δ per seed (n=200)",
             "ood mean Δ ± seed std of Δ [CI]", "2 × pooled seed std (test)", "(a)", "(b)", "SPEC §10 criterion"],
            rows,
        ),
    ]  # fmt: skip


def _step_counts(ds: Dataset) -> dict[str, int]:
    path = Path(ds.cfg["splits_dir"]) / "test_300.jsonl"
    return {p.problem_id: int(p.total_steps) for p in read_jsonl(path)}


def _masked(units: list[Unit], mask: np.ndarray, seeds: list[int], ds: Dataset) -> str:
    """μ ± σ across seeds [CI] tr% of greedy accuracy on the masked problems of test_300."""
    tr = float(np.mean([u.truncated[mask].mean() for u in units]))
    if (
        len(units) == 1
    ):  # same bootstrap seed as report.tier_cells, so tier cells equal tables/per_tier.csv
        u = units[0]
        lo, hi = bootstrap_ci(u.scores[mask], n_boot=int(ds.cfg["n_boot"]), seed=u.seed)
        return f"{a3(float(u.scores[mask].mean()))} {ci(lo, hi)} tr {p1(tr)}"
    s = stats.summarize_arm(
        seeds,
        [u.scores[mask] for u in units],
        n_boot=int(ds.cfg["n_boot"]),
        seed=int(ds.cfg["seed"]),
    )
    return f"{a3(s.mean)} ± {a3(s.std)} {ci(s.ci_low, s.ci_high)} tr {p1(tr)}"


def section_tiers(ds: Dataset) -> list[str]:
    steps_of = _step_counts(ds)
    runs_of = {"base": [ds.base], **ds.arms}
    header: list[str] | None = None
    rows = []
    for key in TIER_ARMS:
        runs = runs_of[key]
        units = [r.units[("test_300", "greedy")] for r in runs]
        assert len({u.problem_ids for u in units}) == 1, (
            f"{key}: seeds scored on different problems"
        )
        u0 = units[0]
        assert u0.scores.ndim == 1 and u0.scores.shape == u0.truncated.shape  # [P]
        tiers = np.asarray(u0.tiers)
        n_steps = np.asarray([steps_of.get(pid, -1) for pid in u0.problem_ids])
        masks = [(f"tier {t}", tiers == t) for t in TIERS if (tiers == t).any()]
        masks += [(f"{k} steps", n_steps == k) for k in sorted(set(n_steps.tolist()))]
        if header is None:
            header = ["arm", "seeds", *(f"{name} (n={int(m.sum())})" for name, m in masks)]
        seeds = [r.seed for r in runs]
        rows.append(
            [
                runs[0].label,
                ",".join(map(str, seeds)),
                *(_masked(units, m, seeds, ds) for _, m in masks),
            ]
        )
    return [
        "## 6. Per-tier (base pass@8 tier) and per-step-count (`total_steps` of the problem) greedy accuracy on test_300 (μ ± σ across seeds [CI of the seed-averaged score] tr = truncation %)",
        md_table(header or ["arm"], rows),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# 7–9. budgets, sweep, dynamics
# ---------------------------------------------------------------------------


def _owned_draw(ds: Dataset, data_condition: str, parent: str) -> int | None:
    """Completions of the shared base draw that belong to ``data_condition``'s prompts."""
    draw = ds.run_root.parent / "data" / "samples" / f"base_{parent}_k192_seed1.jsonl"
    split = Path(ds.cfg["splits_dir"]) / f"{data_condition}.jsonl"
    if not draw.exists() or not split.exists():
        return None
    ids = {p.problem_id for p in read_jsonl(split)}
    with draw.open(encoding="utf-8") as f:
        return sum(1 for line in f if json.loads(line).get("problem_id") in ids)


def section_budgets(
    ds: Dataset, results: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, float]]]:
    billed = float(ds.cfg["gpu_rate"]["billed_usd_per_hour"])
    rows, per_arm = [], {}
    total = 0.0
    for key, runs in _groups(ds).items():
        bs = [results["budgets"][_who(r)] for r in runs]
        if runs[0].method == "rft":
            parent = (
                "train_easy_100"
                if runs[0].data_condition == "train_easy_100"
                else "train_mixed_100"
            )
            gen = _owned_draw(ds, str(runs[0].data_condition), parent)
            gen_txt = (
                f"{gen:,} (base draw, sampled once, shared by all seeds)"
                if gen is not None
                else MISSING
            )
            passes = [int(b["completions_consumed"]) * int(b["epochs"]) for b in bs]
            cons = f"{bs[0]['completions_consumed']:,} distinct × {bs[0]['epochs']} epochs = {passes[0]:,}"
        else:
            gen = int(np.mean([b["completions_consumed"] for b in bs]))
            gen_txt = f"{gen:,} (on-policy, per seed)"
            cons = f"{gen:,} × 1 = {gen:,}"
        sel = ds.selection.get(key, {})
        sweep_h = float(sel.get("other_configs_train_gpu_hours", 0.0)) + float(
            sel.get("other_configs_val_eval_gpu_hours", 0.0)
        )
        arm_h = sum(b["gpu_hours_train"] + b["gpu_hours_eval_final"] for b in bs)
        total += arm_h + sweep_h
        per_arm[key] = {
            "prompt_set": float(sum((results["arms"][key]["train_tier_counts"] or {}).values())),
            "prompts": float(np.mean([b["prompts"] for b in bs])),
            "generated": float(gen) if gen is not None else float("nan"),
            "tokens": float(np.mean([b["training_tokens"] for b in bs])),
            "steps": float(np.mean([b["optimizer_steps"] for b in bs])),
        }
        rows.append([
            runs[0].label, ",".join(str(r.seed) for r in runs), f"{per_arm[key]['prompts']:g} of {per_arm[key]['prompt_set']:g}", gen_txt, cons,
            slash([b["training_tokens"] for b in bs], lambda v: f"{v:,}"),
            slash([b["optimizer_steps"] for b in bs], lambda v: f"{v:,}"),
            slash([b["gpu_hours_train"] for b in bs], lambda v: f"{v:.2f}"),
            f"{arm_h:.2f}", f"{sweep_h:.2f}" if sweep_h else "—", f"${(arm_h + sweep_h) * billed:,.0f}",
        ])  # fmt: skip
    draw_meta = ds.run_root / "rft" / "draw_seed1" / "meta.json"
    draw_h = (
        float(json.loads(draw_meta.read_text(encoding="utf-8")).get("gpu_hours_actual", 0.0))
        if draw_meta.exists()
        else 0.0
    )
    out = [
        f"## 7. Budget accounting, actual (per seed unless stated; GPU = 1× H100 PCIe; $ at the ${billed:g}/h billed rate; total {total + draw_h:.2f} GPU-h = ${(total + draw_h) * billed:,.0f} incl. the {draw_h:.2f} GPU-h base draw, excl. GRPO checkpoint val evals)",
        md_table(
            ["arm", "seeds", "prompts used of prompt set (RFT: prompts with ≥ 1 verified-correct sample; GRPO: all prompts are sampled)", "completions generated",
             "completions consumed in gradient updates (per seed)", "gradient tokens (per seed)", "optimizer steps (per seed)",
             "train GPU-h (per seed)", "GPU-h all seeds (train + final eval)", "GPU-h other sweep configs (train + val eval)", "estimated $ (arm)"],
            rows,
        ),
        _budget_match_line(per_arm),
    ]  # fmt: skip
    return out, per_arm


def _budget_match_line(per_arm: dict[str, dict[str, float]]) -> str:
    """One line: the three SPEC §8 budgets per RFT/GRPO pair, and the largest RFT-vs-GRPO ratio."""
    fields = (
        ("prompt_set", "(1) prompt budget, prompts in the pair's prompt set"),
        ("prompts", "prompts that contribute training examples"),
        ("generated", "(2) generation budget, completions generated (SPEC §8.2 gives train_curated the draw samples of its own prompts)"),
        ("tokens", "(3) gradient/token budget, which SPEC §8.3 states cannot be equalized: gradient tokens"),
        ("steps", "optimizer steps"),
    )  # fmt: skip
    parts, worst = [], (0.0, "")
    for field, name in fields:
        cells, unequal = [], []
        for rft, grpo in PAIRS:
            a, b = per_arm[rft][field], per_arm[grpo][field]
            pair = rft.split("_", 1)[1]
            cells.append(f"{pair} {a:,.0f} vs {b:,.0f}")
            if a != b:
                unequal.append(pair)
            ratio = max(a, b) / min(a, b) if min(a, b) > 0 else float("nan")
            if not math.isnan(ratio) and ratio > worst[0]:
                worst = (
                    ratio,
                    f"{name.split(': ')[-1]}, {rft} {a:,.0f} vs {grpo} {b:,.0f} ({ratio:.1f}×)",
                )
        verdict = "equal in every pair" if not unequal else f"NOT equal for {', '.join(unequal)}"
        parts.append(f"{name}: {verdict} ({'; '.join(cells)})")
    return (
        "Matched budgets, RFT vs GRPO per pair (SPEC §8): "
        + ". ".join(parts)
        + f". Largest discrepancy: {worst[1]}."
    )


def section_sweep(ds: Dataset) -> list[str]:
    rows, header = [], None
    for key, sel in ds.selection.items():
        sweep = sel.get("sweep") or {}
        res = sorted(
            sweep.get("results", []), key=lambda r: (float(r["learning_rate"]), int(r["epochs"]))
        )
        lrs = sorted({float(r["learning_rate"]) for r in res})
        eps = sorted({int(r["epochs"]) for r in res})
        if header is None:
            header = [
                "arm (seed 1, val_mixed_100 greedy, n=100)",
                *(f"lr={r['learning_rate']:g} ep={r['epochs']}" for r in res),
                "selected",
                "selected config on grid edge?",
            ]
        ch = sel["chosen"]
        lr, ep = float(ch["learning_rate"]), int(ch["epochs"])
        edges = [
            n for n, v, grid in (("lr", lr, lrs), ("epochs", ep, eps)) if v in (grid[0], grid[-1])
        ]
        where = [
            f"{'lowest' if v == g[0] else 'highest'} {n}"
            for n, v, g in (("lr", lr, lrs), ("epochs", ep, eps))
            if n in edges
        ]
        rows.append([
            ds.arms[key][0].label,
            *(f"{a3(r['val_accuracy'])} {ci(r.get('ci_low'), r.get('ci_high'))} tr {p1(r.get('truncation_rate'))}"
              + (" ◀" if (float(r["learning_rate"]), int(r["epochs"])) == (lr, ep) else "") for r in res),
            f"lr={lr:g} ep={ep} (val {a3(ch['val_accuracy'])}; tie-break: {ch.get('tie_break', 'n/a')})",
            f"yes ({', '.join(where)})" if edges else "no (interior)",
        ])  # fmt: skip
    return ["## 8. RFT sweep: 9 configs on seed 1, selected on val_mixed_100 only (◀ = selected; cell = val accuracy [95 % CI] tr = truncation %)", md_table(header or ["arm"], rows)]  # fmt: skip


def section_dynamics(ds: Dataset) -> list[str]:
    base_val = ds.base.units.get(("val_mixed_100", "greedy"))
    rows = []
    for runs in (v for v in ds.arms.values() if v and v[0].method == "grpo"):
        s1 = runs[0]

        def both(f: Any, runs: list[Run] = runs, s1: Run = s1) -> list[str]:
            out = []
            for step in DYNAMICS_STEPS:
                vals = [f(r, step) for r in runs]
                mu = [v for v in vals if v is not None]
                out.append(f"{f(s1, step, True)} (μ {np.mean(mu):.3f})" if mu else "—")
            return out

        def num(field: str, scale: float = 1.0, digits: int = 3) -> Any:
            def g(run: Run, step: int, text: bool = False) -> Any:
                v = _window(run, step, field)
                v = None if v is None else v * scale
                return (MISSING if v is None else f"{v:.{digits}f}") if text else v

            return g

        def val(run: Run, step: int, text: bool = False) -> Any:
            c = (run.curves.get("val_greedy") or {}).get(str(step))
            if step == 0 and base_val is not None:
                c = {"accuracy": base_val.metrics["accuracy"], "truncation_rate": base_val.metrics["truncation_rate"],
                     "ci": [base_val.metrics["ci_low"], base_val.metrics["ci_high"]]}  # fmt: skip
            if c is None:
                return "—" if text else None
            return (
                f"{a3(c['accuracy'])} {ci(*c['ci'])} tr {p1(c['truncation_rate'])}"
                if text
                else float(c["accuracy"])
            )

        tiers = [t for t in TIERS if any((r.get("per_tier") or {}).get(t) for r in _steps(s1))]
        label = runs[0].label
        rows.append([label, "train reward", *both(num("reward_mean"))])

        def zero_std(run: Run, step: int, text: bool = False, tiers: list[str] = tiers) -> Any:
            """'all; easy/medium/hard' as text; the overall value as the number averaged across seeds."""
            if not text:
                return _window(run, step, "frac_reward_zero_std")
            per = [_window(run, step, "frac_reward_zero_std", tier=t) for t in tiers]
            return (
                f"{a3(_window(run, step, 'frac_reward_zero_std'))}; {'/'.join(a3(v) for v in per)}"
            )

        def zero_std_cells(
            runs: list[Run] = runs, s1: Run = s1, tiers: list[str] = tiers
        ) -> list[str]:
            out = []
            for step in DYNAMICS_STEPS:
                mu = []
                for t in (None, *tiers):
                    vals = [
                        v
                        for r in runs
                        if (v := _window(r, step, "frac_reward_zero_std", tier=t)) is not None
                    ]
                    mu.append(a3(float(np.mean(vals))) if vals else MISSING)
                out.append(f"{zero_std(s1, step, True)} (μ {mu[0]}; {'/'.join(mu[1:])})")
            return out

        def length_cells(runs: list[Run] = runs, s1: Run = s1) -> list[str]:
            out = []
            for step in DYNAMICS_STEPS:
                cell = []
                for group in ([s1], runs):
                    ln = [
                        v
                        for r in group
                        if (v := _window(r, step, "completions_mean_length")) is not None
                    ]
                    tr = [
                        v
                        for r in group
                        if (v := _window(r, step, "completions_clipped_ratio")) is not None
                    ]
                    cell.append(
                        f"{np.mean(ln):.0f} · {100 * np.mean(tr):.1f}" if ln and tr else MISSING
                    )
                out.append(f"{cell[0]} (μ {cell[1]})")
            return out

        rows.append([label, f"frac_reward_zero_std: all; {'/'.join(tiers)}", *zero_std_cells()])
        rows.append(
            [
                label,
                "mean completion length (tokens) · truncation % of training completions",
                *length_cells(),
            ]
        )
        rows.append(
            [
                label,
                "val_mixed_100 greedy (n=100)",
                *[c if c != "—" else "— (no checkpoint eval)" for c in both(val)],
            ]
        )
    return [
        f"## 9. GRPO training dynamics (cell = seed 1 (μ across seeds 1–3); train-log rows = mean over the {WINDOW} steps ending at the step, 640 completions per seed; column 0 = steps 1–{WINDOW}; tier rows weight by completions; val at step 0 = base model, one evaluation)",
        md_table(["arm", "quantity", *(str(s) for s in DYNAMICS_STEPS)], rows),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# 10–12. gaps, hypotheses, anomalies
# ---------------------------------------------------------------------------


def section_gaps(
    ds: Dataset, results: dict[str, Any], per_arm: dict[str, dict[str, float]]
) -> list[str]:
    rows = []
    for label, keys in GAP_ROWS:
        cols: list[list[str]] = [[] for _ in range(6)]
        for key in keys:
            block = results["contrasts"][key]
            a, b = block["spec"]["a"], block["spec"]["b"]
            t, o = block["metrics"]["test_greedy"], block["metrics"]["ood_greedy"]
            tiers = []
            for k in (a, b):
                arm = results["arms"][k]
                counts = arm["train_tier_counts"] or {}
                ex = results["budgets"][f"{k}/seed{arm['seeds'][0]}"].get("per_tier_examples")
                txt = "/".join(str(counts.get(x, 0)) for x in TIERS)
                tiers.append(
                    txt
                    + (
                        f" (SFT examples {'/'.join(str((ex or {}).get(x, 0)) for x in TIERS)})"
                        if ex
                        else ""
                    )
                )
            tried = [(ds.selection.get(k) or {}).get("n_configs_tried", 1) for k in (a, b)]
            cell = [
                f"{100 * t['truncation_gap']:+.1f} / {100 * o['truncation_gap']:+.1f}",
                f"{100 * t['extraction_failure_gap']:+.1f} / {100 * o['extraction_failure_gap']:+.1f}",
                f"generated {per_arm[a]['generated']:,.0f} vs {per_arm[b]['generated']:,.0f}; gradient tokens {per_arm[a]['tokens']:,.0f} vs {per_arm[b]['tokens']:,.0f}; "
                f"optimizer steps {per_arm[a]['steps']:,.0f} vs {per_arm[b]['steps']:,.0f}; configs tried {tried[0]} vs {tried[1]}",
                f"{tiers[0]} vs {tiers[1]}",
                f"{'yes' if t['seed_ranges_overlap'] else 'no'} / {'yes' if o['seed_ranges_overlap'] else 'no'}",
                f"A: {','.join(map(str, t['flagged_seeds_a'])) or 'none'}; B: {','.join(map(str, t['flagged_seeds_b'])) or 'none'}",
            ]  # fmt: skip
            for col, c in zip(cols, cell, strict=True):
                col.append(c)
        rows.append([label, *(" ‖ ".join(col) for col in cols)])
    return [
        "## 10. \"How could this be wrong\" gaps per contrast (A − B; test / ood greedy; full text in `how_could_this_be_wrong.md`)",
        md_table(
            ["contrast", "truncation gap (pp, test / ood)", "extraction-fail gap (pp, test / ood)", "budget gap (A vs B, mean per seed)",
             "tier composition of training prompts easy/medium/hard (A vs B)", "seed ranges overlap (test / ood)",
             "seeds with test truncation > 5 %"],
            rows,
        ),
    ]  # fmt: skip


def mechanical_anomalies(ds: Dataset, results: dict[str, Any]) -> list[list[str]]:
    """Two rows computed from the run root: training-log health, and eval-side flags."""
    bad = [
        f"{run.rel} step {r.get('step')} {f}"
        for run in ds.trained_runs()
        for r in run.train_log
        for f in ("loss", "grad_norm", "reward_mean")
        if isinstance(r.get(f), float) and not math.isfinite(r[f])
    ]
    grpo = [r for r in ds.trained_runs() if r.method == "grpo"]
    last = DYNAMICS_STEPS[-1]
    lens, lows = [], []
    for r in grpo:
        l0, l1 = (
            _window(r, 0, "completions_mean_length"),
            _window(r, last, "completions_mean_length"),
        )
        c1 = _window(r, last, "completions_clipped_ratio")
        if l0 is not None and l1 is not None and c1 is not None:
            lens.append(f"{r.run_dir.name} {l0:.0f} → {l1:.0f} ({100 * c1:.1f} %)")
        ws = [w for s in range(WINDOW, last + 1) if (w := _window(r, s, "reward_mean")) is not None]
        if ws:
            lows.append(f"{r.run_dir.name} {min(ws):.3f}")
    training = (
        f"NaN / inf in loss, grad_norm, reward_mean: {len(bad)}" + (f" ({', '.join(bad[:5])})" if bad else "")
        + f". Lowest {WINDOW}-step mean train reward per GRPO run: {'; '.join(lows) or MISSING}. Mean training completion length, steps 1–{WINDOW} → "
        f"{last - WINDOW + 1}–{last}, tokens (truncation % of training completions in the last window): {'; '.join(lens) or MISSING}"
    )  # fmt: skip
    trained = {w: r for w, r in results["runs"].items() if r["kind"] in ("arm", "control")}
    flagged = "; ".join(
        f"{a['label']} seeds {','.join(map(str, a['cells']['test_greedy']['flagged_seeds']))}"
        for a in results["arms"].values()
        if a["cells"]["test_greedy"] and a["cells"]["test_greedy"]["flagged_seeds"]
    )
    ood = max(
        ((c["truncation_rate"], w) for w, r in trained.items() if (c := r["cells"]["ood_greedy"])),
        default=(float("nan"), ""),
    )
    alr = sorted(
        (c["answer_line_rate"], w) for w, r in trained.items() if (c := r["cells"]["test_greedy"])
    )[:3]
    base = results["runs"]["base"]["cells"]["test_greedy"]
    dirty = [u for r in ds.all_runs() for u in r.units.values() if u.meta.get("git_dirty")]
    evals = (
        f"Seeds with truncation > 5 % on test_300 greedy (SPEC §7 flag; no correction applied anywhere): {flagged or 'none'}; highest "
        f"ood_hard_200 greedy truncation {p1(ood[0])} % ({ood[1]}). Lowest answer-line rates on test_300 greedy: "
        f"{'; '.join(f'{w} {p1(v)} %' for v, w in alr)} (base {p1(base['answer_line_rate']) if base else MISSING} %). Eval units recorded with git_dirty = true: "
        f"{len(dirty)} at SHAs {', '.join(sorted({str(u.meta.get('git_sha', ''))[:7] for u in dirty})) or 'none'}"
    )  # fmt: skip
    return [
        [training, "train_log.jsonl; section 9"],
        [evals, "sections 4 and 10; sanity.md; tables/units.csv"],
    ]


def section_anomalies(ds: Dataset, results: dict[str, Any], notes: dict[str, Any]) -> list[str]:
    rows = mechanical_anomalies(ds, results)
    rows += [
        [" ".join(str(n["what"]).split()), " ".join(str(n["where"]).split())]
        for n in notes.get("anomalies", [])
    ]
    return [
        "## 12. Anomalies, deviations and workarounds noticed while producing this packet (rows 1–2 computed from the run root; the rest from `configs/analysis/packet_notes.yaml`)",
        md_table(["#", "item", "recorded in"], [[i, *r] for i, r in enumerate(rows, 1)]),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# flat csv
# ---------------------------------------------------------------------------


def flat_rows(ds: Dataset, results: dict[str, Any]) -> list[list[Any]]:
    """One row per (arm, seed, split, metric); seed is an int, or 'mean' / 'std' across seeds."""
    rows: list[list[Any]] = []
    names = {"greedy": "greedy_accuracy", "mean_at_k": "mean_at_8_accuracy"}
    for run in ds.all_runs():
        for (split, dec), u in sorted(run.units.items()):
            m, n = u.metrics, int(u.metrics["n_problems"])
            if dec in names:
                rows.append(
                    [
                        run.label,
                        run.seed,
                        split,
                        names[dec],
                        m["accuracy"],
                        m["ci_low"],
                        m["ci_high"],
                        n,
                    ]
                )
            else:
                for k in sorted(m["pass_at_k"], key=int):
                    lo, hi = bootstrap_ci(
                        u.pass_at(int(k)), n_boot=int(ds.cfg["n_boot"]), seed=u.seed
                    )
                    rows.append(
                        [run.label, run.seed, split, f"pass_at_{k}", m["pass_at_k"][k], lo, hi, n]
                    )
            for f in (
                "truncation_rate",
                "extraction_failure_rate",
                "answer_line_rate",
                "mean_completion_tokens",
            ):
                rows.append([run.label, run.seed, split, f"{dec}_{f}", m[f], "", "", n])
    for key, runs in _groups(ds).items():
        if len(runs) < 2:
            continue
        for mkey, split, dec, k, _ in report.METRICS:
            c = results["arms"][key]["cells"][mkey]
            if c is None:
                continue
            metric = names.get(dec, f"pass_at_{k}")
            rows.append(
                [
                    runs[0].label,
                    "mean",
                    split,
                    metric,
                    c["mean"],
                    c["ci_low"],
                    c["ci_high"],
                    c["n_problems"],
                ]
            )
            rows.append([runs[0].label, "std", split, metric, c["std"], "", "", c["n_problems"]])
    return rows


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_packet(
    ds: Dataset, rep: sanity.CrossRunReport, results: dict[str, Any], notes: dict[str, Any],
    hypotheses: str | None, hashes: dict[str, dict[str, Any]], date: str, meta: dict[str, Any],
) -> str:  # fmt: skip
    budgets, per_arm = section_budgets(ds, results)
    sections = [
        section_provenance(ds, date, meta), section_sanity(rep), section_manifest(ds, hashes),
        section_base_refs(ds, results), section_main(ds, results), section_contrasts(results, notes),
        section_tiers(ds), budgets, section_sweep(ds), section_dynamics(ds),
        section_gaps(ds, results, per_arm),
    ]  # fmt: skip
    failed = bool(rep.failures)
    lines = ["SANITY FAILED"] if failed else []
    for i, sec in enumerate(sections):
        head = sec[0] + (" — **DEPENDS ON A FAILED SANITY CHECK**" if failed and i >= 2 else "")
        lines += [head, *sec[1:]]
    lines.append(
        "## 11. `hypotheses.md`, verbatim (verdict lines blank)"
        + (" — **DEPENDS ON A FAILED SANITY CHECK**" if failed else "")
    )
    lines += hypotheses.rstrip("\n").split("\n") if hypotheses else ["not written (sanity failed)"]
    lines += section_anomalies(ds, results, notes)
    flat = [x for line in lines for x in str(line).split("\n")]
    return "\n".join(flat) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rlordata.analysis.packet")
    ap.add_argument("--config", default=str(loader.DEFAULT_CONFIG))
    ap.add_argument("--notes", default=str(DEFAULT_NOTES))
    ap.add_argument("--run-root", default=None, help="override run_root (read-only)")
    ap.add_argument(
        "--out", default=None, help="override out_dir (must hold make analysis outputs)"
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--date", default=None, help="date printed in section 0 (default: today, UTC)")
    args = ap.parse_args(argv)

    cfg = loader.load_config(args.config, run_root=args.run_root, out_dir=args.out)
    out = Path(cfg["out_dir"])
    if out.resolve().is_relative_to(Path(cfg["run_root"]).resolve()):
        raise SystemExit(f"refusing to write into the run root: {out}")
    notes = yaml.safe_load(Path(args.notes).read_text(encoding="utf-8")) or {}
    cache = out / ".cache"
    ds = loader.load_dataset(cfg, cache_dir=cache / "units", jobs=args.jobs)
    rep = sanity.cross_run_checks(ds, splits_dir=cfg["splits_dir"], cache_dir=cache)
    results = report.build_results(ds)
    hashes = sanity.adapter_hashes(ds, cache_path=cache / "adapter_sha256.json")
    hyp_path = out / "hypotheses.md"
    hypotheses = hyp_path.read_text(encoding="utf-8") if hyp_path.exists() else None
    if hypotheses is not None and not rep.failures:
        fresh = report.hypotheses_markdown(ds, results, loader.analysis_config_hash(cfg))
        if fresh != hypotheses:
            raise SystemExit(f"{hyp_path} is stale or edited: run `make analysis` first")
    date = args.date or dt.datetime.now(dt.UTC).date().isoformat()
    meta_path = out / "analysis_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    text = build_packet(ds, rep, results, notes, hypotheses, hashes, date, meta)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_packet.md").write_text(text, encoding="utf-8")
    with (out / "results_flat.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FLAT_HEADER)
        w.writerows(flat_rows(ds, results))
    n = len(text.splitlines())
    print(f"[packet] {out / 'results_packet.md'}: {n} lines (limit {MAX_LINES})")
    if n > MAX_LINES:
        print(f"[packet] over the {MAX_LINES}-line limit by {n - MAX_LINES}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
