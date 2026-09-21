"""Figures for the write-up. AGENT-OWNED. Outputs to outputs/figures/. Every figure shows seeds and CIs.

tasks/05 item 4, (a)–(h), plus (i) eval-time length/truncation by arm because truncation is the
known confound of every RFT-vs-GRPO contrast. Conventions, fixed for every figure:

  * hue = training signal (RFT blue, GRPO orange, controls aqua, base/reference gray) — three
    hues that pass the all-pairs colour-vision checks; data condition is carried by position,
    panel or marker shape, never by a fourth hue;
  * every seed is a dot with its per-problem bootstrap 95 % CI; the arm's mean ± seed std is the
    dark tick-and-bar next to the dots;
  * a hollow dot = that run exceeds 5 % truncation on the split (SPEC §7 flag on val/test;
    reported-only on ood);
  * one y-axis per panel; small multiples instead of many overlaid series.

PNG output is byte-reproducible (Agg backend, pinned rcParams, no timestamp/software metadata).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rlordata.analysis import stats  # noqa: E402
from rlordata.analysis.loader import PASS_KS, Dataset, Run  # noqa: E402
from rlordata.analysis.sanity import TRUNCATION_MAX  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"rft": "#2a78d6", "grpo": "#eb6834", "control": "#1baf7a", "base": "#7a7974",
          "iter_rft": "#8a5cd6"}  # iter_rft: tasks/06b secondary arm  # fmt: skip
MARKERS = {"train_easy_100": "o", "train_mixed_100": "s", "train_curated": "^", None: "D"}
TIERS = ("easy", "medium", "hard")
SMOOTH = 25  # steps; rolling window for training curves

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "svg.hashsalt": "rlordata",
}


def _color(kind: str, method: str | None) -> str:
    if kind == "control":
        return COLORS["control"]
    return COLORS.get(method or "base", COLORS["base"])


def _save(fig: plt.Figure, path: Path, caption: str) -> None:
    fig.text(0.01, 0.005, caption, fontsize=7.5, color=INK_2, ha="left", va="bottom", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)


def _columns(ds: Dataset, results: dict[str, Any], *, refs: bool = False) -> list[dict[str, Any]]:
    """Plot columns in a fixed order: Base, the six arms, the controls (and optionally references)."""
    cols = [{"label": "Base", "kind": "base", "method": None, "whos": ["base"], "arm": None}]
    for key, a in results["arms"].items():
        cols.append(
            {
                "label": a["label"].replace("-", "\n", 1).replace(" ", "\n"),
                "kind": a["kind"],
                "method": a["method"],
                "whos": [f"{key}/seed{s}" for s in a["seeds"]],
                "arm": key,
            }
        )
    if refs:
        for r in ds.references:
            cols.append(
                {
                    "label": "ref\n" + r.label.split(" ")[0].replace("-Instruct", "\nInstruct"),
                    "kind": "reference",
                    "method": None,
                    "whos": [r.key],
                    "arm": None,
                }
            )
    return cols


def _dot_columns(
    ax: plt.Axes,
    cols: list[dict[str, Any]],
    cells: dict[str, dict[str, Any] | None],
    summaries: dict[str, Any],
) -> None:
    """Every run a dot with its CI; arm mean ± seed std as a dark bar to the right of the dots."""
    for x, col in enumerate(cols):
        color = _color(col["kind"], col["method"])
        present = [(w, cells.get(w)) for w in col["whos"] if cells.get(w) is not None]
        offsets = np.linspace(-0.22, 0.10, len(present)) if len(present) > 1 else [0.0]
        for off, (_, c) in zip(offsets, present, strict=True):
            hollow = c["truncation_rate"] > TRUNCATION_MAX
            ax.errorbar(
                x + off, c["value"], yerr=[[c["value"] - c["ci_low"]], [c["ci_high"] - c["value"]]],
                fmt="none", ecolor=color, elinewidth=1.0, alpha=0.55, zorder=2,
            )  # fmt: skip
            ax.plot(
                x + off, c["value"], marker="o", markersize=6.5, zorder=3, linestyle="none",
                markerfacecolor=SURFACE if hollow else color, markeredgecolor=color, markeredgewidth=1.6,
            )  # fmt: skip
        s = summaries.get(col["arm"]) if col["arm"] else None
        if s is not None and len(s["seeds"]) > 1:
            ax.errorbar(
                x + 0.30, s["mean"], yerr=s["std"], fmt="_", color=INK, markersize=11,
                markeredgewidth=2.0, elinewidth=2.0, capsize=0, zorder=4,
            )  # fmt: skip
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c["label"] for c in cols], fontsize=8)
    ax.grid(axis="x", visible=False)


def _legend_dots(ax: plt.Axes, *, iter_rft: bool = False) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["rft"], markeredgecolor=COLORS["rft"], label="RFT seed (95 % CI)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["grpo"], markeredgecolor=COLORS["grpo"], label="GRPO seed"),
        *(
            [Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["iter_rft"], markeredgecolor=COLORS["iter_rft"], label="IterRFT seed (secondary, registered after unblinding)")]
            if iter_rft
            else []
        ),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["control"], markeredgecolor=COLORS["control"], label="control (1 seed)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=COLORS["base"], markeredgecolor=COLORS["base"], label="base / reference"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=1.6, label="hollow: truncation > 5 %"),
        Line2D([], [], marker="_", linestyle="none", color=INK, markersize=11, markeredgewidth=2.0, label="arm mean ± seed std"),
    ]  # fmt: skip
    ax.legend(
        handles=handles,
        loc="upper left",
        ncols=3,
        fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.2,
    )


def fig_accuracy_by_arm(
    ds: Dataset, results: dict[str, Any], metric: str, path: Path, title: str, note: str
) -> None:
    cols = _columns(ds, results, refs=True)
    cells = {w: r["cells"][metric] for w, r in results["runs"].items()}
    summaries = {k: a["cells"][metric] for k, a in results["arms"].items()}
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(
            10.5 + 0.9 * max(0, len(cols) - 12),
            6.6,
        ),  # 12 columns in Phase 4; wider per extra arm
        height_ratios=[3, 1.25],
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    _dot_columns(ax, cols, cells, summaries)
    n = next(c["n_problems"] for c in cells.values() if c)
    ax.set_title(f"{title} (n = {n} problems; seeds 1–3)")
    ax.set_ylabel("greedy accuracy")
    ymax = max(c["ci_high"] for c in cells.values() if c)
    ax.set_ylim(0, min(1.0, ymax + 0.22))
    _legend_dots(ax, iter_rft=any(c["method"] == "iter_rft" for c in cols))
    for x, col in enumerate(cols):
        present = [cells[w] for w in col["whos"] if cells.get(w) is not None]
        offsets = np.linspace(-0.22, 0.10, len(present)) if len(present) > 1 else [0.0]
        for off, c in zip(offsets, present, strict=True):
            ax2.vlines(
                x + off,
                0,
                100 * c["truncation_rate"],
                color=_color(col["kind"], col["method"]),
                linewidth=3.0,
            )
    ax2.set_ylabel("truncated at\nthe cap (%)")
    ax2.grid(axis="x", visible=False)
    if metric != "ood_greedy":
        ax2.axhline(100 * TRUNCATION_MAX, color=INK_2, linewidth=0.8)
        ax2.text(
            -0.45,
            100 * TRUNCATION_MAX,
            "SPEC §7 flag: 5 %",
            fontsize=7.5,
            color=INK_2,
            va="bottom",
            ha="left",
        )
    ax2.set_ylim(bottom=0)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.17)
    _save(fig, path, note)


def fig_per_tier(ds: Dataset, results: dict[str, Any], path: Path) -> None:
    cols = _columns(ds, results)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.4), sharex=True, gridspec_kw={"hspace": 0.18})
    for ax, tier in zip(axes, TIERS, strict=True):
        cells = {w: r["tiers"].get(tier) for w, r in results["runs"].items()}
        summaries = {
            k: a["tiers"].get(tier) and {**a["tiers"][tier]} for k, a in results["arms"].items()
        }
        _dot_columns(ax, cols, cells, summaries)
        n = next(c["n_problems"] for c in cells.values() if c)
        ax.set_title(f"{tier} tier (n = {n}; tiers by the base model's pass@8)")
        ax.set_ylabel("test greedy accuracy")
        ax.set_ylim(0, 1.05)
    _legend_dots(axes[0], iter_rft=any(c["method"] == "iter_rft" for c in cols))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.12)
    _save(
        fig, path,
        "test_300 greedy accuracy per tier. Dots: seeds with per-problem bootstrap 95 % CIs; dark bar: mean ± seed std. "
        "Hollow = that tier's truncation > 5 % for the run.",
    )  # fmt: skip


def fig_pass_at_k(ds: Dataset, results: dict[str, Any], path: Path) -> None:
    n_boot, seed = int(ds.cfg["n_boot"]), int(ds.cfg["seed"])
    key = ("test_300", "pass_at_k")
    groups = {**ds.arms, **ds.controls}
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12.5, 6.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.08},
    )
    base = ds.base.units.get(key)
    ks = list(PASS_KS)
    for ax, runs in zip(axes.flat, groups.values(), strict=False):
        color = _color(runs[0].kind, runs[0].method)
        if base is not None:
            ax.plot(
                ks,
                [base.metrics["pass_at_k"][str(k)] for k in ks],
                color=COLORS["base"],
                marker="o",
                markersize=4,
                label="Base (1 eval)",
            )
        units = [r.units[key] for r in runs if key in r.units]
        if units:
            for u in units:
                ax.plot(
                    ks,
                    [u.metrics["pass_at_k"][str(k)] for k in ks],
                    color=color,
                    linewidth=0.9,
                    alpha=0.6,
                )
            summ = [
                stats.summarize_arm(
                    [r.seed for r in runs], [u.pass_at(k) for u in units], n_boot=n_boot, seed=seed
                )
                for k in ks
            ]
            ax.fill_between(
                ks,
                [s.ci_low for s in summ],
                [s.ci_high for s in summ],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                ks,
                [s.mean for s in summ],
                color=color,
                marker="o",
                markersize=4,
                label=f"{runs[0].label}, mean of {len(units)}"
                if len(units) > 1
                else f"{runs[0].label} (1 seed)",
            )
            tr = [u.metrics["truncation_rate"] for u in units]
            ax.text(
                0.97,
                0.04,
                "truncation " + " / ".join(f"{100 * t:.1f}%" for t in tr),
                transform=ax.transAxes,
                ha="right",
                fontsize=7.5,
                color=INK_2,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(runs[0].label)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.09), fontsize=7.5)
        ax.set_ylim(0.25, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("pass@k (unbiased, n = 64)")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.14)
    _save(
        fig, path,
        "pass@k on the first 100 problems of test_300, 64 samples per problem at T = 1. Thin lines: seeds; thick: mean across seeds; "
        "band: bootstrap 95 % CI over problems of the seed-averaged pass@k. A curve above base at k = 1 that meets or crosses it at large k is "
        "sharpening; one above base at every k is expansion.",
    )  # fmt: skip


def _rolling(
    values: list[float | None], weights: list[float] | None = None, window: int = SMOOTH
) -> np.ndarray:
    """Trailing weighted mean over ``window`` steps, ignoring None; nan where nothing is available."""
    v = np.asarray([np.nan if x is None else float(x) for x in values], dtype=np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    w = np.where(np.isnan(v), 0.0, w)
    v = np.where(np.isnan(v), 0.0, v)
    num = np.cumsum(v * w)
    den = np.cumsum(w)
    num[window:] = num[window:] - num[:-window]
    den[window:] = den[window:] - den[:-window]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _steps(run: Run) -> list[dict[str, Any]]:
    return [r for r in run.train_log if not r.get("summary") and r.get("reward_mean") is not None]


def _grpo_groups(ds: Dataset, *, controls: bool) -> dict[str, list[Run]]:
    groups = {k: v for k, v in ds.arms.items() if v and v[0].method == "grpo"}
    if controls:
        groups |= {k: v for k, v in ds.controls.items() if v and v[0].method == "grpo"}
    return groups


def fig_reward_vs_val(ds: Dataset, path: Path) -> None:
    groups = _grpo_groups(ds, controls=True)
    fig, axes = plt.subplots(
        1, len(groups), figsize=(3.0 * len(groups), 3.9), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    for ax, (_, runs) in zip(np.atleast_1d(axes), groups.items(), strict=True):
        color = _color(runs[0].kind, runs[0].method)
        for i, run in enumerate(runs):
            rows = _steps(run)
            ax.plot(
                [r["step"] for r in rows],
                _rolling([r["reward_mean"] for r in rows]),
                color=color,
                linewidth=1.0,
                alpha=0.7,
                label=f"training reward, {SMOOTH}-step mean (per seed)" if i == 0 else None,
            )
            curve = run.curves.get("val_greedy") or {}
            xs = sorted(int(s) for s in curve)
            for j, s in enumerate(xs):
                c = curve[str(s)]
                off = (i - (len(runs) - 1) / 2) * 7
                ax.errorbar(
                    s + off,
                    c["accuracy"],
                    yerr=[[c["accuracy"] - c["ci"][0]], [c["ci"][1] - c["accuracy"]]],
                    fmt="o",
                    color=INK,
                    markersize=4.5,
                    elinewidth=0.9,
                    markeredgecolor=SURFACE,
                    markeredgewidth=0.8,
                    label="val_mixed_100 greedy (n = 100, 95 % CI)" if i == 0 and j == 0 else None,
                    zorder=3,
                )
        ax.set_title(runs[0].label)
        ax.set_xlabel("optimizer step")
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel("fraction correct / reward")
    np.atleast_1d(axes)[0].legend(loc="lower right", fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.27)
    _save(
        fig, path,
        "GRPO training reward on the training prompts (coloured, one line per seed) and val_mixed_100 greedy accuracy of the step-100/200/300 adapters "
        "(black, offset per seed). For C1 the reward is Bernoulli(0.5) and for C2 it is answer-line presence, so only the black dots measure correctness there. "
        "RFT has no on-policy training curve; its val numbers are in tables/results.md.",
    )  # fmt: skip


def fig_zero_std(ds: Dataset, path: Path) -> None:
    groups = _grpo_groups(ds, controls=False)
    panels = ("all", *TIERS)
    fig, axes = plt.subplots(
        len(groups),
        len(panels),
        figsize=(12, 2.6 * len(groups) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.3, "wspace": 0.06},
    )
    for row, (_, runs) in zip(axes, groups.items(), strict=True):
        for ax, panel in zip(row, panels, strict=True):
            for run in runs:
                rows = _steps(run)
                if panel == "all":
                    ys = _rolling([r["frac_reward_zero_std"] for r in rows])
                else:
                    tier = [(r.get("per_tier") or {}).get(panel) for r in rows]
                    ys = _rolling(
                        [t and t.get("frac_reward_zero_std") for t in tier],
                        [float((t or {}).get("n", 0)) / 8.0 for t in tier],
                    )
                ax.plot(
                    [r["step"] for r in rows], ys, color=COLORS["grpo"], linewidth=1.1, alpha=0.8
                )
            ax.set_title(
                f"{runs[0].label} — {'all prompts' if panel == 'all' else panel + ' tier'}",
                fontsize=9,
            )
            ax.set_ylim(0, 1.02)
        row[0].set_ylabel("groups with zero\nreward std (share)")
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.13)
    _save(
        fig, path,
        f"Share of the 8 prompt groups per step whose 8 completions all got the same reward (zero advantage, no gradient): GRPO's implicit prompt filter. One line per seed, trailing {SMOOTH}-step mean; "
        "tier panels weight each step by the number of that tier's prompts in the batch. train_easy_100 has only easy prompts.",
    )  # fmt: skip


def fig_budget(ds: Dataset, results: dict[str, Any], path: Path) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, field, label in zip(
        axes,
        ("training_tokens", "optimizer_steps"),
        ("training tokens", "optimizer steps"),
        strict=True,
    ):
        for who, b in results["budgets"].items():
            r = results["runs"][who]
            c = r["cells"]["test_greedy"]
            if c is None or b[field] is None:
                continue
            color = _color(r["kind"], r["method"])
            hollow = c["truncation_rate"] > TRUNCATION_MAX
            xval = b[field] * (1.0 + 0.025 * (r["seed"] - 2))  # seeds of an RFT arm share a budget
            ax.errorbar(
                xval,
                c["value"],
                yerr=[[c["value"] - c["ci_low"]], [c["ci_high"] - c["value"]]],
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                alpha=0.5,
            )
            ax.plot(
                xval,
                c["value"],
                marker=MARKERS[r["data_condition"]] if r["kind"] == "arm" else "D",
                markersize=7,
                linestyle="none",
                markerfacecolor=SURFACE if hollow else color,
                markeredgecolor=color,
                markeredgewidth=1.6,
            )
        base = results["runs"]["base"]["cells"]["test_greedy"]
        ax.axhline(base["value"], color=COLORS["base"], linewidth=1.2)
        ax.text(
            0.99,
            base["value"],
            "Base (no training) ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
        )
        ax.set_xscale("log")
        ticks = (1e7, 2e7, 5e7) if field == "training_tokens" else (300, 1000, 2000, 5000)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(v):,}" for v in ticks])
        ax.minorticks_off()
        ax.set_xlabel(f"{label} (log scale)")
    axes[0].set_ylabel("test_300 greedy accuracy (n = 300)")
    axes[0].set_title("Accuracy vs gradient/token budget, every run")
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=COLORS["rft"], label="RFT"),
        Line2D([], [], marker="o", linestyle="none", color=COLORS["grpo"], label="GRPO"),
        Line2D([], [], marker="D", linestyle="none", color=COLORS["control"], label="controls C1/C2"),
        Line2D([], [], marker="o", linestyle="none", color=INK_2, label="easy"),
        Line2D([], [], marker="s", linestyle="none", color=INK_2, label="mixed"),
        Line2D([], [], marker="^", linestyle="none", color=INK_2, label="curated"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=1.6, label="hollow: truncation > 5 %"),
    ]  # fmt: skip
    axes[1].legend(handles=handles, loc="lower right", ncols=2, fontsize=7.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.2)
    _save(
        fig, path,
        "One marker per run (seeds 1–3, shifted ±2.5 % in x so identical budgets stay visible), bootstrap 95 % CI over problems. All runs had the same 19,200-completion generation budget; RFT trains for several epochs on the "
        "verified-correct subset, GRPO once on everything (SPEC §8: the gradient/token budget cannot be equalised without changing the method).",
    )  # fmt: skip


def fig_length_over_training(ds: Dataset, path: Path) -> None:
    groups = _grpo_groups(ds, controls=True)
    cap = None
    fig, axes = plt.subplots(
        2,
        len(groups),
        figsize=(3.0 * len(groups), 5.6),
        sharex=True,
        sharey="row",
        squeeze=False,
        gridspec_kw={"hspace": 0.16, "wspace": 0.08},
    )
    for col, (_, runs) in enumerate(groups.items()):
        color = _color(runs[0].kind, runs[0].method)
        for run in runs:
            rows = _steps(run)
            xs = [r["step"] for r in rows]
            axes[0][col].plot(
                xs,
                _rolling([r["completions_mean_length"] for r in rows]),
                color=color,
                linewidth=1.1,
                alpha=0.8,
            )
            axes[1][col].plot(
                xs,
                100 * _rolling([r["completions_clipped_ratio"] for r in rows]),
                color=color,
                linewidth=1.1,
                alpha=0.8,
            )
            u = run.units.get(("test_300", "greedy"))
            cap = cap or (u and int(u.config["max_completion_tokens"]))
        axes[0][col].set_title(runs[0].label)
        axes[1][col].set_xlabel("optimizer step")
    if cap:
        for ax in axes[0]:
            ax.axhline(cap, color=INK_2, linewidth=0.8)
        axes[0][-1].text(
            0.99,
            cap,
            f"cap {cap:,} ",
            transform=axes[0][-1].get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=7.5,
            color=INK_2,
        )
    axes[0][0].set_ylabel("mean completion length\n(tokens)")
    axes[1][0].set_ylabel("rollouts cut at the cap (%)")
    axes[1][0].set_ylim(bottom=0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.17)
    _save(
        fig, path,
        f"GRPO training rollouts, one line per seed, trailing {SMOOTH}-step mean. RFT trains on a fixed set of base-model completions and generates nothing during training, "
        "so it has no such curve; eval-time lengths for every arm are in figure (i).",
    )  # fmt: skip


def fig_eval_length(ds: Dataset, results: dict[str, Any], path: Path) -> None:
    cols = _columns(ds, results)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 6.4), sharex=True, gridspec_kw={"hspace": 0.14, "wspace": 0.16}
    )
    for j, (metric, name) in enumerate(
        (("test_greedy", "test_300"), ("ood_greedy", "ood_hard_200"))
    ):
        for i, (field, scale, ylabel) in enumerate(
            (
                ("mean_completion_tokens", 1.0, "mean completion\nlength (tokens)"),
                ("truncation_rate", 100.0, "truncated at the cap (%)"),
            )
        ):
            ax = axes[i][j]
            for x, col in enumerate(cols):
                cells = [results["runs"][w]["cells"][metric] for w in col["whos"]]
                cells = [c for c in cells if c is not None]
                offsets = np.linspace(-0.16, 0.16, len(cells)) if len(cells) > 1 else [0.0]
                ax.plot(
                    [x + o for o in offsets],
                    [scale * c[field] for c in cells],
                    marker="o",
                    markersize=6.5,
                    linestyle="none",
                    color=_color(col["kind"], col["method"]),
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.0,
                )
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([c["label"] for c in cols], fontsize=8)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(bottom=0)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                n = next(
                    results["runs"][w]["cells"][metric]["n_problems"]
                    for c_ in cols
                    for w in c_["whos"]
                    if results["runs"][w]["cells"][metric]
                )
                ax.set_title(f"{name} greedy (n = {n} completions per run)")
            if i == 1 and j == 0:
                ax.axhline(100 * TRUNCATION_MAX, color=INK_2, linewidth=0.8)
                ax.text(
                    len(cols) - 0.5,
                    100 * TRUNCATION_MAX,
                    "SPEC §7 flag: 5 % ",
                    fontsize=7.5,
                    color=INK_2,
                    va="bottom",
                    ha="right",
                )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.15)
    _save(
        fig, path,
        "Eval-time completion length and truncation, one dot per seed (exact counts, no sampling error bar: every greedy completion of the split is included). "
        "A truncated completion is scored wrong by rule (SPEC §5). On ood_hard_200 truncation is reported, not flagged.",
    )  # fmt: skip


def write_figures(ds: Dataset, results: dict[str, Any], out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    with plt.rc_context(RC):
        note = (
            "Dots: seeds with per-problem bootstrap 95 % CIs (10,000 resamples); dark bar: arm mean ± seed std. Hollow = truncation > 5 %. "
            "Lower panel: share of that run's completions cut at the 4,352-token cap and therefore scored wrong."
        )
        jobs = [
            ("a_test_accuracy_by_arm.png", lambda p: fig_accuracy_by_arm(ds, results, "test_greedy", p, "(a) test_300 greedy accuracy by arm", note)),
            ("b_ood_accuracy_by_arm.png", lambda p: fig_accuracy_by_arm(ds, results, "ood_greedy", p, "(b) ood_hard_200 greedy accuracy by arm", note + " On ood truncation is reported, not flagged (SPEC §7).")),
            ("c_per_tier_accuracy.png", lambda p: fig_per_tier(ds, results, p)),
            ("d_pass_at_k.png", lambda p: fig_pass_at_k(ds, results, p)),
            ("e_grpo_reward_vs_val.png", lambda p: fig_reward_vs_val(ds, p)),
            ("f_frac_reward_zero_std.png", lambda p: fig_zero_std(ds, p)),
            ("g_accuracy_vs_budget.png", lambda p: fig_budget(ds, results, p)),
            ("h_completion_length_training.png", lambda p: fig_length_over_training(ds, p)),
            ("i_eval_length_truncation.png", lambda p: fig_eval_length(ds, results, p)),
        ]  # fmt: skip
        for name, make in jobs:
            make(out_dir / name)
            paths.append(out_dir / name)
    return paths
