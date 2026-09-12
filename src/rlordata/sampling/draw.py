"""The matched-budget sample draw for the RFT arms (SPEC §8, tasks/03 §1). AGENT-OWNED.

For every prompt in a training split, the base model's samples are the **8 tiering samples**
already drawn in Phase 1 (``data/samples/tiering_pass8.jsonl``, generation order preserved)
**plus 184 new samples** at T=1.0 / top_p=1.0 / the locked cap, giving exactly 192 per prompt
(19,200 for a 100-prompt split = GRPO's 300 × 8 × 8 rollout budget). The tiering 8 come first so
``core.rft_select``'s "first 8 in generation order" is the same 8 that defined the tiers.

Drawn once, written to ``data/samples/base_{split}_k{192}_seed{seed}.jsonl`` and made read-only.
Every RFT arm and seed reads these files; nothing re-samples.

Seeding: the tiering run used ``VLLMSampler(seed=1)``, whose per-prompt seeds are
``1 + pool_index * 8``. If the new draw also used ``seed=1`` its per-prompt seeds would be
``1 + split_index * 184`` and could coincide with a tiering seed, reproducing a tiering completion
verbatim. The draw therefore offsets the sampler seed by :data:`DRAW_SEED_OFFSET`; the run seed
stays 1 and both values are recorded in the resolved config.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rlordata.core.verify import verify
from rlordata.sampling.eval_runner import build_samples, read_samples, write_samples
from rlordata.sampling.prompts import format_prompt
from rlordata.types import Problem, Sample

SAMPLES_PER_PROMPT = 192  # SPEC §8 / training.yaml rft.samples_per_prompt
TIERING_K = 8  # SPEC §6.1
DRAW_SEED_OFFSET = 100_000  # see module docstring
DEFAULT_SAMPLES_DIR = Path("data/samples")
DEFAULT_TIERING_SAMPLES = DEFAULT_SAMPLES_DIR / "tiering_pass8.jsonl"


def draw_path(split: str, seed: int, *, samples_dir: str | Path = DEFAULT_SAMPLES_DIR) -> Path:
    return Path(samples_dir) / f"base_{split}_k{SAMPLES_PER_PROMPT}_seed{seed}.jsonl"


# ---------------------------------------------------------------------------
# Tiering samples
# ---------------------------------------------------------------------------


def load_tiering_samples(
    path: str | Path, problems: list[Problem], *, k: int = TIERING_K
) -> dict[str, list[Sample]]:
    """The first ``k`` tiering samples of each problem, in generation order (``extra.sample_idx``).

    Checks, so a wrong or stale samples file cannot slip through: exactly ``k`` per problem; the
    stored verdicts agree with ``core.verify`` under the current rule; the number correct equals
    the problem's frozen ``pass8``.
    """
    wanted = {p.problem_id: p for p in problems}
    by_pid: dict[str, list[Sample]] = {pid: [] for pid in wanted}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec["problem_id"]
            if pid in by_pid:
                by_pid[pid].append(Sample(**rec))
    problems_missing = [pid for pid, ss in by_pid.items() if len(ss) != k]
    if problems_missing:
        raise ValueError(
            f"{path}: {len(problems_missing)} problem(s) do not have exactly {k} tiering samples "
            f"(first {problems_missing[0][:12]})"
        )
    for pid, ss in by_pid.items():
        ss.sort(key=lambda s: int(s.extra["sample_idx"]))
        if [int(s.extra["sample_idx"]) for s in ss] != list(range(k)):
            raise ValueError(f"{path}: sample_idx of {pid[:12]} is not 0..{k - 1}")
        problem = wanted[pid]
        n_correct = 0
        for s in ss:
            v = verify(problem, s.completion, truncated=bool(s.truncated))
            if bool(v.reward == 1.0) != bool(s.correct):
                raise ValueError(
                    f"{path}: stored verdict for {pid[:12]} sample {s.extra['sample_idx']} disagrees "
                    "with core.verify under the current rule; rescore the tiering file first"
                )
            n_correct += int(s.correct)
        if problem.pass8 is not None and n_correct != int(problem.pass8):
            raise ValueError(
                f"{path}: {pid[:12]} has {n_correct} correct of {k} but the split records pass8="
                f"{problem.pass8}; the samples file does not match data/splits"
            )
    return by_pid


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def draw_split(
    problems: list[Problem],
    tiering: dict[str, list[Sample]],
    sampler: Any,
    *,
    split: str,
    run_id: str,
    config_hash: str,
    seed: int,
    n_total: int = SAMPLES_PER_PROMPT,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[Sample]:
    """``n_total`` samples per problem: the tiering samples first, then new draws from ``sampler``.

    ``sampler`` must already carry the offset seed (``seed + DRAW_SEED_OFFSET``); this function
    only asserts it does not equal the run seed. Returns one ``Sample`` per completion with
    ``extra.sample_idx`` in 0..n_total-1 and ``extra.source`` in {"tiering", "draw"}.
    """
    assert len(problems) > 0
    k = len(next(iter(tiering.values())))
    assert all(len(v) == k for v in tiering.values()), "tiering samples must be uniform"
    n_new = n_total - k
    assert n_new >= 1, f"n_total={n_total} must exceed the {k} tiering samples"
    assert int(getattr(sampler, "seed", -1)) != int(seed), (
        "the draw sampler must use seed + DRAW_SEED_OFFSET, not the run seed (see draw.py)"
    )
    if getattr(sampler, "model_kind", "base") != "base":
        raise ValueError("the RFT draw samples the base policy with the plain TEMPLATE")
    prompts = [format_prompt(p, "base") for p in problems]
    for p, prompt in zip(problems, prompts, strict=True):
        stored = tiering[p.problem_id][0].prompt
        if stored != prompt:
            raise ValueError(
                f"prompt for {p.problem_id[:12]} differs from the tiering prompt; the template drifted"
            )
    completions = sampler.sample(prompts, n=n_new, temperature=temperature, top_p=top_p)
    new_samples = build_samples(
        problems,
        prompts,
        completions,
        run_id=run_id,
        config_hash=config_hash,
        seed=seed,
        arm="base",
        data_condition=split,
        extra={
            "model_id": sampler.model_id,
            "decoding": "rft_draw",
            "temperature": temperature,
            "top_p": top_p,
            "sampler_seed": int(sampler.seed),
            "source": "draw",
        },
    )
    out: list[Sample] = []
    new_by_pid: dict[str, list[Sample]] = {}
    for s in new_samples:
        new_by_pid.setdefault(s.problem_id, []).append(s)
    for p in problems:
        pid = p.problem_id
        for j, s in enumerate(tiering[pid]):
            out.append(
                Sample(
                    **{
                        **s.to_dict(),
                        "data_condition": split,
                        "extra": {**s.extra, "sample_idx": j, "source": "tiering"},
                    }
                )
            )
        fresh = new_by_pid[pid]
        assert len(fresh) == n_new
        for j, s in enumerate(fresh):
            out.append(Sample(**{**s.to_dict(), "extra": {**s.extra, "sample_idx": k + j}}))
    assert len(out) == len(problems) * n_total
    return out


def write_draw(samples: list[Sample], path: str | Path) -> Path:
    """Write once; refuse to overwrite (the draw happens exactly once per split, tasks/03 §1)."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{p} exists; the 192-sample draw is made once and never re-drawn")
    write_samples(samples, p)
    from rlordata.train.common import make_read_only

    make_read_only(p)
    return p


def read_draw(
    path: str | Path, *, expect_n: int | None = SAMPLES_PER_PROMPT
) -> dict[str, list[Sample]]:
    """problem_id -> samples in ``sample_idx`` order; checks every problem has ``expect_n``."""
    by_pid: dict[str, list[Sample]] = {}
    for s in read_samples(Path(path)):
        by_pid.setdefault(s.problem_id, []).append(s)
    for pid, ss in by_pid.items():
        ss.sort(key=lambda s: int(s.extra["sample_idx"]))
        if expect_n is not None and len(ss) != expect_n:
            raise ValueError(f"{path}: {pid[:12]} has {len(ss)} samples, expected {expect_n}")
        if [int(s.extra["sample_idx"]) for s in ss] != list(range(len(ss))):
            raise ValueError(f"{path}: sample_idx of {pid[:12]} is not contiguous from 0")
    return by_pid


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_draw(by_pid: dict[str, list[Sample]], *, k: int = TIERING_K) -> dict[str, Any]:
    """Per-split statistics printed after the draw (tasks/03 §1)."""
    all_samples = [s for ss in by_pid.values() for s in ss]
    n = len(all_samples)
    pass8 = Counter(sum(int(s.correct) for s in ss[:k]) for ss in by_pid.values())
    return {
        "n_problems": len(by_pid),
        "n_samples": n,
        "samples_per_problem": n // max(len(by_pid), 1),
        "mean_pass_rate": sum(int(s.correct) for s in all_samples) / max(n, 1),
        "pass8_histogram": {str(i): int(pass8.get(i, 0)) for i in range(k + 1)},
        "truncation_rate": sum(int(s.truncated) for s in all_samples) / max(n, 1),
        "extraction_failure_rate": sum(int(s.extraction_failed) for s in all_samples) / max(n, 1),
        "n_correct": sum(int(s.correct) for s in all_samples),
        "per_tier_pass_rate": {
            t: sum(int(s.correct) for s in all_samples if s.tier == t)
            / max(sum(1 for s in all_samples if s.tier == t), 1)
            for t in sorted({s.tier for s in all_samples})
        },
    }


def format_draw_summary(split: str, summary: dict[str, Any]) -> str:
    hist = " ".join(f"{i}:{c}" for i, c in summary["pass8_histogram"].items())
    tiers = " ".join(f"{t}={r:.3f}" for t, r in summary["per_tier_pass_rate"].items())
    return (
        f"[draw] {split}: {summary['n_problems']} prompts × {summary['samples_per_problem']} = "
        f"{summary['n_samples']} samples; mean pass rate {summary['mean_pass_rate']:.3f} "
        f"({summary['n_correct']} correct; per tier {tiers}); pass8 hist {hist}; "
        f"truncation {100 * summary['truncation_rate']:.2f}%; "
        f"extraction failure {100 * summary['extraction_failure_rate']:.2f}%"
    )
