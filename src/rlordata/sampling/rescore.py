"""Re-verify stored completions offline (extractor or cap changed; no resampling). AGENT-OWNED.

Every completion is stored, sampling is bitwise reproducible (batch-invariant vLLM, per-sample
seeds), and a completion's first ``cap`` tokens do not depend on ``max_tokens``. So when the answer
extractor (``core/verify.py``) or the locked cap changes after sampling, the stored samples can be
rescored instead of resampled:

- ``correct`` / ``reward`` / ``extracted_answer`` / ``extraction_failed`` are recomputed with the
  *current* ``core.verify`` against the pool's answer (joined by ``problem_id``; the stored
  ``extra["answer"]`` is the fallback).
- If ``cap`` is given and is below the cap the samples were generated with, completions longer than
  ``cap`` are treated as truncated at ``cap``: ``n_tokens = cap``, ``truncated = True``, unscored.
  This is conservative (a completion that had its answer line before the cap and rambled on would
  also be counted as truncated); the fraction affected is recorded in ``extra["rescore"]``.

The original file is kept next to the rescored one as ``<name>.raw.jsonl`` (written once).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from rlordata.core.verify import EXTRACTION_RULE, verify
from rlordata.run_dir import git_sha, now_iso
from rlordata.sampling.eval_runner import metrics_for, read_samples, write_samples
from rlordata.types import Problem, Sample


def _answer_problem(problem_id: str, answer: int) -> Problem:
    """Minimal Problem carrying the answer; ``verify`` reads nothing else."""
    return Problem(
        problem_id=problem_id,
        text="",
        answer=int(answer),
        pipeline={},
        range_scale="S",
        n_filters=0,
        n_transforms=0,
        total_steps=0,
    )


def rescore_samples(
    samples: list[Sample],
    *,
    answers: Mapping[str, int] | None = None,
    cap: int | None = None,
) -> tuple[list[Sample], dict[str, Any]]:
    """Recompute verdicts (and simulate ``cap``) for stored samples. Returns ``(samples, summary)``."""
    out: list[Sample] = []
    n_changed = 0
    n_cap_truncated = 0
    n_missing_answer = 0
    for s in samples:
        answer = (answers or {}).get(s.problem_id, s.extra.get("answer") if s.extra else None)
        if answer is None:
            n_missing_answer += 1
            answer = s.extra.get("answer") if s.extra else None
        if answer is None:
            raise ValueError(
                f"no answer for problem {s.problem_id[:12]} (pass a pool or store extra.answer)"
            )
        truncated = bool(s.truncated)
        n_tokens = int(s.n_tokens)
        if cap is not None and truncated and n_tokens < cap:
            # Sampled at a LOWER cap than the one requested: the tokens between the two caps were
            # never generated, so rescoring cannot raise the cap. Resample at the locked cap instead.
            raise ValueError(
                f"sample {s.problem_id[:12]} was truncated at {n_tokens} tokens, below the requested "
                f"cap {cap}; rescoring can only simulate a lower cap, not raise one — resample"
            )
        cap_hit = False
        if cap is not None and n_tokens > cap:
            truncated, n_tokens, cap_hit = True, int(cap), True
            n_cap_truncated += 1
        if truncated and cap_hit:
            extracted, reward, failed = None, 0.0, True
        else:
            v = verify(
                _answer_problem(s.problem_id, int(answer)), s.completion, truncated=truncated
            )
            extracted, reward, failed = v.extracted, float(v.reward), bool(v.extraction_failed)
        new = replace(
            s,
            extracted_answer=extracted,
            correct=bool(reward == 1.0),
            reward=reward,
            n_tokens=n_tokens,
            truncated=truncated,
            extraction_failed=failed,
            extra={
                **(s.extra or {}),
                "answer": int(answer),
                "rescore": {
                    "cap": cap,
                    "cap_truncated": cap_hit,
                    "original_n_tokens": int(s.n_tokens),
                },
            },
        )
        if (new.correct, new.truncated, new.extraction_failed, new.extracted_answer) != (
            s.correct,
            s.truncated,
            s.extraction_failed,
            s.extracted_answer,
        ):
            n_changed += 1
        out.append(new)
    summary = {
        "n_samples": len(samples),
        "n_verdicts_changed": n_changed,
        "n_cap_truncated": n_cap_truncated,
        "frac_cap_truncated": n_cap_truncated / max(1, len(samples)),
        "n_missing_answer_in_pool": n_missing_answer,
        "cap": cap,
        "accuracy_before": sum(s.correct for s in samples) / max(1, len(samples)),
        "accuracy_after": sum(s.correct for s in out) / max(1, len(samples)),
        "extraction_failure_before": sum(s.extraction_failed for s in samples)
        / max(1, len(samples)),
        "extraction_failure_after": sum(s.extraction_failed for s in out) / max(1, len(samples)),
        "extraction_rule": EXTRACTION_RULE,
        "verify_git_sha": git_sha(),
        "rescored_at": now_iso(),
    }
    return out, summary


def answers_from_pools(pools: list[list[Problem]]) -> dict[str, int]:
    return {p.problem_id: int(p.answer) for pool in pools for p in pool}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rescore_file(
    samples_path: Path,
    *,
    answers: Mapping[str, int] | None,
    cap: int | None,
) -> tuple[list[Sample], dict[str, Any]]:
    """Rescore ``samples_path`` in place, keeping the first-ever original as ``<stem>.raw.jsonl``."""
    samples_path = Path(samples_path)
    raw = samples_path.with_name(samples_path.stem + ".raw.jsonl")
    if not raw.exists():
        raw.write_bytes(samples_path.read_bytes())
    # Always rescore from the raw completions so repeated rescoring does not compound cap simulation.
    samples = read_samples(raw)
    rescored, summary = rescore_samples(samples, answers=answers, cap=cap)
    write_samples(rescored, samples_path)
    summary.update(
        {"samples_path": str(samples_path), "raw_path": str(raw), "raw_sha256": sha256_file(raw)}
    )
    return rescored, summary


def rescore_run_dir(
    run_dir: Path,
    *,
    answers: Mapping[str, int] | None,
    cap: int | None,
) -> dict[str, Any]:
    """Rescore an eval/cap run directory: samples.jsonl, metrics.json, meta.json (provenance)."""
    run_dir = Path(run_dir)
    rescored, summary = rescore_file(run_dir / "samples.jsonl", answers=answers, cap=cap)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    config_path = run_dir / "config.yaml"
    ks = None
    split = meta.get("split")
    seed = int(meta.get("seed", 0) or 0)
    if config_path.exists():
        import yaml

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        ks = (cfg.get("decoding") or {}).get("ks")
        split = cfg.get("split", split)
        seed = int(cfg.get("seed", seed))
    metrics_path = run_dir / "metrics.json"
    old_metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    metrics = metrics_for(
        rescored, ks=ks, seed=seed, cap=cap or old_metrics.get("max_completion_tokens"), split=split
    )
    for key in (
        "run_id",
        "config_hash",
        "model_id",
        "arm",
        "split",
        "subset",
        "decoding",
        "temperature",
        "n",
        "seed",
        "sampler",
    ):
        if key in old_metrics:
            metrics[key] = old_metrics[key]
    metrics["rescore"] = summary
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    history = list(meta.get("rescore_history", []))
    history.append(summary)
    meta["rescore_history"] = history
    meta["rescored_to_cap"] = cap
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return summary
