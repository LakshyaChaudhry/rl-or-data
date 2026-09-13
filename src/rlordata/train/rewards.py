"""Reward functions for GRPO / controls (tasks/04 §3). AGENT-OWNED.

Thin wrappers around ``core.verify`` — never reimplement extraction or correctness.
Every call also appends a record to ``reward_records.jsonl`` so true accuracy is logged
even when the training reward is random (C1) or format-only (C2).
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rlordata.core.verify import has_answer_line, verify
from rlordata.types import Problem

RewardFn = Callable[..., list[float]]


@dataclass
class RewardRecord:
    """One completion's reward + always-on correctness diagnostics."""

    step: int | None
    problem_id: str
    tier: str
    reward: float
    correct: bool
    n_tokens: int | None
    truncated: bool
    extraction_failed: bool
    reward_name: str
    extra: dict[str, Any] = field(default_factory=dict)


class RewardRecorder:
    """Thread-safe append-only JSONL of per-completion reward diagnostics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.n_records = 0
        self._step: int | None = None

    def set_step(self, step: int | None) -> None:
        self._step = step

    def write(self, records: Sequence[RewardRecord]) -> None:
        if not records:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(asdict(r), ensure_ascii=False, default=str) + "\n")
            self.n_records += len(records)


def _as_list(x: Any, n: int) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * n


def _problem_stub(problem_id: str, answer: int, tier: str) -> Problem:
    return Problem(
        problem_id=problem_id,
        text="",
        answer=int(answer),
        pipeline={},
        range_scale="S",
        n_filters=0,
        n_transforms=0,
        total_steps=1,
        tier=tier if tier in ("easy", "medium", "hard", "untiered") else "untiered",  # type: ignore[arg-type]
    )


def _completion_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict) and "content" in c:
        return str(c["content"])
    if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
        return str(c[0]["content"])
    return str(c)


def _n_tokens_and_trunc(
    completions: list[str],
    *,
    completion_ids: Any | None,
    max_completion_length: int | None,
) -> tuple[list[int | None], list[bool]]:
    n_tok: list[int | None] = [None] * len(completions)
    trunc = [False] * len(completions)
    if completion_ids is None:
        return n_tok, trunc
    for i, ids in enumerate(completion_ids):
        if ids is None:
            continue
        n = len(ids)
        n_tok[i] = n
        if max_completion_length is not None and n >= int(max_completion_length):
            trunc[i] = True
    return n_tok, trunc


def _rng_for(seed: int, problem_id: str, step: int | None, i: int) -> np.random.Generator:
    payload = f"{seed}:{problem_id}:{step}:{i}".encode()
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def make_reward_fn(
    name: str,
    *,
    recorder: RewardRecorder,
    seed: int,
    max_completion_length: int | None = None,
) -> RewardFn:
    """Build a TRL-compatible reward callable for ``verify_binary`` / C1 / C2."""
    if name not in ("verify_binary", "random_bernoulli", "format_only", "random_bernoulli_0_5"):
        raise ValueError(f"unknown reward {name!r}")
    if name == "random_bernoulli_0_5":
        name = "random_bernoulli"

    def _fn(
        completions: list[Any],
        problem_id: Any = None,
        answer: Any = None,
        tier: Any = None,
        completion_ids: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        texts = [_completion_text(c) for c in completions]
        n = len(texts)
        pids = [str(x) for x in _as_list(problem_id if problem_id is not None else "unknown", n)]
        answers = _as_list(answer if answer is not None else 0, n)
        tiers = [str(x) for x in _as_list(tier if tier is not None else "untiered", n)]
        n_tok, trunc = _n_tokens_and_trunc(
            texts, completion_ids=completion_ids, max_completion_length=max_completion_length
        )
        step = recorder._step
        rewards: list[float] = []
        records: list[RewardRecord] = []
        for i, text in enumerate(texts):
            problem = _problem_stub(pids[i], int(answers[i]), tiers[i])
            verdict = verify(problem, text, truncated=bool(trunc[i]))
            if name == "verify_binary":
                r = float(verdict.reward)
            elif name == "format_only":
                # C2 (SPEC §8): 1 iff the answer came from an explicit answer line — §5 layers
                # (a)/(b), the same signal reported as answer_line_rate — never the last-integer
                # fallback (c), and never for a truncated completion. Independent of correctness.
                r = 1.0 if has_answer_line(text, truncated=bool(trunc[i])) else 0.0
            else:  # random_bernoulli — independent of content
                r = float(_rng_for(seed, pids[i], step, i).integers(0, 2))
            rewards.append(r)
            records.append(
                RewardRecord(
                    step=step,
                    problem_id=pids[i],
                    tier=tiers[i],
                    reward=r,
                    correct=bool(verdict.reward == 1.0),
                    n_tokens=n_tok[i],
                    truncated=bool(trunc[i]),
                    extraction_failed=bool(verdict.extraction_failed),
                    reward_name=name,
                )
            )
        recorder.write(records)
        return rewards

    _fn.__name__ = name
    return _fn


def load_reward_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
