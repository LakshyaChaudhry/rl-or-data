"""Transfer evaluation sets (SPEC §6.3, tasks/02 §5): Reasoning Gym candidates and GSM8K-500.

Code only in tasks/02a; the base-model pick (20–60 % accuracy) happens on the GPU box.
Instances are rendered as ``types.Problem`` so the eval runner formats them with the locked
TEMPLATE and scores them with the same answer regex. Counting-specific fields are placeholders
(``pipeline["source"]`` marks the origin; ``total_steps=0``).

Optional dependencies (``tests/data/test_transfer.py`` skips cleanly without them):
  - ``reasoning-gym``   (``uv pip install -e ".[transfer]"``)   → ``load_reasoning_gym``
  - ``datasets``        (``ml`` extra; needs network once)        → ``load_gsm8k_test``

Incompatibility found while wiring (2026-09-07): Reasoning Gym ``number_filtering`` answers are
Python lists of decimal strings (e.g. ``"['12.5', '-3.00']"``), not integers, so they cannot be
scored by ``^Answer:\\s*(-?\\d+)\\s*$``. ``load_reasoning_gym("number_filtering")`` raises
:class:`TransferTaskIncompatibleError`; the pick is between ``basic_arithmetic`` and ``count_primes``
unless SPEC §6.3 is amended.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from typing import Any

from rlordata.types import Problem

RG_CANDIDATES = ("basic_arithmetic", "number_filtering", "count_primes")
RG_N = 300
RG_SEED = 20260903  # distinct from the pool (20260901) and ood (20260902) seeds
GSM8K_N = 500
GSM8K_DATASET = ("openai/gsm8k", "main", "test")

_INT_RE = re.compile(r"^-?\d+$")
_GSM8K_FINAL = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


class TransferTaskIncompatibleError(ValueError):
    """The task's answers are not integers and cannot be scored with the locked regex."""


def reasoning_gym_available() -> bool:
    return importlib.util.find_spec("reasoning_gym") is not None


def datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def parse_int_answer(answer: Any) -> int | None:
    """``"27"`` -> 27, ``"-5"`` -> -5, ``"1,000"`` -> 1000; lists / decimals -> None."""
    if isinstance(answer, bool):
        return None
    if isinstance(answer, int):
        return int(answer)
    text = str(answer).strip().replace(",", "")
    if _INT_RE.match(text):
        return int(text)
    return None


def parse_gsm8k_answer(answer_field: str) -> int:
    """Final integer after ``####`` in a GSM8K ``answer`` field (commas stripped)."""
    m = _GSM8K_FINAL.search(answer_field)
    if not m:
        raise ValueError(f"no '#### <answer>' in GSM8K answer field: {answer_field[-80:]!r}")
    value = parse_int_answer(m.group(1))
    if value is None:
        raise ValueError(f"GSM8K final answer is not an integer: {m.group(1)!r}")
    return value


def _transfer_problem(
    *, source: str, task: str, index: int, seed: int | None, question: str, answer: int, split: str
) -> Problem:
    key = f"{source}:{task}:{seed}:{index}:{question}"
    return Problem(
        problem_id=hashlib.sha256(key.encode("utf-8")).hexdigest(),
        text=question,
        answer=int(answer),
        pipeline={"source": source, "task": task, "seed": seed, "index": index},
        range_scale="S",  # placeholder: transfer sets have no range scale
        n_filters=0,
        n_transforms=0,
        total_steps=0,
        tier="untiered",
        pass8=None,
        split=split,
    )


def rg_split_name(task: str, n: int = RG_N) -> str:
    return f"rg_{task}_{n}"


def load_reasoning_gym(task: str, n: int = RG_N, seed: int = RG_SEED) -> list[Problem]:
    """``n`` instances of a Reasoning Gym task at a fixed seed, as integer-answer Problems."""
    if task not in RG_CANDIDATES:
        raise ValueError(
            f"unknown Reasoning Gym candidate {task!r}; SPEC §6.3 lists {RG_CANDIDATES}"
        )
    if task == "number_filtering":
        raise TransferTaskIncompatibleError(
            "reasoning_gym 'number_filtering' answers are lists of decimal strings, not integers; "
            "they cannot be scored with the locked answer regex (SPEC §5). Not a valid candidate."
        )
    try:
        import reasoning_gym  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "reasoning-gym is not installed: uv pip install -e '.[transfer]'"
        ) from exc
    dataset = reasoning_gym.create_dataset(task, size=n, seed=seed)
    problems: list[Problem] = []
    for i in range(n):
        item = dataset[i]
        answer = parse_int_answer(item["answer"])
        if answer is None:
            raise TransferTaskIncompatibleError(
                f"{task}[{i}] answer {item['answer']!r} is not an integer"
            )
        problems.append(
            _transfer_problem(
                source="reasoning_gym",
                task=task,
                index=i,
                seed=seed,
                question=str(item["question"]).strip(),
                answer=answer,
                split=rg_split_name(task, n),
            )
        )
    return problems


def gsm8k_problems(rows: list[dict[str, str]], n: int = GSM8K_N) -> list[Problem]:
    """First ``n`` GSM8K rows (``question``, ``answer``) by index as Problems."""
    if len(rows) < n:
        raise ValueError(f"need {n} GSM8K rows, got {len(rows)}")
    return [
        _transfer_problem(
            source="gsm8k",
            task="test",
            index=i,
            seed=None,
            question=str(row["question"]).strip(),
            answer=parse_gsm8k_answer(str(row["answer"])),
            split=f"gsm8k_{n}",
        )
        for i, row in enumerate(rows[:n])
    ]


def load_gsm8k_test(n: int = GSM8K_N) -> list[Problem]:
    """GSM8K ``main`` test split, first ``n`` by index (needs ``datasets`` and network/cache)."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("datasets is not installed (ml extra)") from exc
    name, config, split = GSM8K_DATASET
    ds = load_dataset(name, config, split=split)
    rows = [{"question": r["question"], "answer": r["answer"]} for r in ds.select(range(n))]
    return gsm8k_problems(rows, n=n)
