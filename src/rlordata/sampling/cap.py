"""The token cap (SPEC §7): loading, enforcement, and the Phase-1 computation. AGENT-OWNED; tasks/02a.

Rule (locked, SPEC §7): ``max_completion_tokens = ceil_to_256(1.25 × p99 length of *correct*
base-model completions at T=1.0)``, minimum 512. Computed once by ``scripts/compute_cap.py``
into ``configs/locked/cap.yaml`` and then identical for every arm, control, reference model,
training and evaluation. This module never writes to the locked path on its own; the script
does, and refuses to overwrite.

Enforcement: :func:`resolve_cap` is called by every sampler constructor. Once ``cap.yaml``
exists, any other value raises :class:`CapError`. Before it exists, construction is allowed only
with ``allow_provisional=True`` (the provisional cap run uses :data:`PROVISIONAL_CAP`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CAP_PATH = Path("configs/locked/cap.yaml")
PROVISIONAL_CAP = 4096  # SPEC §7 / tasks/02: the pre-cap sampling run
CAP_MULTIPLIER = 1.25
CAP_ROUND_TO = 256
CAP_MINIMUM = 512
CAP_PERCENTILE = 99.0
MAX_CORRECT_TRUNCATED_FRAC = (
    0.01  # tasks/02: correct completions truncated at the cap must be < 1 %
)


class CapError(RuntimeError):
    """Raised when a run would use a cap different from ``configs/locked/cap.yaml``."""


def load_cap_yaml(path: str | Path = DEFAULT_CAP_PATH) -> dict[str, Any] | None:
    """Parsed ``cap.yaml`` or None if the file does not exist yet."""
    p = Path(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "max_completion_tokens" not in data:
        raise CapError(f"{p} exists but has no max_completion_tokens field")
    return data


def load_locked_cap(path: str | Path = DEFAULT_CAP_PATH) -> int | None:
    """``max_completion_tokens`` from ``cap.yaml`` or None if the cap is not locked yet."""
    data = load_cap_yaml(path)
    return None if data is None else int(data["max_completion_tokens"])


def resolve_cap(
    requested: int | None,
    *,
    cap_path: str | Path = DEFAULT_CAP_PATH,
    allow_provisional: bool = False,
) -> int:
    """Return the cap a run must use, enforcing SPEC §7.

    - cap.yaml exists: ``requested`` must be None or equal to it; anything else raises.
    - cap.yaml missing: allowed only with ``allow_provisional`` and an explicit ``requested``.
    """
    locked = load_locked_cap(cap_path)
    if locked is not None:
        if requested is not None and int(requested) != locked:
            raise CapError(
                f"max_completion_tokens={requested} differs from the locked cap {locked} in {cap_path}. "
                "The cap is a protocol constant (SPEC §7); refusing to run."
            )
        return locked
    if not allow_provisional:
        raise CapError(
            f"{cap_path} does not exist. Only the provisional cap run (allow_provisional_cap=True, "
            f"cap {PROVISIONAL_CAP}) may sample before scripts/compute_cap.py has written it."
        )
    if requested is None:
        raise CapError("no locked cap and no requested cap; pass max_completion_tokens explicitly")
    return int(requested)


def ceil_to_multiple(x: float, multiple: int) -> int:
    assert multiple > 0
    return int(math.ceil(x / multiple) * multiple)


def cap_from_correct_lengths(
    correct_lengths: Sequence[int],
    *,
    percentile: float = CAP_PERCENTILE,
    multiplier: float = CAP_MULTIPLIER,
    round_to: int = CAP_ROUND_TO,
    minimum: int = CAP_MINIMUM,
) -> tuple[int, float]:
    """``(cap, p99)`` from the lengths of correct completions (SPEC §7 formula)."""
    lengths = np.asarray(list(correct_lengths), dtype=np.float64)  # [N]
    assert lengths.ndim == 1
    if lengths.size == 0:
        raise CapError("no correct completions; cannot compute a cap")
    p99 = float(np.percentile(lengths, percentile))
    cap = max(minimum, ceil_to_multiple(multiplier * p99, round_to))
    return cap, p99


@dataclass(frozen=True)
class CapResult:
    max_completion_tokens: int
    p99_correct_len: float
    n_correct_used: int
    n_samples_total: int
    n_problems: int
    frac_correct_truncated_at_cap: float  # correct completions with n_tokens > cap
    frac_all_truncated_at_cap: float  # all completions with n_tokens > cap
    provisional_truncation_rate: float  # truncated flag under the provisional cap
    mean_correct_len: float
    max_correct_len: int
    mean_incorrect_len: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_cap(
    correct_lengths: Sequence[int],
    incorrect_lengths: Sequence[int],
    *,
    n_problems: int,
    provisional_truncated: Sequence[bool],
) -> CapResult:
    """Apply the SPEC §7 rule and compute the diagnostics the script prints."""
    correct = np.asarray(list(correct_lengths), dtype=np.int64)  # [Nc]
    incorrect = np.asarray(list(incorrect_lengths), dtype=np.int64)  # [Ni]
    trunc = np.asarray(list(provisional_truncated), dtype=bool)  # [Nc + Ni]
    assert correct.ndim == 1 and incorrect.ndim == 1 and trunc.ndim == 1
    assert trunc.size == correct.size + incorrect.size
    cap, p99 = cap_from_correct_lengths(correct.tolist())
    all_lengths = np.concatenate([correct, incorrect])
    return CapResult(
        max_completion_tokens=cap,
        p99_correct_len=p99,
        n_correct_used=int(correct.size),
        n_samples_total=int(all_lengths.size),
        n_problems=int(n_problems),
        frac_correct_truncated_at_cap=float(np.mean(correct > cap)),
        frac_all_truncated_at_cap=float(np.mean(all_lengths > cap)) if all_lengths.size else 0.0,
        provisional_truncation_rate=float(trunc.mean()) if trunc.size else 0.0,
        mean_correct_len=float(correct.mean()),
        max_correct_len=int(correct.max()),
        mean_incorrect_len=float(incorrect.mean()) if incorrect.size else float("nan"),
    )


def length_histogram(
    correct_lengths: Sequence[int],
    incorrect_lengths: Sequence[int],
    *,
    bin_width: int = 128,
    max_len: int | None = None,
    bar_width: int = 40,
) -> str:
    """Text histogram, correct vs incorrect, one row per ``bin_width`` tokens."""
    correct = np.asarray(list(correct_lengths), dtype=np.int64)
    incorrect = np.asarray(list(incorrect_lengths), dtype=np.int64)
    hi = (
        max_len
        if max_len is not None
        else int(max(correct.max(initial=0), incorrect.max(initial=0)))
    )
    n_bins = max(1, ceil_to_multiple(hi + 1, bin_width) // bin_width)
    edges = np.arange(0, (n_bins + 1) * bin_width, bin_width)
    hc, _ = np.histogram(correct, bins=edges)
    hi_, _ = np.histogram(incorrect, bins=edges)
    scale = max(1, int(max(hc.max(initial=0), hi_.max(initial=0))))
    rows = [f"{'tokens':>13}  {'correct':>8}  {'incorrect':>9}  (bar = correct, '.' = incorrect)"]
    for i in range(n_bins):
        lo, hi_edge = int(edges[i]), int(edges[i + 1]) - 1
        bar_c = "#" * int(round(bar_width * hc[i] / scale))
        bar_i = "." * int(round(bar_width * hi_[i] / scale))
        rows.append(f"{lo:>6}-{hi_edge:<6}  {int(hc[i]):>8}  {int(hi_[i]):>9}  {bar_c}{bar_i}")
    return "\n".join(rows)


def write_cap_yaml(path: str | Path, record: dict[str, Any]) -> Path:
    """Write ``record`` as YAML. Refuses to overwrite: the cap is written exactly once."""
    p = Path(path)
    if p.exists():
        raise CapError(
            f"{p} already exists; refusing to overwrite a locked protocol constant. "
            "If the cap genuinely must change, that is a SPEC §12 amendment."
        )
    if "max_completion_tokens" not in record:
        raise CapError("cap record lacks max_completion_tokens")
    p.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# locked (SPEC §7) — written once by scripts/compute_cap.py. Never edit by hand.\n"
        "# cap = max(512, ceil_to_256(1.25 * p99 length of correct base-model completions at T=1.0))\n"
    )
    with p.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(record, f, sort_keys=True, default_flow_style=False)
    return p
