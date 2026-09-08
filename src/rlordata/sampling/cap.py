"""The token cap (SPEC §7 v1.3): loading, enforcement, and the Phase-1 computation. AGENT-OWNED; tasks/02a+02b.

Rule (locked, SPEC §7 v1.3)::

    cell      = (range_scale, total_steps)                      # pool cells, e.g. "S2" .. "M5"
    p99_cell  = p99 of n_tokens over CORRECT base-model completions in that cell (T=1.0)
    measured  = ceil_to_256(1.25 × max over cells of p99_cell)
    cap       = max(2048, measured)                             # 2048 = Bauer et al. Table 1

Exactly one term binds: ``floor`` (the anchor paper's cap) or ``measured`` (a documented raise).
The cap is computed once by ``scripts/compute_cap.py`` into ``configs/locked/cap.yaml`` and is then
identical for every arm, control, reference model, training and evaluation. Correct completions
truncated at the chosen cap must stay below 1 % overall and in every cell.

Enforcement: :func:`resolve_cap` is called by every sampler constructor. Once ``cap.yaml`` exists,
any other value raises :class:`CapError`. Before it exists, construction is allowed only with
``allow_provisional=True`` (the provisional cap run uses :data:`PROVISIONAL_CAP`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CAP_PATH = Path("configs/locked/cap.yaml")
PROVISIONAL_CAP = 4096  # SPEC §7 / tasks/02: the pre-cap sampling run
ANCHOR_CAP = 2048  # SPEC §7 v1.3 floor = Bauer et al. Table 1
CAP_MULTIPLIER = 1.25
CAP_ROUND_TO = 256
CAP_PERCENTILE = 99.0
MAX_CORRECT_TRUNCATED_FRAC = (
    0.01  # correct completions truncated at the cap must be < 1 % (overall and per cell)
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


def cell_key(range_scale: str, total_steps: int) -> str:
    """``("S", 3)`` -> ``"S3"`` (the pool's ``range_scale × total_steps`` cell)."""
    return f"{range_scale}{int(total_steps)}"


def measured_cap_from_lengths(
    correct_lengths: Sequence[int],
    *,
    percentile: float = CAP_PERCENTILE,
    multiplier: float = CAP_MULTIPLIER,
    round_to: int = CAP_ROUND_TO,
) -> tuple[int, float]:
    """``(ceil_to_256(1.25 × p99), p99)`` for one population of correct-completion lengths. No floor."""
    lengths = np.asarray(list(correct_lengths), dtype=np.float64)  # [N]
    assert lengths.ndim == 1
    if lengths.size == 0:
        raise CapError("no correct completions; cannot compute a cap")
    p99 = float(np.percentile(lengths, percentile))
    return ceil_to_multiple(multiplier * p99, round_to), p99


@dataclass(frozen=True)
class CellStats:
    cell: str
    n_correct: int
    p50: float
    p99: float
    measured_cap: int  # ceil_to_256(1.25 × p99) for this cell alone
    frac_correct_over_anchor: float  # correct completions with n_tokens > 2048
    frac_correct_over_cap: float  # correct completions with n_tokens > the chosen cap

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapResult:
    max_completion_tokens: int
    anchor_cap: int
    measured_cap: int  # max over cells of the per-cell measured cap
    binding_term: str  # "floor" | "measured"
    driving_cell: str  # cell with the largest p99
    p99_max: float
    per_cell: dict[str, CellStats]
    n_correct_used: int
    n_samples_total: int
    n_problems: int
    frac_correct_over_anchor: float  # overall: correct completions above 2048 (anchor comparison)
    frac_correct_truncated_at_cap: float  # overall: correct completions with n_tokens > cap
    frac_all_truncated_at_cap: float  # all completions with n_tokens > cap
    provisional_truncation_rate: float  # truncated flag under the provisional cap
    mean_correct_len: float
    max_correct_len: int
    mean_incorrect_len: float
    cells_over_limit: tuple[str, ...]  # cells with >= 1 % correct completions above the cap

    @property
    def per_cell_p99(self) -> dict[str, float]:
        return {k: v.p99 for k, v in self.per_cell.items()}

    def violations(self, limit: float = MAX_CORRECT_TRUNCATED_FRAC) -> list[str]:
        """Human-readable reasons the cap must not be written (empty = OK)."""
        out: list[str] = []
        if self.frac_correct_truncated_at_cap >= limit:
            out.append(
                f"overall: {self.frac_correct_truncated_at_cap:.3%} of correct completions exceed the cap "
                f"{self.max_completion_tokens} (limit {limit:.0%})"
            )
        for cell in self.cells_over_limit:
            out.append(
                f"cell {cell}: {self.per_cell[cell].frac_correct_over_cap:.3%} of correct completions exceed "
                f"the cap {self.max_completion_tokens} (limit {limit:.0%})"
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_cell"] = {k: v.to_dict() for k, v in self.per_cell.items()}
        d["cells_over_limit"] = list(self.cells_over_limit)
        return d


def compute_cap(
    correct_by_cell: Mapping[str, Sequence[int]],
    incorrect_lengths: Sequence[int],
    *,
    n_problems: int,
    provisional_truncated: Sequence[bool],
    anchor_cap: int = ANCHOR_CAP,
    limit: float = MAX_CORRECT_TRUNCATED_FRAC,
) -> CapResult:
    """Apply the SPEC §7 v1.3 rule to correct-completion lengths grouped by pool cell."""
    cells = {
        k: np.asarray(list(v), dtype=np.int64) for k, v in correct_by_cell.items() if len(v) > 0
    }
    if not cells:
        raise CapError("no correct completions in any cell; cannot compute a cap")
    incorrect = np.asarray(list(incorrect_lengths), dtype=np.int64)  # [Ni]
    trunc = np.asarray(list(provisional_truncated), dtype=bool)  # [Nc + Ni]
    correct_all = np.concatenate(list(cells.values()))  # [Nc]
    assert incorrect.ndim == 1 and trunc.ndim == 1 and correct_all.ndim == 1
    assert trunc.size == correct_all.size + incorrect.size, (
        "provisional_truncated must cover every sample"
    )

    per_cell_measured: dict[str, tuple[int, float]] = {
        k: measured_cap_from_lengths(v.tolist()) for k, v in cells.items()
    }
    # Driving cell = largest p99 (ties -> first in sorted cell order), never the largest n.
    driving_cell = sorted(per_cell_measured, key=lambda k: (-per_cell_measured[k][1], k))[0]
    measured_cap, p99_max = per_cell_measured[driving_cell]
    cap = max(anchor_cap, measured_cap)
    binding = "floor" if cap == anchor_cap and measured_cap <= anchor_cap else "measured"

    stats: dict[str, CellStats] = {}
    for k in sorted(cells):
        v = cells[k]
        stats[k] = CellStats(
            cell=k,
            n_correct=int(v.size),
            p50=float(np.percentile(v, 50)),
            p99=per_cell_measured[k][1],
            measured_cap=per_cell_measured[k][0],
            frac_correct_over_anchor=float(np.mean(v > anchor_cap)),
            frac_correct_over_cap=float(np.mean(v > cap)),
        )
    over = tuple(k for k, s in stats.items() if s.frac_correct_over_cap >= limit)
    all_lengths = np.concatenate([correct_all, incorrect])
    return CapResult(
        max_completion_tokens=cap,
        anchor_cap=anchor_cap,
        measured_cap=measured_cap,
        binding_term=binding,
        driving_cell=driving_cell,
        p99_max=p99_max,
        per_cell=stats,
        n_correct_used=int(correct_all.size),
        n_samples_total=int(all_lengths.size),
        n_problems=int(n_problems),
        frac_correct_over_anchor=float(np.mean(correct_all > anchor_cap)),
        frac_correct_truncated_at_cap=float(np.mean(correct_all > cap)),
        frac_all_truncated_at_cap=float(np.mean(all_lengths > cap)),
        provisional_truncation_rate=float(trunc.mean()) if trunc.size else 0.0,
        mean_correct_len=float(correct_all.mean()),
        max_correct_len=int(correct_all.max()),
        mean_incorrect_len=float(incorrect.mean()) if incorrect.size else float("nan"),
        cells_over_limit=over,
    )


def per_cell_table(result: CapResult) -> str:
    """Printable per-cell table: n_correct, p50, p99, measured cap, % over 2048, % over the chosen cap."""
    cap = result.max_completion_tokens
    head = (
        f"{'cell':<6}{'n_correct':>10}{'p50':>9}{'p99':>9}{'1.25*p99->256':>15}"
        f"{'>' + str(result.anchor_cap) + ' %':>10}{'>cap ' + str(cap) + ' %':>13}  note"
    )
    rows = [head, "-" * len(head)]
    for k, s in result.per_cell.items():
        note = []
        if k == result.driving_cell:
            note.append("driving cell (largest p99)")
        if k in result.cells_over_limit:
            note.append("OVER 1% LIMIT")
        rows.append(
            f"{k:<6}{s.n_correct:>10}{s.p50:>9.1f}{s.p99:>9.1f}{s.measured_cap:>15}"
            f"{100 * s.frac_correct_over_anchor:>10.2f}{100 * s.frac_correct_over_cap:>13.2f}  {'; '.join(note)}"
        )
    rows.append(
        f"cap = max({result.anchor_cap}, {result.measured_cap}) = {cap}  binding term: {result.binding_term}"
        f"  (p99_max {result.p99_max:.1f} in cell {result.driving_cell})"
    )
    rows.append(
        f"correct completions over {result.anchor_cap} (anchor comparison): "
        f"{100 * result.frac_correct_over_anchor:.3f}%   over cap {cap}: {100 * result.frac_correct_truncated_at_cap:.3f}%"
    )
    return "\n".join(rows)


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
        "# locked (SPEC §7 v1.3) — written once by scripts/compute_cap.py. Never edit by hand.\n"
        "# cap = max(2048, ceil_to_256(1.25 * max over (range_scale x total_steps) cells of the p99 length\n"
        "#       of correct base-model completions at T=1.0)); binding_term says which term bound.\n"
    )
    with p.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(record, f, sort_keys=True, default_flow_style=False)
    return p
