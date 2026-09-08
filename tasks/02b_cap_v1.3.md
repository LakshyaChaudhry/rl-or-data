# tasks/02b — Cap rule v1.3 and per-tier truncation reporting (after tasks/02a, before any GPU run)

Owner: agent. Prerequisite: SPEC §7 has been amended to v1.3 by Laksh (check the changelog in §12
before starting; if the v1.3 entry is not there, stop and say so). Read `CLAUDE.md`, SPEC §7 and
§10, and `tasks/02a_local_half.md` first. Do not touch `src/rlordata/core/**` or
`configs/locked/**`; `configs/locked/cap.yaml` is still written only by `scripts/compute_cap.py`
on the GPU box.

## Why

Two problems with the v1.2 cap rule that Laksh and the agent identified on 2026-09-07:

1. It calibrated on *all* correct base-model completions on val. The base model mostly gets easy
   problems right, so the length distribution was dominated by easy problems and could sit below
   what correct reasoning on 5-step problems needs. A truncated completion has no answer line
   and scores as wrong, so a low cap converts would-be-correct hard answers into errors, hurting
   the arms that learn hard problems and starving GRPO of reward on hard prompts (the failure
   mode Bauer et al. report on their graph task).
2. It had no relation to Bauer et al.'s fixed 2048-token completion cap (Table 1), which the
   study extends.

## The v1.3 rule (SPEC §7; this is the contract, restated)

```
cell        = (range_scale, total_steps)                      # structural knobs from the pool
p99_cell    = p99 of n_tokens over CORRECT base-model completions in that cell (val, T=1.0)
measured    = ceil_to_256(1.25 × max over cells of p99_cell)
cap         = max(2048, measured)
max prompt tokens = 4096
```

Exactly one term binds: if `measured <= 2048` the floor binds and the cap equals the anchor
paper's; otherwise the measurement binds and the raise is documented. The cap is computed once,
identical for every arm / control / reference model, training and eval.

## Change

1. **`scripts/compute_cap.py`**
   - Join each sample to its problem by `problem_id` against `data/pool/pool.jsonl` to get
     `range_scale` and `total_steps` (also have the eval runner store both in `Sample.extra`;
     the join is the source of truth).
   - Per-cell table over correct completions: n_correct, p50, p99, and the fraction that would be
     truncated at 2048 and at the chosen cap. Print it.
   - Print which cell set the cap, which term bound (`floor` or `measured`), and the overall
     fraction of correct completions above 2048 (the anchor-comparison number for the write-up).
   - The correct-truncation check at the chosen cap (< 1 %) is now evaluated per cell as well as
     overall; fail loudly if any cell exceeds 1 %.
   - `cap.yaml` provenance gains: `anchor_cap: 2048`, `binding_term`, `measured_cap`,
     `per_cell_p99` (dict), `driving_cell`, `frac_correct_over_2048`. Keep every existing field.
     Still refuses to overwrite.
   - Drop the 512 floor (moot under the 2048 floor) and remove any reference to it.
2. **`sampling/vllm_sampler.py`**: max prompt tokens 1024 → 4096; `max_model_len = 4096 + cap`.
   The cap-identity check against `cap.yaml` is unchanged.
3. **Per-tier and per-split truncation everywhere.** `metrics.json` and the printed summary table
   report truncation rate and extraction-failure rate per tier for every split, and the fraction
   of completions that hit the cap exactly. `analysis/sanity.py`: keep the > 5 % flag on
   `test_300`; for `ood_hard_200` report per-tier truncation prominently but do not flag, since
   high truncation there is a finding ("ran out of tokens" vs "extrapolation failed") that the
   write-up must state, not hide.
4. **Tests** (agent-owned, `tests/`): synthetic samples spanning several cells; (a) the floor
   binds when the longest cell's 1.25 × p99 ≤ 2048; (b) the measurement binds otherwise and the
   result is a multiple of 256; (c) the driving cell is the one with the largest p99, not the
   largest n; (d) a cell over 1 % correct-truncation at the chosen cap fails loudly;
   (e) provenance fields present; (f) overwrite refused; (g) per-tier truncation in metrics.
5. Update the docstrings in `compute_cap.py` and the sampler, `tasks/02a_local_half.md` item 3
   (point it here), and the command sequence for the GPU half in your report.

## Acceptance

`make test` and `make lint` clean; `make eval-dry` still produces every file; the cap script's
printed table shows the per-cell p99s, the driving cell, and the binding term on synthetic data;
report lists every skipped / `gpu`-marked test by name.

## Do not

Do not apply the SPEC amendment yourself; Laksh applies it. Do not change the 1.25 multiplier,
the 256 rounding, the 2048 floor, or the < 1 % rule. Do not run anything on a GPU.
