# tasks/01b — Generator revision after review (SPEC v1.2)

Owner: agent. Same files as tasks/01. Do not touch `core/**`. SPEC §4 changed (v1.2); re-read it and `configs/data/pool.yaml` before starting.

## What changed and why
Review of the first 20 sample problems found: no-op steps ("keep positives" on [58,149], "below 35" on [25,34], "discard multiples of 6" after a prime filter, the same filter twice in a row) inflating `total_steps` without adding difficulty — which would corrupt the complexity-extrapolation axis; tiny ranges ([45,46], [48,49]) producing single-element sets so max/median/mode are trivial; `mode` on all-unique sets collapsing to `min`; and rejection skewing the pool toward 2-step problems.

## Implement
1. **Range bands** from `configs/data/pool.yaml`: scale = span `hi − lo`, disjoint bands, `lo ≥ lo_min`. Sample the span uniformly within the band, then `lo` uniformly so that `hi = lo + span` — do not sample lo and hi independently.
2. **Every step must do work.** Reject a pipeline if any filter or transform leaves the multiset unchanged, if two consecutive filters are identical, if fewer than `min_values_at_final_op` values reach the final op, if final op is `mode` and no value repeats, or if final op is `unique_count` and no value repeats. Remove `positive`/`negative` from the main-pool taxonomy (keep the code path; disabled via `excluded_filters`).
3. **Stratified generation.** Target `n_problems / (|scales| × |total_steps values|)` per cell; keep sampling until every cell is full; print the cell table. Same for `ood.yaml` over steps {6,7,8}.
4. **Connectives by position.** First step → "First", middle → "Then"/"Next" (varied), last op → "Finally"/"Of these numbers". Paraphrase only the operation phrase.
5. **Tests.** Add: (a) no generated pipeline contains a no-op step (re-execute step by step in the independent executor and assert every step changes the multiset); (b) every final op sees ≥ 3 values; (c) cell counts are exactly equal; (d) spans lie in their bands; (e) connective order is monotone (First … Finally). Keep all 7 original tests. **List every skipped test by name and reason** — agent-owned tests must not skip except the live-tiering test that depends on tasks/02.
6. Regenerate the pool (`make gen`) and attach 20 random problems (seed 20260908) with answers and per-step intermediate set sizes.

## Acceptance
- All tests pass; skipped tests listed and justified; lint clean.
- Cell table shows equal counts; no problem in the 20-sample has a no-op step or a set of size < 3 at the final op.
- Rejection rate per cell printed (so we know how hard the generator is working for 5-step problems).
