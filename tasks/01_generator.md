# tasks/01 — Procedural counting-problem generator, tiering, and splits

Owner: agent. Files: `src/rlordata/data/generator.py`, `src/rlordata/data/tiers.py`, `tests/data/test_generator.py`,
`cli` dispatch for `gen` and `tier`. Do not touch `src/rlordata/core/**`.

## What to build

1. `generate_pool(config, seed) -> list[Problem]` per SPEC §4 and `configs/data/pool.yaml`:
   - Pipeline = range + 1–4 filters + 0–3 transforms + 1 final op, sampled from the taxonomy in SPEC §4.
   - `execute_pipeline(pipeline) -> int` computes the answer. Product uses product-mod-m (m in pipeline).
     Mean is integer-rounded (round half to even; document). Median of even-length lists: lower median
     (document). Mode: smallest value among ties (document). Bitwise ops fold over the set.
   - Reject and resample any pipeline whose filtered set is empty before the final op, or whose answer
     magnitude exceeds 10^9.
   - `render(pipeline, rng) -> str` with ≥3 paraphrase templates per operator, chosen by `rng`. Rendering
     must be unambiguous: an expert human reading the text must be able to compute the same answer.
   - `canonical_id(pipeline)` = sha256 over `json.dumps(pipeline, sort_keys=True, separators=(",", ":"))`.
   - Determinism: `random.Random(seed)` / `numpy.random.default_rng(seed)` only; no global RNG.
2. `tiers.py`: `tier_from_pass8`, `build_splits` per SPEC §6 and `configs/data/tiering.yaml`. The pass@8
   sampling itself calls `rlordata.sampling.vllm_sampler` (tasks/02) — implement `tier` CLI so it can run
   once the sampler exists; unit-test `build_splits` with synthetic pass8 values now.
   Disjointness: by `problem_id` and by `structure_id = canonical_id(pipeline without "range")`.
3. `write_jsonl/read_jsonl` round-trip for `Problem`.
4. CLI: `rlordata gen --config configs/data/pool.yaml [--seed N] [--dry-run]` prints pool statistics
   (per range_scale, per total_steps, per final op) and writes the JSONL.

## Acceptance criteria

- All 7 required tests in `tests/data/test_generator.py` implemented and passing; the independent
  reference executor in the test file is written from the SPEC text, not by importing generator code.
- `make gen` produces 6,000 problems in < 60 s on the Mac; stats printed; JSONL round-trips.
- 20 randomly sampled problems attached in the PR description with their answers; Laksh will hand-solve 5.
- `make lint` clean.

## Do not

- Do not add "easy mode" prompts, hints, or few-shot examples. The prompt template is locked (SPEC §5).
- Do not define tiers by structural difficulty; tiers come from base-model pass@8 only (SPEC §6).
- Do not implement any reward or answer-extraction logic; that is `core/verify.py`.
