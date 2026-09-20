# Phase 4 results — analysis, seed aggregation, grading sheet (tasks/05), SPEC v1.8

A committed snapshot of `outputs/` (which is gitignored) as written by `make analysis` at 9198a23 on
2026-09-19 from the local mirror of the artifact store, `runs/store_mirror/runs` (checksum-verified
against the store at the Phase 2/3 close-out). Analysis config `configs/analysis/default.yaml`, hash
`d9af2a8c5fb5`; bootstrap seed 0, 10,000 resamples. No GPU was used. No file here exceeds 1 MB.

Everything except `analysis_meta.json` (git SHA, package versions, wall-clock) is byte-reproducible:
`make analysis OUT=/tmp/x && diff -r -x .cache -x analysis_meta.json /tmp/x results/phase4`.

## What is here

- `hypotheses.md` — the grading sheet: H1–H3, the two pre-registered predictions and the controls, each
  with the numbers, the SPEC §10 criterion evaluated mechanically, the truncation next to it, and a
  **blank verdict line for Laksh**. Grade a copy; regenerating overwrites `outputs/hypotheses.md`.
- `tables/results.md` (+ `units.csv`, `arms.csv`, `per_tier.csv`, `budgets.csv`, `truncation.csv`) — per
  arm, every seed as a row, mean ± std; base, controls and reference models; per-tier test accuracy;
  truncation by arm/split/decoding; the three SPEC §8 budgets and GPU-hours.
- `tables/contrasts.md` (+ `contrasts.csv`) — seed-wise paired contrasts with truncation beside every Δ,
  and the H2 ratio.
- `how_could_this_be_wrong.md` — auto-generated per headline contrast.
- `sanity.md` / `sanity.json` — the 13 cross-run checks (all pass) and their notes.
- `figures/` — (a)–(h) of tasks/05 item 4 plus (i) eval-time length and truncation by arm.
- `results_packet.md` (300 lines, tables only, `hypotheses.md` verbatim, no verdicts) and
  `results_flat.csv` (one row per arm, seed, split, metric, value, ci_lo, ci_hi, n) — written by
  `make packet` (`analysis/packet.py`, 2026-09-20) from the same run root after `make analysis` was re-run at
  a9ec8af and reproduced every file here byte for byte. Its section 12 rows 3–6 come from
  `configs/analysis/packet_notes.yaml`; the H2 threshold there is null (none is registered).

## Read before using any number

- **No truncation correction is applied anywhere.** All three seeds of GRPO-Mixed and of GRPO-Curated
  exceed 5 % truncation on test_300 greedy (5.3–10.3 %); no RFT run does. Under SPEC §7 those numbers
  are flagged and are not headline numbers as they stand; this touches H3, the H2 denominator and
  H1's GRPO contrast. Whether a correction is warranted is Laksh's decision.
- tasks/05 refers to a pre-registered threshold for the H2 ratio; none exists in writing. The sheet
  leaves a blank for it.
- RFT's configuration is the best of 9 on val (n = 100); GRPO ran one fixed recipe.
- Dollar figures use the $3.29/h Lambda billed and say so; meta.json recorded $4.29/h. GPU-hours are
  the primary quantity.
- Base gsm8k_500 mean@8 was never evaluated and shows as `missing`. Gemma was never evaluated.

## Not verified from the mirror

- The weights of the evaluated GRPO adapter (`adapter/step_300`) are not in the mirror; only
  `adapter/final` is. Adapter hashes (seeds differ) are therefore taken from `adapter/final`.
- The gsm8k_500 problem set is compared between units only (it is downloaded, not committed).
