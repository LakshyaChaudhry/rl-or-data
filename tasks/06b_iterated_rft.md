# tasks/06b — Iterated RFT arm, exploratory larger-cap re-eval, base gsm8k mean@8

**DRAFT written by the agent on 2026-09-20 from Laksh's decisions in session; Laksh edits and signs off before
anything runs.** Owner: agent for code and runs; Laksh for the pre-registration text and every verdict.
Branch: `tasks06b-iterated-rft`. GPU budget: ≈ 15 GPU-h on 1× H100 (≈ 12 + 2 + 0.3), idle guard on.

Decided by Laksh (2026-09-20): run iterated RFT (tasks/06 item 3); rollout budget 14,016; run the restricted
larger-cap re-eval, marked in the paper as "what performance would have been without truncation"; run base
gsm8k_500 mean@8; **no** seeds 4–5, **no** lr sensitivity run. Still open: item B3 below (marked ☐).

**Prerequisites (do not start a GPU run before all exist):**
1. `notebook/PREREGISTRATION.md` §6 filled in **by Laksh** (his predictions) and §4 deviation entries approved,
   committed; every run dir records a git SHA descended from that commit. If it isn't, stop and say so.
2. Store filesystem mounted with the RFT draw (`data/samples/base_train_mixed_100_k192_seed1.jsonl`) and, for
   part C, the final adapters of `rft_curated` and `grpo_curated` seeds 1–3 (`adapter/final`, `adapter/step_300`).

## A. Iterated RFT (`iter_rft_curated`), SPEC §8 Priority 2

- A1. Prompts: the frozen `train_curated` (73). Never re-curated: selection is
  `rft_select(train_curated_problems, samples, mode="all", max_per_problem=None)` — never `mode="curated"`.
- A2. Seeds 1, 2, 3. Three rounds. 64 samples per prompt per round → 73 × 64 × 3 = **14,016** rollouts, equal
  to the samples RFT-Curated's prompts own in the 192-draw (SPEC §8.2).
- A3. Round 1 data = the **first 64 samples per prompt, in generation order,** of the existing base draw (no new
  sampling; identical for all three seeds, like RFT-Curated). Rounds 2 and 3: 64 per prompt from the current
  policy π_{r−1} through vLLM native LoRA, T = 1.0, top_p = 1.0, locked cap 4352, locked prompt; sampler seed
  = 1000 × train_seed + round. Score with `core.verify`; dedup and EOS-append exactly as tasks/03 §3.
- ☑ B3 (**confirmed by Laksh 2026-09-20**, on the agent's recommendation). Each round **continues the previous round's adapter**
  and trains on **that round's kept samples only**, with the tasks/03 recipe applied per round: lr 1e-5,
  4 epochs, batch 16, fresh AdamW, fresh cosine schedule with 10 % warmup. (Alternative: restart from base on
  the union of all rounds so far.) Expected ≈ 1,900 optimizer steps / ≈ 15M tokens in total, against
  RFT-Curated's 1,664 / 13.4M.
- A4. Hyperparameters: RFT-Curated's val-chosen config (lr 1e-5, 4 epochs) reused in every round; **no new
  sweep**. This departs from tasks/03 §4 and is logged in PREREGISTRATION §4 with its bias direction.
- A5. Final = the adapter after round 3, whatever the val curve says. `val_mixed_100` greedy after every
  round (val only). Then the identical 9 `eval/final` units, once. Nothing reads test_300 or ood_hard_200
  before that; keep the grep test of tasks/03 §4 for the new orchestrator.
- A6. Per-round diagnostics in `rounds.json` and `train_log.jsonl`: per-prompt and per-tier pass rate of the
  64 samples; # all-correct and # zero-correct prompts; kept examples before/after dedup by tier; unique-
  completion fraction; mean / p99 tokens and truncation of sampled vs kept completions; answer_line_rate;
  mean log-prob of the kept samples under π_0 and under π_{r−1}; val greedy accuracy, truncation, mean
  tokens, fraction of val outputs changed vs the previous round; tokens and optimizer steps per round.
- A7. Budgets (`budgets.json`): prompts 73, completions_available 14,016, completions_consumed = Σ kept,
  training_tokens, optimizer_steps, per-round breakdown. Same sanity gates as tasks/03 §5 (adapter
  non-trivial, final = round 3, tokenizer hashes, outputs differ from base, `append_eos: true`).
- A8. Layout: `runs/rft/iter_rft_curated/seed{s}/round_{1,2,3}/…`, `adapter/final` = round 3,
  `eval/final/…`. No new entry in `chosen.json` machinery; the run config records the reused lr/epochs.

## B. Analysis additions (after A finishes)

- Add the arm to `configs/analysis/default.yaml`; add contrasts `IterRFT − RFT-Curated` and
  `GRPO-Curated − IterRFT` (SPEC §10 criterion, unchanged) to `report.CONTRASTS`, labelled **secondary,
  registered after unblinding**; grading-sheet section with a blank verdict; footnote that RFT-Curated's
  seed std excludes sampling variance (one shared draw) while IterRFT's includes it.
- Show that every existing number in `results/phase4` is byte-identical after the config change (diff the
  CSVs for the existing arms), then snapshot to `results/phase5/`.
- Post-hoc truncation bounds (**approved by Laksh 2026-09-20**): appendix table, never in `tables/results.md` or on a
  criterion line; threshold recomputed inside each scenario; ood bounds stated to be uninformative.

## C. Exploratory larger-cap re-eval (deviation; PREREGISTRATION §4)

- Cap fixed by rule before running: **8,704 = 2 × the locked 4,352**. Greedy only, `test_300` and
  `ood_hard_200`. Models: base, RFT-Curated seeds 1–3, GRPO-Curated seeds 1–3, IterRFT seeds 1–3 (20 units).
  Same prompt, extractor, seeds and vLLM settings as the primary evals.
- `sampling/cap.resolve_cap` refuses any other cap by design. Add one narrow, explicit path
  (`--exploratory-cap 8704`) that: refuses to write under `runs/eval/` or any `eval/final/`; writes to
  `runs/exploratory_cap8704/…`; stamps `exploratory: true` and `cap_deviation: "PREREGISTRATION §4 2026-09-20"`
  into config.yaml and metrics.json; is covered by a test that the normal path still raises.
- Verify checkpoint identity first: sha256 of each evaluated adapter equals the one the primary eval read
  (GRPO: `adapter/step_300`).
- Reporting: a separate, asterisked table/figure — "* re-generated at a cap of 8,704 tokens to show what
  accuracy would have been without truncation; exploratory, run after the primary results were seen; the
  policies were trained with completions capped at 4,352". Never enters the SPEC §10 criterion, the primary
  tables, or the abstract's headline number. The analysis loader must not be able to load it as a result.

## D. Base gsm8k_500 mean@8

- `Qwen/Qwen3-4B-Base`, gsm8k_500, n = 8, T = 1.0, locked cap, seed 1 → `runs/eval/Qwen__Qwen3-4B-Base/
  gsm8k_500/mean_at_k`; then remove the `known_missing` entry from the analysis config.
- Not in scope: the Gemma reference eval (still owed since Phase 1; the write-up must say it was not run).

## Acceptance
- `make test` and `make lint` pass; determinism test for the round sampler seeds; dry run of the whole
  orchestrator with the stub sampler and a tiny model (as `make rft-dry`).
- Every run dir complete (CLAUDE.md list); cost printed at start and end; synced to the store; box terminated.
- `make analysis` regenerates everything; no number without n, seeds, CI, truncation; verdict lines blank.
