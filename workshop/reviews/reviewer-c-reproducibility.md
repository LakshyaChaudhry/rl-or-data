# Reviewer C: reproducibility

Document under review: `workshop/aaai27-provisional/latex/main.pdf` (6 pages; body in `shared.tex`). Compared with the preprint appendices in `workshop/preserved-arxiv-v1/arxiv-source/main.tex` (arXiv:2610.25812v1), the evaluation and cap rules in `SPEC.md` §§5–10, `notebook/PREREGISTRATION.md`, `README.md`, and `results/phase5/tables/contrasts.md`. This review does not treat the Hugging Face adapters as public. The draft says they were described as private and were not re-checked, and that statement was left as written.

## Verdict

The 6-page PDF is careful about the quantities that define the claims: three seeds, the shared RFT draw, the selected RFT learning rates and epoch counts, the 4,352-token cap, the 5% truncation flag, and the pre-registration dates. A reader of this PDF alone still cannot rebuild the scorer, the GRPO update, or the RFT training texts, and cannot obtain the adapters. One gap blocks artifact-level checks. Several others are major because they change scores or hide a tie in the sweep that selected the baseline.

Logged aggregates can be audited. Table 4 matches the preprint per-seed appendix. The registered and secondary rows of Table 3 match `results/phase5/tables/contrasts.md` (analysis hash prefix `3ba220e48a66`, 10,000 resamples, bootstrap seed 0). The two post hoc GRPO-minus-RFT rows match a different file, cited below.

## What the PDF already gives a reader

- Code URL and the condensed preprint id, page 6: `https://github.com/LakshyaChaudhry/rl-or-data` and arXiv:2610.25812v1. The same paragraph says no new training was run for the condensation, and that logged greedy accuracies were checked against `results/`.
- Seeds 1, 2, and 3, with greedy accuracy and truncation on `test_300` and `ood_hard_200`, Table 4, page 5. Means in Table 2 are the means of those cells.
- Shared draw, §2.2, page 2: 192 completions per prompt from the base model at temperature 1, once, shared by every RFT arm and seed; exact duplicates dropped; completion-only loss, including the end-of-sequence token.
- Selected RFT settings and the search axes, §2.3, page 2: learning rates \(10^{-5}\), \(5\times10^{-5}\), \(10^{-4}\) and epochs 2, 4, 8; keep the best greedy accuracy on `val_mixed_100`; ties break toward fewer epochs, then a lower learning rate. Selected cells: Easy \(5\times10^{-5}\) for 8 epochs, Mixed \(5\times10^{-5}\) for 4 epochs, Curated \(10^{-5}\) for 4 epochs. The text says this is the best of nine on 100 problems, and is not a proven optimum.
- Cap value 4,352 for training and evaluation, with Bauer et al.'s 2,048 as a floor (§2.1, pages 1–2). A completion that hits the cap is a failure and is not scored from a trailing integer. Runs above 5% truncation on `test_300` are flagged (Tables 2 and 4).
- Extractor in one sentence, §2.1, page 2: last integer on an answer line, else in a boxed expression, else anywhere in the completion; exact match.
- GRPO knobs that are stated, §2.2, page 2: TRL `GRPOTrainer` v1.13.0, 300 steps, 8 prompts by 8 completions, learning rate \(5\times10^{-5}\), KL coefficient 0, one update per sampled batch, `loss_type=dapo`, equal-reward groups contribute no gradient, capped completions stay in the loss with reward 0, rollouts from vLLM. LoRA rank 64, \(\alpha=16\), all linear layers; AdamW; cosine schedule with 10% warmup; gradient clipping at 1.0; last-step evaluation.
- Registration trail, §2.3, page 2: hypotheses and the success rule committed 2026-09-07, predictions 2026-09-12, first training job 2026-09-13. The EOS restart, the H3 rollout mismatch (19,200 versus 14,016), and the post hoc status of the easy and mixed GRPO-versus-RFT rows and of iterated RFT are stated in the body.
- Bootstrap: paired over problems, 10,000 resamples, bootstrap seed 0 (§2.3, page 2, and the reproducibility note, page 6).
- Host sensitivity is in the limitations, page 5: re-evaluation on another machine moves test accuracy by up to 2.3 points and OOD accuracy by up to 5.0 points.

## Blocker

### B1. The adapters and the shared RFT completions are not obtainable

Page 6 states that Hugging Face adapters were described as private at the time of the preprint and were not re-checked as public for this draft. `README.md` says the same collection is private until the write-up is out, and that run directories (weights and samples) are not in git. The draw index at `results/phase2/rft/draw_seed1/draw_summary.json` points at `data/samples/base_train_easy_100_k192_seed1.jsonl` and `data/samples/base_train_mixed_100_k192_seed1.jsonl`. That `data/samples/` directory is not in the tree. `configs/rft/curated.yaml` and `configs/rft/iter_curated.yaml` read round 1 and the single-round arms from those files (`draw_seed: 1`).

The problem statements are obtainable: `data/pool/pool.jsonl` (seed 20260901 in `data/pool/pool.meta.json`) and the split files under `data/splits/`. The completions that were accepted or rejected for RFT, and the strings that were scored at evaluation, are not. The limitations on page 5 say a second machine moves test accuracy by up to 2.3 points, which is the size of the curation gap (−2.7), the mixed-data gap for RFT (+3.7), and the iterated-RFT gap (+3.2). Those three comparisons cannot be re-scored from released weights or from saved completions. The preprint reproducibility sentence said adapters were in the repository. The workshop correctly withdraws that claim. It does not replace it with a public artifact id, a checksum, or a committed sample file.

This is a release gap for artifacts that already exist in the authors' store. It is not a request for another training run.

## Major

### M1. The extractor a reader would code from the PDF, the locked config, and SPEC are three different rules

§2.1 (page 2) describes a three-layer rule and says the same extractor applies to every arm, control, and reference model. It does not mention decoration stripping, a "Final" prefix, digit-group commas, or the case where the label and the integer sit on different lines. Those details are the difference between SPEC §5 v1.6 and the earlier rule. The preprint appendix "Answer extraction" records why they matter: on the 49,600-completion tiering run, the earlier rule scored 3,326 correct answers as wrong and cut extraction failure from 37.1% to 0.9% after the change. That history is not in the 6-page PDF.

The repository the PDF cites does not resolve the conflict. `configs/locked/prompt.yaml` is marked locked to SPEC §5 and still says `extraction_rule: v1.5`, with two regular expressions and no last-integer fallback. `README.md` says the code and every result use layered v1.6 (`src/rlordata/core/verify.py`) and that the locked file was not edited. A reader who treats the locked config as the protocol, which SPEC tells them to do, will not implement the scorer that produced Tables 2–4.

The cap inherits the same split. The PDF gives the value 4,352 and the 2,048 floor (§2.1). It does not give the formula. SPEC §7 writes `max(2048, ceil_to_256(1.25 × p99_max))` and names `val_mixed_100` as the calibration set. The preprint appendix "Token cap", `configs/locked/cap.yaml`, and `PREREGISTRATION.md` §3 document a different calibration: 500 pool problems, before the splits existed, binding cell M3, p99 3,477.94 under the v1.5 lengths, which rounds to 4,352. The same note says rescoring those completions under v1.6 gives p99 3,161.72 and a cap of 4,096, and that 4,352 was kept. None of that counterfactual is in the PDF. The number used in the experiment is stated. The rule a later reader would recompute is not.

### M2. The 27-run sweep grid is gone, including a tie on the Mixed arm

§2.3 (page 2) states the axes, the tie-break, and the three selected cells. The validation accuracies for the other cells are not in the PDF. They are Table `tab:sweep` in the preprint appendix "Hyperparameters". On that table, RFT-Mixed scores 67 at both \(5\times10^{-5}\) for 4 epochs and \(10^{-4}\) for 4 epochs. The published cell is the one the tie-break selects. The next-lower Easy and Curated cells sit 3 points and 1 point below the winners, on a 100-problem validation set whose uncertainty the paper itself flags. Without the grid, a reader cannot see that the Mixed selection used the tie-break, or how close the rejected cells were. The data comparisons in §3.2 and §3.3 (pages 4–5) already warn that Easy, Mixed, and Curated were selected at different epoch counts or learning rates. The grid is the evidence for how fragile that choice was.

### M3. Library versions and the GRPO knobs that were pinned against defaults are not in the PDF

The only package version in the PDF is TRL `GRPOTrainer` v1.13.0 (§2.2, page 2). vLLM is cited as a system paper, with no version. The preprint hyperparameter appendix lists Python 3.11.16, PyTorch 2.13.0, Transformers 5.16.1, PEFT 0.20.0, and vLLM 0.28.0. Committed `meta.json` files under `results/phase5/` carry the same torch, transformers, peft, and vllm versions. SPEC v1.8 exists because an earlier draft called 0.04 the TRL default for the KL coefficient and was wrong. The PDF does state KL coefficient 0, `loss_type=dapo`, and one update per batch. It does not state clip \(\varepsilon = 0.2\), group advantage normalization by mean and standard deviation (`scale_rewards=group`), `top_p = 1.0`, weight decay 0, LoRA dropout 0, the module names, bf16, the RFT batch of 16, the GRPO micro-batch, greedy temperature 0, or the 4,096 prompt-token limit. Those values are in the preprint table and in `configs/locked/training.yaml` and `configs/eval/final.yaml`. Leaving clip, advantage scaling, and weight decay to a library default is how the KL coefficient was previously mis-set. Page 5's 2.3-point host effect is large enough that the stack is part of the small-effect claims, and the PDF does not record it.

### M4. The reproducibility note points at the wrong file for two headline intervals

Page 6 says the paired intervals are those in `results/phase5/tables/contrasts.md`. That file contains the registered rows, the controls, the GRPO curation row, and the two iterated-RFT rows. It does not contain GRPO-Easy minus RFT-Easy or GRPO-Mixed minus RFT-Mixed. Those rows are Table 3 on page 4 (+8.4 [+4.8, +12.1] and +9.2 [+4.8, +13.7], with OOD +8.3 and +3.5). They match `results/phase5/appendix/exploratory_grpo_vs_rft.md`, which the PDF never names. §3.1 uses those two matched-rollout gaps to argue that the extra curated rollouts are a poor sole explanation of the +10.9 gap. A reader who follows the page 6 pointer will not find them.

The adverse test bounds quoted in §3.1 (+7.8, +6.0, and +7.2 for curated, easy, and mixed) match the lower-bound scenario in `results/phase5/appendix/truncation_bounds.md` and the preprint truncation table. The PDF does not point at that file either.

### M5. Iterated RFT is underspecified relative to the registered procedure

§2.2 (page 2) says three rounds, 64 samples per prompt per round, round 1 from the shared base draw, rounds 2 and 3 from the current policy, and that each round continues the previous adapter and keeps only correct samples. `PREREGISTRATION.md` §6 and the preprint hyperparameter table add three choices that change the run: round 1 is the first 64 samples per prompt of the existing draw, each round starts a fresh AdamW and a fresh cosine schedule, and the reported checkpoint is the end of round 3 regardless of the validation curve. The PDF's phrase "continues the previous adapter" fixes the weights and leaves the optimizer state open. The pre-registration records that restarting from the base model on the union of rounds was considered and rejected. That fork is not visible in the 6-page text. The secondary contrasts (+3.2 and +7.7, Table 3) depend on it.

## Minor

### m1. Generator operators, rejection rules, and generator seeds are only in the preprint

§2.1 (pages 1–2) says 1–4 filters, 0–3 transformations, one final operation, every step must change the set, and generation is stratified by range size and step count. Spans are 10–200 in the pool and 201–1,000 on `ood_hard_200`. The operator lists (filters, transformations, final operations), the minimum of three paraphrases, the rejection rules, the S/M/L band edges, and seeds 20260901 and 20260902 are in the preprint appendix "Task generator and splits". They are also in `configs/data/pool.yaml` and `data/pool/pool.meta.json`. The committed pool makes the problems obtainable without regenerating them. The PDF itself does not name the seeds or the operators. `ood_hard_200`'s tier mix (14 easy / 47 medium / 139 hard in the preprint split table) is not in the workshop text.

### m2. Compute hours and the GPU are not in the PDF

The preprint limitations and appendix "Compute" report one NVIDIA H100 PCIe 80 GB per job, 212.1 GPU-hours for arms and controls, and 252.7 GPU-hours once base and reference evaluations, tiering, the shared draw, the exploratory re-evaluation, and discarded runs are included. `README.md` points at `reports/compute_log.md` and names the same GPU class. Page 5 of the workshop discusses the scientific limits and does not state hours or the device. This does not change the accuracy tables. It does remove the budget context the preprint used to justify three seeds on one task.

### m3. Truncation bounds beyond the three GRPO-minus-RFT test numbers, and the 8,704-token outcomes, are dropped

§3.1 reports the adverse test bounds for curated, easy, and mixed, and refuses an OOD conclusion because 20–55% of GRPO completions are cut. The full bound table (both directions, the data comparisons, iterated RFT, and OOD) is in the preprint appendix "Truncation". On OOD the curated lower bound in `truncation_bounds.md` is −11.8 points, so the sign is not identified. That figure is not in the PDF. The exploratory re-generation at 8,704 tokens is mentioned on page 5 as non-primary. The preprint numbers (GRPO-Curated ahead of RFT-Curated by 15.0 points on test and 11.5 on OOD at that cap, models still trained at 4,352) are not in the workshop PDF.

### m4. Sampled-accuracy detail is reduced to ranges

§3.1 gives mean@8 ranges (base 42.5%, single-round RFT 51.2–54.4%, GRPO 70.1–77.1%, format control 60.3%) and pass@64 on the first 100 test problems (base 98.0%, GRPO 95.7–96.7%). The per-arm table and the pass@k figure are in the preprint appendix. Hard-tier greedy numbers are still in the prose (base 29.0%, RFT 23.3–29.7%, GRPO-Easy 40.7%, GRPO-Mixed 58.0%, GRPO-Curated 52.7%). Medium-tier accuracies and mean completion lengths are not. GSM8K is reported as ranges (page 4: base 88.0%, single-round RFT 90.9–91.9%, GRPO 92.7–93.3%, iterated RFT 92.1%). The preprint lists each arm. The ranges match that list. Per-arm assignment does not survive.

### m5. Prompt text, seed meaning, and the reference-model template

The exact prompt string is in SPEC §5 and `configs/locked/prompt.yaml`. The PDF paraphrases it (§2.1: step-by-step reasoning and a final line `Answer: <integer>`, no chat template, no `<think>` tags). Seeds 1, 2, and 3 are tabulated. The PDF does not say that they control LoRA initialization, data order, and sampling, which the preprint hyperparameter table does say. It also does not give the draw seed. The draw seed is 1 in `configs/rft/curated.yaml` and in the preprint table.

§2.1 says every run uses one plain-text prompt with no chat template, and that the same cap and extractor apply to the reference model. SPEC §5 and the preprint appendix "Additional results" put the same text inside each reference model's chat template, with thinking disabled. Table 2 reports Qwen3-4B at 74.7% / 45.5%. A reader who follows the workshop sentence literally will not use that template. The other reference numbers in the preprint (Qwen2.5-7B-Instruct 65.3 / 28.0, Llama-3.1-8B-Instruct 37.3 / 9.0) are not in the PDF. `README.md` says Gemma-4-E4B-it was planned and never evaluated. The PDF does not mention it.

### m6. Extraction-failure rate and answer-line rate are not in the result tables

SPEC §10 asks for extraction-failure rate and `answer_line_rate` beside truncation. Tables 2–4 report truncation only. The preprint gives an answer-line movement in prose (about 77% to 82–85%, and 99% for the format control) and the tiering-run extraction-failure drop in the appendix. Neither document reports per-arm extraction failure on `test_300`. The draw summary in the repo does record extraction failure on the shared draw (about 0.5–0.6%). That number is not in the PDF.

### m7. Training-run config hashes are not in the PDF

Page 6 gives the analysis hash prefix `3ba220e48a66`. Individual training and eval config hashes live in the committed run metadata. The PDF does not list them. The per-seed table is enough to check the printed means. It is not enough to identify which resolved config produced a seed.

## Nits

- The analysis hash is printed as 12 hex characters (page 6 and `contrasts.md`). `exploratory_grpo_vs_rft.md` stores a longer hash that begins with that prefix. A reader cannot see whether the prefix is unique.
- Pooled seed standard deviation is not given a formula in the PDF. `contrasts.md` uses the square root of the pooled sample variance with divisor \(n-1\). Table 4 is sufficient to recompute the printed \(2\sigma\) column.
- `shared.tex` has an anonymous branch that drops the repository URL and the arXiv id. The named `main.pdf` keeps both. The anonymous companion, if that is what is submitted, has no code pointer in the text.

## What the preprint appendices held, and where it went

| Preprint appendix | In the 6-page PDF? |
|---|---|
| Pre-registration, quoted hypotheses, deviations | Dates, the three predictions, the EOS restart, the H3 budget mismatch, and post hoc labels are in §2.3. The verbatim quote is not. `notebook/PREREGISTRATION.md` still has it. |
| Task generator: operators, constraints, seeds, split table, examples | Two constraints, spans, and the train split counts are in §2.1. Operators, seeds, OOD tier counts, and examples are not. |
| Answer extraction, including the v1.5 failure counts | One sentence in §2.1. The decoration rules and the 3,326-answer history are not. |
| Token cap formula, per-cell lengths, v1.6 rescore to 4,096 | The value 4,352 and the 2,048 floor are in §2.1. The formula, the cell table, and the 4,096 counterfactual are not. |
| Hyperparameters, package versions, full sweep grid | Selected cells, search axes, tie-break, and TRL v1.13.0 are in §2.2–2.3. The 27 validation scores, the other library versions, clip, advantage scaling, batch size, and weight decay are not. |
| Per-seed greedy table | Present as Table 4, page 5. This is the appendix that was kept. |
| Truncation figure, full bound table, 8,704-token numbers, same-host byte match | Adverse test bounds for three GRPO-minus-RFT gaps are in §3.1. Host shifts of 2.3 and 5.0 points are on page 5. The rest is not. The preprint's same-host check (1,396 of 1,396 completions) is not in the PDF. |
| pass@k figure and sampled-accuracy table | Ranges and two pass@64 figures are in §3.1. The table and figure are not. |
| Extra GSM8K and reference-model numbers | GSM8K ranges and one Qwen3-4B row remain. Per-arm GSM8K and the other reference models do not. |
| Compute hours and GPU type | Not in the PDF. |

## Obtainability, as of this draft

| Object | Status |
|---|---|
| Code URL | Stated on page 6 of the named PDF. |
| Pre-registration | Stated as being in that repository. The file is `notebook/PREREGISTRATION.md`. |
| Problem pool and splits | In the tree (`data/pool/`, `data/splits/`). The PDF does not name those paths or the generator seeds. |
| Shared RFT draw (192 per prompt, seed 1) | Described in §2.2. The jsonl files named by the draw summary are not in the tree. |
| Resolved training configs | `configs/locked/training.yaml`, `configs/grpo/`, `configs/rft/`, `configs/eval/final.yaml`. The PDF does not point at them. `prompt.yaml` still labels the extractor v1.5. |
| Per-seed metrics and contrasts | Committed under `results/phase5/`. Table 4 and most of Table 3 can be checked there. The two post hoc intervals are under `results/phase5/appendix/`, not in the file named on page 6. |
| Eval completion strings | Not in the committed results tree (metrics and train logs only). |
| LoRA adapters | Page 6: described as private, not re-checked. This review does not treat them as public. |

The logged tables are checkable. The training texts, the scored strings, and the weights are not, and the 6-page PDF drops the sweep grid, the cap formula, the generator operators, the package versions, and the compute record that the preprint had put in appendices.

## Second pass

Re-read `main.pdf` (7 pages) and `shared.tex` only. Status is against B1 and M1 through M5 above. Adapters are still not treated as public.

### B1. Still open

The reproducibility note on page 6 still says the Hugging Face adapters were described as private at the time of the preprint and were not re-checked as public for this draft. §2.2 on page 2 still describes the shared base-model draw (192 completions per prompt, once, shared by every RFT arm and seed) and does not name a committed sample file. The code URL on page 6 still points at the repository for code, per-seed metrics, and the pre-registration. It does not add an artifact id or a checksum for the weights or for that draw.

### M1. Partly fixed

§2.1 on page 2 now states the cap rule that was missing: fixed before the splits existed, the larger of 2,048 and 1.25 times the largest per-cell 99th percentile of correct base-model lengths, rounded up to a multiple of 256. The same paragraph says a later extractor would have given 4,096 and that 4,352 was kept. The extractor sentence now says the fallback is the last integer anywhere in the completion. What remains open is the rest of that rule. Page 2 still does not mention decoration stripping, a "Final" prefix, or digit-group commas, and it still does not say that the locked prompt config names an older extractor than the one used for the tables.

### M2. Partly fixed

§2.3 on page 2 now reports the Mixed tie: RFT-Mixed also scored 67% at \(10^{-4}\) for 4 epochs, and the tie-break kept the lower learning rate. The same sentence says seed 1 of that sweep is one of the three seeds in the test mean. The other 24 validation cells are still absent, including the 3-point and 1-point margins on Easy and Curated.

### M3. Partly fixed

Page 6 now lists the stack for the logged runs: Python 3.11.16, PyTorch 2.13.0, Transformers 5.16.1, PEFT 0.20.0, TRL 1.13.0, and vLLM 0.28.0. §2.2 on page 2 still omits clip \(\varepsilon\), group advantage normalization by mean and standard deviation, `top_p`, weight decay, LoRA dropout, the module names, dtype, the RFT batch size, greedy temperature, and the prompt-token limit.

### M4. Fixed

The reproducibility note on page 6 no longer sends every interval to one file. Registered and secondary intervals are placed in `results/phase5/tables/contrasts.md`. The two post hoc GRPO-minus-RFT intervals are placed in `results/phase5/appendix/exploratory_grpo_vs_rft.md`. Truncation bounds are placed in `results/phase5/appendix/truncation_bounds.md`. The analysis hash is labeled a prefix (`3ba220e48a66`), with 10,000 resamples and bootstrap seed 0.

### M5. Partly fixed

§2.2 on page 2 now states the three missing choices: round 1 is the first 64 samples per prompt of the shared draw, each round starts a fresh AdamW optimizer and cosine schedule, and the reported checkpoint is the end of round 3. It also says rounds 2 and 3 refresh the sampler twice after that draw, against GRPO's 300 updates. §3.4 on page 5 repeats "resampling twice." The limitations on page 6 still say the iterated arm refreshes only three times. Those two counts disagree.
