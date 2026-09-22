# rl-or-data — is it the RL or the data?

Under matched prompt and rollout budgets, how much of low-data RLVR's gain comes from **data selection**
and how much from the **RL objective itself**? A controlled, pre-registered comparison of
rejection-sampling fine-tuning (RFT) and GRPO on procedurally generated counting problems, in the regime
of Bauer et al. (Snorkel, MLSys 2026): `Qwen/Qwen3-4B-Base`, LoRA, 100 training prompts, 19,200 sampled
completions per arm, three seeds per arm, one token cap for everything.

- `SPEC.md` — the locked protocol (v1.8). It wins every disagreement. `notebook/PREREGISTRATION.md` —
  hypotheses, predictions and logged deviations; `notebook/LAB_NOTEBOOK.md` — one entry per run or decision.
- `results/phase1…5/` — committed snapshots of every table, contrast, figure and grading sheet.
  **Start with `results/phase5/README.md`.** Hypotheses are graded by the author in `hypotheses.md`;
  nothing in the code writes a verdict.
- `reports/compute_log.md` — GPU type, GPU-hours and cost per arm.
- `src/rlordata/core/` — hand-written: verifier, log-probs, SFT loss, RFT selection, GRPO math, metrics.
  Everything else under `src/`, `scripts/`, `setup/`, `configs/` (outside `configs/locked/`) is
  agent-written against task files in `tasks/`; `CLAUDE.md` holds the rules those agents worked under.

## Arms (SPEC §8)

| arm | prompts | signal |
|---|---|---|
| RFT-Easy / -Mixed / -Curated | `train_easy_100` / `train_mixed_100` / `train_curated` (73) | SFT on verified-correct samples from one base-model draw (192 per prompt); lr/epochs = best of 9 on `val_mixed_100` |
| GRPO-Easy / -Mixed / -Curated | same three sets | binary verifiable reward, 300 steps × 8 prompts × 8 generations, one fixed recipe |
| C1, C2 (controls) | `train_mixed_100` | random reward; format-only reward |
| IterRFT-Curated (**secondary**) | `train_curated` | 3 rounds × 64 samples per prompt from the current policy; registered after the primary results were seen (PREREGISTRATION §6) |

Evaluation: greedy accuracy on `test_300` and `ood_hard_200` (primary), mean@8, pass@k (n = 64), GSM8K-500
transfer; per-problem bootstrap 95 % CIs; mean ± std across seeds with every seed shown; paired seed-wise
contrasts; the SPEC §10 criterion evaluated mechanically. A completion cut at the 4,352-token cap is scored
wrong; runs over 5 % truncation on `test_300` are flagged and the flag travels with every number.

## Reproduce

Everything that produces a number runs through vLLM on one GPU (1× H100 80 GB is what was used; see
`setup/LAMBDA_LAUNCH.md` — note that on Lambda only *terminate* stops billing, and
`setup/idle_shutdown.sh` does that after 30 idle minutes). The analysis needs no GPU.

```bash
# 0. environment
make setup-mac            # local (uv, CPU): tests, generator, analysis
make setup-gpu            # GPU box: env, tests, weights, idle guard (needs .env — see .env.example)
make test && make lint

# 1. data (CPU; the pool is committed — gen-check proves it regenerates identically)
make gen && make gen-ood && make gen-check

# 2. cap, tiering, base and reference evals (GPU, in this order)
make cap-run && make cap          # writes configs/locked/cap.yaml once; commit it
make tier                         # pass@8 tiers -> data/splits/
make eval-base                    # base + reference models: val / test / ood, greedy, mean@8, pass@k
make transfer-pick                # GSM8K-500 greedy for the base model

# 3. RFT arms (GPU): one shared draw, a 9-config sweep on seed 1 selected on val only, then seeds 2–3 + final evals
make rft-draw
for a in easy mixed curated; do make rft-sweep ARM=$a && make rft-finals ARM=$a; done

# 4. GRPO arms and controls (GPU): 11 trainings + 11 evals, idempotent queue
make grpo-queue                   # or one run: make grpo-train CONFIG=configs/grpo/mixed100.yaml SEED=1

# 5. secondary arm, exploratory larger-cap re-eval, base gsm8k mean@8 (GPU)
bash setup/tasks06b_box.sh        # = make tasks06b-queue after restore + preflight

# 6. analysis (CPU): cross-run sanity (fails loudly), tables, contrasts, figures, grading sheet, appendices
make analysis                     # RUN_ROOT=<store>/runs OUT=outputs to point it elsewhere
uv run python scripts/compare_phase_tables.py results/phase4 results/phase5 --expect-changed base,gsm8k_mean8
uv run python scripts/compute_log.py
```

Local dry runs of every GPU pipeline with a stub sampler and a tiny model (never results):
`make tier-dry eval-dry rft-dry iter-rft-dry`.

Run directories are not in git (weights, samples). Each holds the resolved config, config hash, git SHA,
package versions, wall-clock, GPU type and a `NOTES.md`; the small provenance files of every
result-bearing run are committed under `results/phase*/`. `make analysis` reads a store-style `runs/`
directory (default `runs/store_mirror/runs`), never writes into it, and loads only allow-listed final
units — superseded, failed, pre-amendment and exploratory runs are refused by path.

## What to know before using a number

- **Truncation differs between methods.** Every GRPO-Mixed and GRPO-Curated seed exceeds the 5 % flag on
  `test_300`; no RFT run does. No correction is applied anywhere. `results/phase5/appendix/` has post-hoc
  bounds (informative on test, uninformative on ood) and an asterisked re-generation at twice the cap that
  never enters the pre-registered criterion.
- **Tuning asymmetry.** RFT picked the best of 9 configurations on a 100-problem val set; GRPO ran one recipe.
- **Greedy decoding reproduces bit for bit on one host and not across hosts** (identical settings, packages
  and GPU model): 6–10 % of problems change correctness on re-generation, up to about 2 points of
  `test_300` accuracy for a single run. Arms were evaluated on different hosts. This is small next to the
  large contrasts and comparable to the small ones.
- Three seeds, one model family, LoRA, one synthetic task family, tiers defined by the trained model's own
  pass rate. `google/gemma-4-E4B-it` was planned as a reference model and was never evaluated.

## Release

Adapters: every trained run's final LoRA adapter (23: 6 arms × 3 seeds, 2 controls, the secondary
iterated-RFT arm × 3) is on the Hugging Face Hub in one collection —
<https://huggingface.co/collections/LakshC/rl-or-data-is-it-the-rl-or-the-data-lora-adapters-6ab22615cf8f3073cb1d17ae>
— **private until the write-up is out**; each model card states base model, data condition, seed, config
hash, budgets, and the run's own evaluation with n, CI and truncation, and every weights file was verified
against the local copy by sha256 after upload. Regenerate the staging with
`uv run python scripts/hf_release.py` (uploads nothing; `--push --namespace <name>` publishes).
