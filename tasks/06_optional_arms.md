# tasks/06 — Optional arms (run only after tasks/05 shows where the uncertainty is)

Ordered by expected value; Laksh picks which to run.
1. **Seeds 4–5 on the H3 pair** (GRPO-Curated, RFT-Curated) if the H3 Δ is within ~1.5× its seed std.
2. **RFT step-matched**: RFT-Curated with optimizer steps = GRPO's (300 × 64 completions/step), config chosen on
   val — answers "did RFT win on compute?" if the RFT sweep picked 8 epochs.
3. **Iterated RFT** (ReST-style, 3 rounds, same 19,200 total) on train_curated — isolates on-policy re-sampling.
4. **Medium-only / hard-only 100** for RFT and GRPO — tests the "medium-difficulty" hypothesis from the VLM paper.
5. **Static vs adaptive prompt selection** for GRPO — re-curate every 100 steps from the current policy.
6. **Qwen3-8B-Base** point for the H3 pair only.
Each follows the tasks/03 or tasks/04 template exactly; no new hyperparameters.
