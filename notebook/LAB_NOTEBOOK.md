# Lab notebook — rl-or-data

One entry per run or decision. Newest at the top. Copy the template.

## Template
```
### YYYY-MM-DD — <run_id or decision>
- Config hash: | git SHA: | GPU: | wall-clock: | est. cost:
- What I ran / decided:
- Result (with n, seed, CI, truncation%):
- What I learned (one sentence):
- Next:
```

## Entries

### 2026-09-13 — tasks/04 GRPO: micro-batch 8 × 8 → 4 × 16 after step-1 OOM (engineering, not a protocol change)
- Config hash: (first relaunch run dir) | git SHA: 271280b + micro-batch change | GPU: 1× H100 PCIe 80 GB | wall-clock: 0.036 GPU-h (failed attempt) | est. cost: $0.15
- What I ran / decided: train_grpo_mixed_s1 initialized colocated vLLM (gpu_memory_utilization 0.3, max model length 8448; 32.1 GiB in use before step 1) and OOMed in the first backward pass (77.6 of 79.2 GiB in use; a 18.0 GiB allocation failed, which is the fp32 full-vocab logits for 8 sequences × ~3,974 tokens × 151,936). Micro-batch changed to 4 completions per backward × 16 grad-accum steps (`MICRO_BATCH_SIZE` in train/grpo_trl.py), identical for all 11 runs. Authorized by Laksh. Checked in the installed TRL 1.13.0 / transformers 5.16.1 source that the update is unchanged: rewards and group-std advantages are computed over the whole 64-completion generation batch before it is split; GRPOTrainer disables Trainer's grad-accum loss scaling; the dapo normalizer is the generation batch's completion-token count × grad_accum / steps_per_generation (= 16/16 = 1); one optimizer step per generation batch, num_iterations 1. Locked values (8 prompts × 8 gens = 64, 300 steps, 19,200 completions, cap, LoRA, lr, β, ε, loss type) untouched; RFT already trains with micro-batch 4 + accumulation. Differences are bf16 numerics from per-chunk padding and slower steps; logged `clip_ratio` / `step_time` average over 16 chunks instead of 8. The failed attempt's run dir (64 stale reward records, no checkpoints) is kept aside, not reused, so the 19,200-record gate stays exact. Also found: the `grpo` CLI and scripts/run_queue.py do not call `load_env()`; the queue must be launched with `.env` exported (fix off-box).
- Result: n/a (no training step completed).
- What I learned (one sentence): with vLLM colocated at 0.3, the full-vocab logits of an 8-sequence micro-batch at the 4352 cap do not fit in 80 GB, so micro-batch size is a memory knob that must be pinned and recorded like RFT's.
- Next: relaunch the queue; report GPU memory after the first backward, first 10 steps, and the eval_grpo_mixed_s1 sanity gates.

### 2026-09-12 — Phase 1 close-out: metrics committed to results/phase1/, Lambda box terminated
- Config hash: per unit (results/phase1/*/*/*/*/config_hash.txt) | git SHA: ffb604e (runs) | GPU: 1× H100 PCIe | wall-clock: 3.135 GPU-h over 31 units | est. cost: $13.45
- What I ran / decided: verified every stored eval/transfer unit against the frozen v1.6 splits (digests, extractor, sampler, cap, template), regenerated the summary, ran analysis/sanity, committed the small files. Store on the persistent filesystem keeps samples.jsonl. Gemma never ran for real (stub files only) → queued for the Phase 2 box. Stale clone at /lambda/nfs/rl-or-data/rl-or-data (d6cdae0) to be deleted next box.
- Result (seed 1, cap 4352, n=100/300/200): base greedy val 0.600 [0.510,0.700] (trunc 1.0 %), test 0.673 [0.617,0.727] (trunc 4.3 %), ood 0.250 [0.190,0.315] (trunc 13.0 %, reported); base mean@8 test 0.425, ood 0.173; base pass@1/8/64 on test first-100 0.426/0.803/0.980. Qwen3-4B test 0.747 / ood 0.455. Llama flagged (>5 % trunc). GSM8K-500 base greedy 0.880 (trunc 0). RG candidates 0.757 / 0.023 — none in band.
- What I learned (one sentence): greedy truncation on ood (13–22 % for Qwen models vs 3–4 % at T=1) means greedy loops on long pipelines, so ood greedy accuracy is partly a token-budget number and must be stated as such.
- Next: SPEC amendment for the transfer task (drop RG, keep GSM8K-500); Laksh fills predictions in PREREGISTRATION §3; tasks/03 (RFT arms).

### 2026-09-08 — SPEC v1.6: layered extraction, splits re-frozen (ffb604e)
- Config hash: tier_rescore run (results/phase1/tier/tier_rescore_Qwen__Qwen3-4B-Base_k8_seed1) | git SHA: ffb604e | GPU: none (offline rescore) | wall-clock: — | est. cost: $0
- What I ran / decided: v1.5 rejected `**Answer: 36**` and other decoration-only forms (3,326 correct answers scored wrong on the 49,600-completion tiering run) and gated correctness on format; v1.6 layers answer-line / \boxed{} / last-integer fallback, truncated → extraction failed, format compliance reported as answer_line_rate. Authorized by Laksh.
- Result: extraction failure 37.1 % → 0.9 %; pool tiers 350/2807/2843 → 1177/2657/2166; train_curated 79 → 73; cap unchanged at 4352 (v1.6 would give 4096; kept conservative).
- What I learned (one sentence): tiering by the model's own pass rate is only about reasoning if the verifier scores the number, not the formatting.
- Next: re-run base + reference evals on the new splits (done 2026-09-08, see above).

### 2026-09-08 — Phase 1 on Lambda: cap locked (1be86cb), splits frozen (e80bd00, superseded by ffb604e)
- Config hash: 34f9d68f (cap run) | git SHA: 8a39ac7 / d68b3d8 | GPU: 1× H100 PCIe | wall-clock: 0.127 GPU-h (cap run) + tiering 49,600 completions | est. cost: ~$5
- What I ran / decided: provisional cap run (500 val_candidates × 8, cap 4096) → SPEC §7 v1.3 rule → cap 4352 (measured term, cell M3 p99 3478). Tiering k=8 T=1 on 6000 pool + 200 ood, batch-invariant, bitwise identical to the provisional pass up to 4096 tokens.
- Result: 998 correct of 4000 on the cap run; 1.4 % of correct completions over 2048; 0 correct truncated at cap.
- What I learned (one sentence): the 2048 anchor cap would have truncated correct 3-step M-range completions, so the measured term binds.
- Next: v1.5 → v1.6 extractor (see above).

### 2026-09-01 — project locked
- SPEC v1.0 locked. Anchor: Bauer et al. MLSys 2026. Core grid: {RFT-all, RFT-curated, GRPO} × {easy-100, mixed-100} × 3 seeds on Qwen3-4B-Base.
- Credits: $10k AWS (YC Startup School). Expiry: ____. Region: ____. Quota requested: ____.

### 2026-09-07 — verify.py implemented
- What I ran / decided: implemented extract_answer + verify; 5 tests pass
- Result: n/a (unit tests)
- What I learned: reward is strict last-line Answer: N vs problem.answer
- Next: logprobs.py
- Parses a models' response for deterministic answer and assigns a reward based on that (0/1 in our case)