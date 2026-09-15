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

### 2026-09-15 — two GRPO boxes terminated by the idle guard with nothing synced; guard fixed, relaunch made one command
- Config hash: n/a (infrastructure) | git SHA: 928825d + this commit | GPU: 1× H100 PCIe (both boxes) | wall-clock: ≈8.5 GPU-h box 1 (training landed), <1.2 GPU-h box 2 (setup only) | est. cost: ≈$36 + ≈$5
- What I ran / decided: box 1 finished `grpo_mixed_s1` training (300 steps, 19,200 records, 8.17 GPU-h, $35.04, finished 00:14 UTC) and its run dir reached the store at 00:15–00:17. The next job, `eval_grpo_mixed_s1`, never returned to the queue (a return, pass or fail, re-syncs the run dir and bumps ctimes on the store; none after 00:17:11), and no `eval/` was ever written. Box 2 (relaunch) ran `setup_gpu.sh` 03:22–03:30 UTC (HF cache touched) and wrote nothing to the store; gone by 04:31. Reproduced on the RFT box that the guard's pre-terminate sync fails every time: cron runs it as root and `sudo -u ubuntu -E` keeps HOME=/root, so `uv` dies on `/root/.cache/uv` (permission denied); the guard logs that to local syslog only and terminates. A guard termination therefore left no trace, which matches both boxes; nothing else does. What idled the GPU ≥30 min is not recoverable: box 1's eval hung or ran GPU-idle; box 2 either the same, or >30 min between the guard arming (03:30) and first GPU work (restore + tests). Fixes (engineering only): `-H` on the guard's sudo; guard log lines and a process/GPU snapshot appended to `<store>/logs/idle_guard_<host>.log` before terminate; `setup/grpo_box.sh` (keys + env + restore + queue preview + setup with guard last + launch, log on the store); `scripts/gpu_watch.sh` py-spy-dumps every rlordata process to the store after 10 idle minutes, before the guard's 30. Guard kept: it is what bounds a hung job's cost.
- Result (with n, seed, CI, truncation%): none. `grpo_mixed_s1` training artifacts are intact and re-usable (config hash 0b32e0bbbc9a); no GRPO eval exists yet.
- What I learned (one sentence): a safety script that cannot leave evidence turns every failure into a mystery — the sync-before-terminate must be tested under the exact user/HOME cron uses.
- Next: relaunch with `bash setup/grpo_box.sh`; the first job is `eval_grpo_mixed_s1` — if it idles the GPU, the dump under `<store>/logs/grpo_hang_*` says where it hangs.

### 2026-09-14 — grpo_mixed_s1 collapsed at step ~150: TRL merged the LoRA into bf16 base weights for vLLM; fixed with native vLLM LoRA (train/vllm_lora.py)
- Config hash: 172e905ff750 (failed run, now runs/grpo/_failed/grpo_mixed_s1_bf16merge_*) | git SHA: bbf832d (failed run) | GPU: 1× H100 PCIe | wall-clock: 11.02 GPU-h | est. cost: $47.32 (of which 7.6 GPU-h ≈ $33 were zero-gradient steps 154–300)
- What I ran / decided: the run trained normally to step ~140 (train reward 0.4→0.8, grad_norm ≈ 0.01), then grad_norm rose 800× over steps 142–151, entropy 0.05→4.5, and from step 154 all 64 completions per step were "toe toe …" to the 4352 cap: reward std 0 → advantage 0 → loss 0 for the last 146 steps. Cause (three independent measurements): TRL 1.13's colocate sync (`trl/generation/vllm_generation.py` sync_weights) does merge_adapter → load_weights → unmerge_adapter on the **bf16** base weights. Per-entry |ΔW| is 1e-6–1e-5 (layer 18 medians) while the bf16 half-ulp of a typical 0.014 weight is 3e-5, so the merge rounds 97.5 % (ckpt 25) → 87 % (ckpt 150) of ΔW entries to exactly zero; only 10 % → 60 % of ‖ΔW‖² reached vLLM (CPU measurement over all 252 LoRA modules; fp32 target keeps 100 %). vLLM therefore sampled from ≈ base + a coarsely quantised adapter while the trainer differentiated base + the full adapter: at ckpt 150 vLLM's own log-probs match the bf16-merged model within 0.006 nats, the true policy (fp32 base+adapter, matched by the trainer's bf16 path within 0.05–0.25 nats) rates the same samples 0.8–5 nats/token lower; mean |Δlogp| grew 0.009 (ckpt 0) → 0.02 (100) → 0.04 (125) → 5.2 (150). With ratio ≡ 1 (num_iterations 1), β = 0 and `vllm_importance_sampling_correction=False` (TRL's default True would have logged it), nothing corrected the off-policy update until the trainer's policy broke; the surviving part of the now-large adapter then dragged vLLM along (steps 152–154). Merging in fp32 first does not help (identical log-probs after the bf16 cast: the loss is the cast, and vLLM's weights are bf16). Fix (engineering, not a protocol change; authorised by Laksh): vLLM is constructed with `enable_lora=True, max_lora_rank=64`; each sync saves the PEFT adapter and hands vLLM a fresh `LoRARequest`; base weights inside vLLM are never touched. Verified on the failed run's adapters: vLLM-native-LoRA log-probs match fp32 base+adapter within 0.003–0.08 nats/token at ckpts 125/150, the same as the trainer (`runs/grpo/_diag/nativelora_grpo_mixed_s1`). The eval path (train/rft_eval.py) had the same bf16 merge and now evaluates adapters through `VLLMSampler(lora_path=…)` (native LoRA). New per-step metric `sampling/logp_absdiff_{mean,max}` (|logp_trainer − logp_vLLM| on the sampled tokens) in train_log.jsonl; alert above ~0.05 nats. Every result-bearing GRPO run must use this path (grpo_trl refuses vLLM server mode).
- Result (with n, seed, CI, truncation%): none; the run and its step-100 val eval (0.69, bf16-merged) are void. Base val greedy 0.60 [0.51, 0.70] for reference.
- What I learned (one sentence): a low-rank adapter is per-entry far below bf16 resolution of the base weights even when its effect on the outputs is large, so any bf16 merge (for sampling or for eval) silently evaluates mostly the base model; adapters must be applied as a separate branch and the sampler-vs-trainer log-prob gap must be logged.
- Next: rerun the queue from grpo_mixed_s1 on the fixed code; re-evaluate every RFT sweep adapter through native LoRA before `chosen.json` is trusted (the RFT adapters themselves are unaffected: no merge during SFT; measured bf16-merge loss for rft_mixed seed 1: lr1e-5/ep4 91 % of entries erased, lr1e-4/ep2 64 %, lr5e-5/ep8 41 %).

### 2026-09-13 — tasks/04 GRPO: micro-batch 8 × 8 → 4 × 16 after step-1 OOM (engineering, not a protocol change)
- Config hash: (first relaunch run dir) | git SHA: 271280b + micro-batch change | GPU: 1× H100 PCIe 80 GB | wall-clock: 0.036 GPU-h (failed attempt) | est. cost: $0.15
- What I ran / decided: train_grpo_mixed_s1 initialized colocated vLLM (gpu_memory_utilization 0.3, max model length 8448; 32.1 GiB in use before step 1) and OOMed in the first backward pass (77.6 of 79.2 GiB in use; a 18.0 GiB allocation failed, which is the fp32 full-vocab logits for 8 sequences × ~3,974 tokens × 151,936). Micro-batch changed to 4 completions per backward × 16 grad-accum steps (`MICRO_BATCH_SIZE` in train/grpo_trl.py), identical for all 11 runs. Authorized by Laksh. Checked in the installed TRL 1.13.0 / transformers 5.16.1 source that the update is unchanged: rewards and group-std advantages are computed over the whole 64-completion generation batch before it is split; GRPOTrainer disables Trainer's grad-accum loss scaling; the dapo normalizer is the generation batch's completion-token count × grad_accum / steps_per_generation (= 16/16 = 1); one optimizer step per generation batch, num_iterations 1. Locked values (8 prompts × 8 gens = 64, 300 steps, 19,200 completions, cap, LoRA, lr, β, ε, loss type) untouched; RFT already trains with micro-batch 4 + accumulation. Differences are bf16 numerics from per-chunk padding and slower steps; logged `clip_ratio` / `step_time` average over 16 chunks instead of 8. The failed attempt's run dir (64 stale reward records, no checkpoints) is kept aside, not reused, so the 19,200-record gate stays exact. Also found: the `grpo` CLI and scripts/run_queue.py do not call `load_env()`; the queue must be launched with `.env` exported (fix off-box).
- Result: n/a (no training step completed).
- What I learned (one sentence): with vLLM colocated at 0.3, the full-vocab logits of an 8-sequence micro-batch at the 4352 cap do not fit in 80 GB, so micro-batch size is a memory knob that must be pinned and recorded like RFT's.
- Next: relaunch the queue; report GPU memory after the first backward, first 10 steps, and the eval_grpo_mixed_s1 sanity gates.
- grpo_mixed_s1 training ran on bbf832d; every later job runs on 7ab34a9 (bookkeeping-only diff: resume log trimming, .env loading, completions counted from file; no change to training, sampling, or dependencies).

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