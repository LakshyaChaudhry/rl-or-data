# Phase 1 results — base and reference evals, transfer pick, tiering (SPEC v1.6)

Small provenance files (metrics.json, config.yaml, config_hash.txt, meta.json, NOTES.md) copied from the
artifact store on 2026-09-12. The samples.jsonl files (every completion) stay in the store at
`$RLORDATA_ARTIFACTS/runs/...` on the Lambda persistent filesystem `rl-or-data`; the layout below mirrors it.

Every unit here was verified on 2026-09-12 (scripts inline in the commit that added this directory):
`extraction_rule: v1.6`, `sampler: vllm`, `max_completion_tokens: 4352`, `max_prompt_tokens: 4096`,
identical prompt template, `status: finished`, `n_samples == n_samples_planned`, and `problem_ids_sha256`
equal to the digest of the committed split (ffb604e):

| split | problem_ids_sha256 |
|---|---|
| val_mixed_100 | f56d989915dcc48f69a71b20be8d9faa3d7f4a9880c7fea08fb1e54641b676c2 |
| test_300 | 78bc8efb2a37a6417b13bcf83ae8090840f5e90096b0114daaf091ed6d9989fc |
| test_300 first 100 (pass@k) | 2be7e27b899fb50b8f79db8795b8fc451b9e85547249ff90c5017a32ef60f266 |
| ood_hard_200 | 5a7fb3b7a2ea01028690ae5fe516f2d70400fb26f294c12d68a00e49105c8d4d |

`python -m rlordata.analysis.sanity --splits-dir data/splits --run-dirs results/phase1/eval/*/*/*` passes the
protocol-identity check; the only flags are the ones listed below.

## What is here

- `eval/` — Qwen3-4B-Base, Qwen3-4B (non-thinking), Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct on
  val_mixed_100 / test_300 / ood_hard_200; greedy, mean@8 (n=8, T=1), pass@k (n=64 on test_300 first 100).
  `summary.md` / `summary.json` regenerated locally with `eval_runner.summarize`. Seed 1, cap 4352.
- `transfer_pick/` — base greedy on rg_basic_arithmetic_300 (0.757), rg_count_primes_300 (0.023, 9.3 %
  truncation), gsm8k_500 (0.880, 0 % truncation). No Reasoning Gym candidate lands in the SPEC §6.3
  20–60 % band; amendment pending.
- `tier/` — `tier_Qwen__Qwen3-4B-Base_k8_seed1` is the v1.5-scored tiering (superseded);
  `tier_rescore_Qwen__Qwen3-4B-Base_k8_seed1` is the v1.6 offline rescore of the same 49,600 completions
  that produced the committed splits (pool easy 1177 / medium 2657 / hard 2166; train_curated 73);
  `_provisional` holds only provenance for the pre-cap sampling pass.

## Not here

- **google/gemma-4-E4B-it**: never evaluated for real. The store holds stub-sampler v1.5 units under
  `runs/eval/google__gemma-4-E4B-it` (excluded here) and 4-problem vLLM smoke runs (`runs/gemma_smoke`).
  Needs a full run on the next GPU box.
- `runs/out2` (stub layout test) and other leftovers listed in the 2026-09-12 close-out report.

## Flags (SPEC §7)

- Llama-3.1-8B-Instruct exceeds 5 % truncation on val and test (8.0 % / 11.3 % greedy; 14.5 % / 15.2 %
  at T=1): reported, **not a headline reference row**.
- Greedy truncation on ood_hard_200 is high for every model (base 13.0 %, Qwen3-4B 22.5 %,
  Qwen2.5-7B 19.0 %, Llama 47.5 %) while T=1 truncation on the same split is 3–4 % for the Qwen models:
  greedy decoding loops on long pipelines. Reported, not flagged, per §7; the write-up must state that
  ood greedy accuracy is partly "ran out of tokens".
- Under v1.6 a truncated completion is always an extraction failure, so the two columns are identical
  for greedy runs by construction and are not independent evidence.

## Cost

31 units, 3.135 GPU-hours on 1× H100 PCIe at $4.29/h ≈ $13.45 (from meta.json `gpu_hours_actual`).
