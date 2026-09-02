# tasks/02 — vLLM sampler, base + reference evals, token cap

Owner: agent. Files: `src/rlordata/sampling/vllm_sampler.py`, `sampling/prompts.py` (instruct wrapping only),
`scripts/compute_cap.py`, `src/rlordata/analysis/sanity.py` (split + cap checks), CLI `eval`.
Runs on the GPU box. Do not touch `core/**`.

## Order of operations (the cap must exist before tiering and before any other eval)

1. `VLLMSampler` per the contract in `vllm_sampler.py`. Deterministic given seed. Returns text,
   n_tokens, truncated (finish_reason == "length"). Batches all prompts × n in one vLLM call.
2. **Provisional cap run:** sample n=8 at T=1.0 on the *untiered* pool subset `val_candidates`
   (500 random pool problems, seed 1) with `max_completion_tokens=4096` for Qwen3-4B-Base. Score with
   `core.verify`. Run `scripts/compute_cap.py` → writes `configs/locked/cap.yaml` (p99 of correct
   completions × 1.25, ceil to 256, min 512). Print the length histogram (correct vs incorrect) and the
   fraction of correct completions that would be truncated at the chosen cap (must be < 1%).
3. **Tiering:** `rlordata tier` on the full pool with the locked cap → `data/splits/*.jsonl`.
   Print tier counts and the pass8 histogram. Run `analysis/sanity.py` disjointness checks.
4. **Evals** per `configs/eval/base.yaml`: greedy, mean@8, pass@k(n=64 on test_300 first 100) for the
   base and each reference model on val_mixed_100 / test_300 / ood_hard_200. Instruct models: chat
   template with thinking disabled (per-model kwargs recorded). Output `runs/eval/<model>/samples.jsonl`
   + `metrics.json` computed by `core.evaluate.compute_metrics`.
5. Transfer set selection: evaluate the base on the three Reasoning Gym candidates (SPEC §6.3) and pick
   the one with base accuracy in 20–60%; record the choice and its config hash in `configs/locked/transfer.yaml`.

## Acceptance criteria

- `configs/locked/cap.yaml` written with provenance; `VLLMSampler` refuses a different cap.
- Truncation rate < 5% and correct-completion truncation < 1% for the base on val at the locked cap.
- `runs/eval/` contains samples.jsonl + metrics.json for base and all reference models, with
  bootstrap CIs; a one-table summary printed (model × split × {greedy, mean@8, pass@8, trunc%}).
- Cost printed: GPU-hours × rate. Instance shut down or next job queued.

## Do not

- Do not compare instruct models in thinking mode. Do not raise the cap for a model that truncates a lot;
  report it.
- Do not evaluate test_300 more than once per model.
