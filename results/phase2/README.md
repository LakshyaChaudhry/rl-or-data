# Phase 2 results — RFT arms (tasks/03), SPEC v1.8

Small provenance files copied from the artifact store on 2026-09-19 (close-out): `metrics.json`,
`config.yaml`, `config_hash.txt`, `meta.json`, `NOTES.md`, `sanity.json`, `summary.json`, `budgets.json`,
`train_log.jsonl`, and per arm `sweep.json`, `chosen.json`, `report.json`, `report.md`. The layout mirrors
the store: `rft/<arm>/<run>/...` here is `$RLORDATA_ARTIFACTS/runs/rft/<arm>/<run>/...` on the Lambda
persistent filesystem `rl-or-data`. No file here exceeds 1 MB.

No arm-vs-arm claim is made here; that is tasks/05.

## What is here

- `rft/draw_seed1/` — provenance of the 192-sample draw (git a90d24e); `draw_summary.json`.
- `rft/rft_{easy,mixed,curated}/` — per arm:
  - 9 sweep runs `seed1_lr{1e-05,5e-05,0.0001}_ep{2,4,8}`, each with its `eval/val` unit (val_mixed_100
    greedy through native vLLM LoRA — the unit selection used);
  - `sweep.json`, `chosen.json` — selection on `val_mixed_100` only, tie-break fewer epochs then lower lr:
    easy lr 5e-05 / 8 epochs (val 0.67), mixed lr 5e-05 / 4 epochs (val 0.67), curated lr 1e-05 / 4 epochs
    (val 0.66). n = 100 val problems, seed 1; bootstrap 95 % CI and truncation rate per sweep run in
    `sweep.json`;
  - the chosen config at seeds 1, 2, 3, each with `eval/final`: val_mixed_100, test_300, ood_hard_200,
    gsm8k_500 (greedy and mean@8) and test_300 first-100 pass@k (n = 64);
  - `report.md` / `report.json` — per-seed table with bootstrap 95 % CIs, truncation and extraction-failure
    rates, budgets, GPU-hours, and mean ± std over the 3 seeds.
- `*_bf16merge_stale.json` and `eval/val_bf16merge_stale/` — the first selection, evaluated through a bf16
  merge of the adapter into the base weights. Superseded: all three winners changed when re-selected
  through native vLLM LoRA (notebook 2026-09-17, 42aeb3a). Kept as a record; never rank on these.

## Not here

- `samples.jsonl` (every completion), adapters and epoch checkpoints: in the store only (95 GB under
  `runs/rft`). Adapters go to Hugging Face in tasks/07.
- `runs/rft_noeos_ablation/` (20 GB in the store): the discarded pre-amendment runs trained without EOS
  (PREREGISTRATION §4, 2026-09-13). Not results.
- **google/gemma-4-E4B-it**: still never evaluated for real — the store holds only the 2026-09-08
  stub-sampler units. Owed since Phase 1.

## Verification (2026-09-19, read-only against the store)

All 33 training runs: `status: finished`, `append_eos: true`, `adapter/final` present, `git_dirty: false`.
All 135 eval units (108 current + 27 stale): `status: finished`, `n_samples == n_samples_planned`,
`sampler: vllm`, `max_completion_tokens: 4352`, `max_prompt_tokens: 4096`, `extraction_rule: v1.6`,
`samples.jsonl` present with `n_samples` lines, and `problem_ids_sha256` equal to the committed split digests listed in
`results/phase1/README.md`. gsm8k_500 digest, identical in every unit of phases 2 and 3:
`21b915d91ce064b3fd0ef190a07897f9fb673f19ba258fcfaed1ea5aa8a3ee1c`.

Training git SHAs: sweeps b877fe3; finals f5c79dd / 59d65a0. The finals queue ran unattended at 59d65a0
and exited 0 (store `logs/rft_finals_queue_20260918T1604Z_requeue1.log`); the box was then terminated by
the idle guard after a full sync.

## Flags (SPEC §7)

- No final unit exceeds 5 % truncation on val_mixed_100 or test_300.
- ood_hard_200 greedy truncation is 8.0–15.5 % across the 9 final runs: reported, not flagged, per §7.

## Cost

From `meta.json` (`gpu_hours_actual`): 33 training runs 104.77 GPU-h, 135 eval units 5.62 GPU-h, draw
0.58 GPU-h, on 1× H100 PCIe. The recorded dollar figures ($449 + $24 + $2) use the $4.29/h rate
recorded in `meta.json`; Lambda's terminate response for this box shows $3.29/h, so actual spend is about 23 % lower.
