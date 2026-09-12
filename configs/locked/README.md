# configs/locked/

Protocol constants. Written once, never edited by the agent.

- `cap.yaml` — produced in Phase 1 by `scripts/compute_cap.py` from the length distribution of *correct*
  base-model completions on val_mixed_100 (SPEC §7). Contains: `max_completion_tokens`, `p99_correct_len`,
  `n_correct_used`, `computed_on`, `config_hash`. Every sampler and trainer reads it.
- `prompt.yaml` — the prompt template text (mirrors sampling/prompts.py) and the answer regex.
- `training.yaml` — LoRA/optimizer/GRPO constants from SPEC §9.
- `transfer.yaml` — secondary transfer splits (SPEC §6.3 / v1.7: GSM8K-500 only).
