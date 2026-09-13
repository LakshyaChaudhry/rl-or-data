# tasks/04b — Reference GRPO loop vs TRL (the "I can explain every number" check)

Owner: split. **Laksh** writes `src/rlordata/train/reference_grpo.py` (his file; agent may not edit) using
`core.completion_logprobs`, `core.verify`, `core.group_advantages`, `core.grpo_loss`, mlx-free plain PyTorch +
PEFT, on `Qwen/Qwen3-0.6B-Base`. **Agent** provides the harness below.

## Agent provides
1. `tests/train/test_grpo_loss_oracle.py`: an oracle that reproduces TRL's loss for our pinned config
   (loss_type="dapo", scale_rewards="group", num_iterations=1, beta=0.0, epsilon=0.2) on *synthetic tensors*
   `(logp_new, logp_old, logp_ref, rewards[B,G], mask)`, written by reading the TRL source for the installed
   version and citing file:line in a comment. Then `core.grpo_loss(core.group_advantages(rewards)…)` must match it
   to 1e-6. This test is skipped until `core/grpo.py` is implemented; it is the only allowed skip.
2. `configs/grpo/tiny_0p6b.yaml`: Qwen3-0.6B-Base, 40 steps, 4 prompts × 8 gens, cap 512, on `train_mixed_100`,
   seed 1 — runnable on the Mac (MPS or CPU) and on the GPU.
3. `scripts/compare_reference_vs_trl.py`: runs TRL on the tiny config, runs `reference_grpo.py` on the same
   config, and writes both training-reward curves + final val greedy accuracy to `outputs/reference_check.json`
   with a plot.

## Acceptance
- Oracle test passes once `core/grpo.py` lands.
- On the tiny config, both curves rise; final val accuracies within each other's bootstrap CI; any divergence
  in the first 10 steps explained in `NOTES.md` (typical causes: masking of the first completion token,
  ratio computed with detached vs attached old log-probs, padding side).
