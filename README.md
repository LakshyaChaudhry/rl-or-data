# rl-or-data

**Is it the RL or the data?** A matched-budget comparison of rejection-sampling SFT and GRPO on
procedurally generated counting tasks, extending Bauer et al. (Snorkel, MLSys 2026) with the SFT
control, multi-seed runs, and transfer evals.

- `SPEC.md` — the locked research protocol. Read first.
- `CLAUDE.md` — rules for coding agents (file ownership, scientific standards, commands).
- `tasks/` — one task file per component; agents work from these in order.
- `src/rlordata/core/` — hand-written by Laksh: verifier, log-probs, SFT loss, RFT selection, GRPO math, metrics.
- `notebook/` — lab notebook and pre-registration.
- `setup/` — Mac and GPU environment scripts, AWS notes, idle shutdown.

```
bash setup/setup_mac.sh      # local
make test                    # core tests skip until implemented; agent tests gate each task
```

Status: v1.0 scaffold, 2026-09-01.
