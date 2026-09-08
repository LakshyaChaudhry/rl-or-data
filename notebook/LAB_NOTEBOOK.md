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

### 2026-09-01 — project locked
- SPEC v1.0 locked. Anchor: Bauer et al. MLSys 2026. Core grid: {RFT-all, RFT-curated, GRPO} × {easy-100, mixed-100} × 3 seeds on Qwen3-4B-Base.
- Credits: $10k AWS (YC Startup School). Expiry: ____. Region: ____. Quota requested: ____.

### 2026-09-07 — verify.py implemented
- What I ran / decided: implemented extract_answer + verify; 5 tests pass
- Result: n/a (unit tests)
- What I learned: reward is strict last-line Answer: N vs problem.answer
- Next: logprobs.py
- Parses a models' response for deterministic answer and assigns a reward based on that (0/1 in our case)