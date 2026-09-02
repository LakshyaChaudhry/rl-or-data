"""Phase 1: compute the locked token cap (SPEC §7) from correct base-model completions on val_mixed_100.

AGENT-OWNED (tasks/02). Reads runs/eval/<base run>/samples.jsonl (T=1.0, n=8), keeps correct samples,
takes p99 of n_tokens, multiplies by 1.25, rounds up to a multiple of 256, floors at 512, and writes
configs/locked/cap.yaml with provenance. Refuses to overwrite an existing cap.yaml.
"""

raise SystemExit("tasks/02: not implemented")
