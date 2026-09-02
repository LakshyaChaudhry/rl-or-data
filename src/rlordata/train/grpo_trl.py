"""GRPO arm via TRL GRPOTrainer + colocated vLLM (SPEC §8 arm 3, §9). AGENT-OWNED; tasks/04.

Reward function passed to TRL wraps rlordata.core.verify (primary) or the control rewards (configs/controls).
Record the exact TRL/vLLM versions, beta, clip eps, loss type, and normalization in the run directory.
Also provide `reference_grpo_loop.py` hooks so Laksh's hand-written loop (train/reference_grpo.py, his file)
can be run on the same tiny config and its reward curve diffed against TRL's (tasks/04 acceptance criterion).
"""

from __future__ import annotations
