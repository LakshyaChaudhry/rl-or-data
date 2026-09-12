"""sft_loss.py — supervised fine-tuning loss with completion-only masking. Hand-written (Laksh).

INTUITION
    SFT says: "make the tokens in this demonstration more likely." We only want to imitate the
    answer, not the prompt we wrote ourselves, so prompt tokens are masked out of the loss.

CONNECTION
    This is exactly cross-entropy, i.e. -mean(log-prob of the actual next token), restricted to
    completion positions. You already have completion_logprobs(); the loss is one line on top of it.

PRECISE
    L = - sum_{b,t} mask[b,t] * logp[b,t]  /  sum_{b,t} mask[b,t]        (token-mean)
    Record both token-mean and sequence-mean variants; TRL uses token-mean by default. Use token-mean.

TESTS YOU WRITE (tests/core/test_sft_loss.py)
    - all-masked batch -> raises (never silently return 0)
    - loss equals -mean(logp[mask]) computed with plain indexing
    - loss is invariant to right-padding
"""

from __future__ import annotations

import torch


def sft_loss(
    logp: torch.Tensor,  # [B, T] per-token log-probs (from completion_logprobs)
    completion_mask: torch.Tensor,  # [B, T] float/bool
) -> torch.Tensor:  # scalar
    """Token-mean negative log-likelihood over completion tokens."""
    assert logp.shape == completion_mask.shape and logp.ndim == 2
    mask = completion_mask.to(dtype=logp.dtype)
    n = mask.sum()
    if n.item() == 0:
        raise ValueError("sft_loss: completion_mask has no tokens (all-masked batch)")
    return -(logp * mask).sum() / n
