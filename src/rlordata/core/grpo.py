"""grpo.py — group relative policy optimization, the core math. Hand-written (Laksh).

INTUITION
    REINFORCE: sample an answer, get a reward, push the answer's tokens up in proportion to the
    reward. Problem: if every answer gets reward ~0.3, you push everything up and learn nothing
    about WHICH answers are better. Fix: subtract a baseline. GRPO's trick is that the baseline
    is free — sample G answers to the SAME prompt and use their mean reward. Above-average answers
    get positive advantage, below-average negative. A group where every answer is equally good
    (all right or all wrong) has zero advantage everywhere and teaches nothing: that is the
    "implicit filtering" in SPEC H2.

CONNECTION
    You have per-token log-probs for each completion (completion_logprobs). Multiply each token's
    log-prob by its completion's advantage and average: that is the REINFORCE loss. PPO/GRPO add
    two safety rails: (1) a clipped probability RATIO between the current policy and the policy
    that generated the samples, so one step cannot move too far; (2) a KL penalty to a frozen
    reference model so the policy does not drift into gibberish that happens to score well.

SHAPES
    rewards        [B, G]           B prompts, G samples each
    advantages     [B, G]           -> broadcast to [B*G, T] over tokens
    logp_new       [B*G, T]         current policy (with grad)
    logp_old       [B*G, T]         policy that generated the samples (no grad; == logp_new on the
                                    first inner step, ratio = 1)
    logp_ref       [B*G, T]         frozen reference (no grad)
    mask           [B*G, T]         completion tokens

PRECISE
    A[b,g]   = (r[b,g] - mean_g r[b,:]) / (std_g r[b,:] + eps)             # group-normalized
    ratio    = exp(logp_new - logp_old)                                     # per token
    surr     = min(ratio * A, clip(ratio, 1-eps_clip, 1+eps_clip) * A)     # per token
    kl       = exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1        # k3 estimator, >= 0
    L        = - mean_over_completion_tokens( surr - beta * kl )
    Token-mean over ALL completion tokens in the batch (TRL default at time of writing). Record
    the choice; Dr.GRPO/DAPO change exactly this normalization.

TESTS YOU WRITE (tests/core/test_grpo.py) — compute expected values BY HAND first, then encode:
    - rewards [[1,0,0,1],[0,0,0,0]] -> row 0 advantages = ±1 pattern (std over G with ddof=0 or 1:
      pick one, document it, match TRL); row 1 -> all zeros
    - with logp_new == logp_old, ratio == 1 and the loss equals -mean(A * mask) exactly
    - with beta = 0 and A = 0 everywhere, loss == 0 and grad == 0
    - kl term is >= 0 and == 0 when logp_ref == logp_new
"""

from __future__ import annotations

import torch


def group_advantages(
    rewards: torch.Tensor,  # [B, G]
    eps: float = 1e-6,
    normalize_std: bool = True,
) -> torch.Tensor:  # [B, G]
    """Group-relative advantages: (r - mean_G) / (std_G + eps). Zero for constant groups.

    std is unbiased (torch.std, ddof=1) to match TRL scale_rewards='group'.
    A constant group has std 0 → advantages 0 (implicit filtering).
    """
    assert rewards.ndim == 2
    mean = rewards.mean(dim=1, keepdim=True)
    centered = rewards - mean
    if not normalize_std:
        return centered
    std = rewards.std(dim=1, keepdim=True)  # unbiased
    # constant groups: std==0 (or nan for G=1) → zero advantage
    std = torch.where(std < eps, torch.ones_like(std), std)
    adv = centered / (std + eps)
    constant = rewards.max(dim=1, keepdim=True).values == rewards.min(dim=1, keepdim=True).values
    return torch.where(constant, torch.zeros_like(adv), adv)


def grpo_loss(
    logp_new: torch.Tensor,  # [N, T]
    logp_old: torch.Tensor,  # [N, T]
    logp_ref: torch.Tensor,  # [N, T]
    advantages: torch.Tensor,  # [N]
    mask: torch.Tensor,  # [N, T]
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped-ratio policy loss with KL-to-reference; token-mean over the batch (DAPO-style)."""
    n, t = logp_new.shape
    assert logp_old.shape == logp_ref.shape == mask.shape == (n, t) and advantages.shape == (n,)
    m = mask.to(dtype=logp_new.dtype)
    n_tok = m.sum().clamp_min(1.0)
    adv = advantages.unsqueeze(1)  # [N, 1] → broadcast over T
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surr = torch.minimum(unclipped, clipped)
    # k3 KL: exp(ref-new) - (ref-new) - 1  ≥ 0; 0 iff logp_ref == logp_new
    log_ratio_ref = logp_ref - logp_new
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    per_token = surr - beta * kl
    loss = -(per_token * m).sum() / n_tok
    with torch.no_grad():
        clip_frac = ((unclipped != clipped) * m).sum() / n_tok
        stats = {
            "mean_ratio": float((ratio * m).sum() / n_tok),
            "clip_fraction": float(clip_frac),
            "mean_kl": float((kl * m).sum() / n_tok),
            "mean_advantage": float(advantages.mean()),
            "n_tokens": float(n_tok),
        }
    return loss, stats
