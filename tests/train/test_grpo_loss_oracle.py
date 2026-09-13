"""tasks/04b §1 — an oracle for TRL's GRPO loss under our pinned config, on synthetic tensors.

(Citations re-pointed from TRL 1.12.0 to 1.13.0 on 2026-09-13: every 1.12→1.13 change in
grpo_trainer.py sits before line 1044, so each cited line moved +4 with identical content;
utils.py nanstd moved +5, body unchanged.)

The oracle below is a line-by-line transcription of TRL **1.13.0** (the version pinned in uv.lock;
source read from the wheel ``trl-1.13.0-py3-none-any.whl``), restricted to the configuration SPEC
v1.8 §9 pins and ``train/grpo_trl.py`` sets explicitly:

    loss_type="dapo"  scale_rewards="group"  num_iterations=1  beta=0.0  epsilon=0.2 (epsilon_high
    unset → symmetric)  importance_sampling_level="token"  delta=None  top_entropy_quantile=1.0
    mask_truncated_completions=False  vllm_importance_sampling_correction=False  one process,
    gradient_accumulation_steps == steps_per_generation (so the dapo normalizer ratio is 1).

Citations are ``file:line`` inside the wheel:

  advantages   trl/trainer/grpo_trainer.py:2791-2792   mean_grouped_rewards = nanmean(rewards.view(-1, G), dim=1)
               trl/trainer/grpo_trainer.py:2793-2796   std_rewards = nanstd(rewards.view(-1, G), dim=1)
               trl/trainer/utils.py:864-887            nanstd: Bessel-corrected (count/(count-1)); NaN for count==1
               trl/trainer/grpo_trainer.py:2811-2813   advantages = (rewards - mean) / (std + 1e-4)
               trl/trainer/grpo_trainer.py:2836        advantages = nan_to_num(advantages, nan=0.0)
  ratio        trl/trainer/grpo_trainer.py:3158-3159   old_per_token_logps = per_token_logps.detach() if None
               trl/trainer/grpo_trainer.py:3176-3178   log_ratio = new - old  (token level)
               trl/trainer/grpo_trainer.py:3188        coef_1 = exp(log_importance_weights)
  kl           trl/trainer/grpo_trainer.py:3191-3198   only if beta != 0: k3 = exp(ref-new) - (ref-new) - 1,
                                                       then kl *= coef_1 when use_bias_correction_kl (default True)
  clip         trl/trainer/grpo_trainer.py:3206        coef_2 = clamp(coef_1, 1 - eps_low, 1 + eps_high)
               trl/trainer/grpo_trainer.py:3211-3213   per_token_loss = -min(coef_1 * A, coef_2 * A)
               trl/trainer/grpo_trainer.py:3242-3243   per_token_loss += beta * per_token_kl   (beta != 0)
  dapo norm    trl/trainer/grpo_trainer.py:2520        num_items_in_batch = gather(loss_mask.sum()).sum()
               trl/trainer/grpo_trainer.py:3261-3266   loss = (per_token_loss * mask).sum() / num_items_in_batch
                                                       (× grad_accum / steps_per_generation, == 1 for us)

With ``num_iterations=1`` TRL passes ``old_per_token_logps=None`` and the ratio is exactly 1; the
synthetic ``logp_old != logp_new`` cases below exercise the clip path the reference loop must still
get right. The KL case is included with TRL's bias correction *off* to document the one place the
two implementations differ when beta > 0 (SPEC pins beta = 0, so it never enters a result).

Skipped until Laksh's ``core/grpo.py`` is implemented — the only allowed skip in tasks/04b.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:  # a half-edited core file raises SyntaxError, not ImportError — still "not ready", not a failure
    from rlordata.core import grpo as core_grpo
except Exception as e:  # noqa: BLE001
    pytest.skip(f"core/grpo.py not importable yet (Laksh): {e}", allow_module_level=True)

from tests.conftest import skip_unless_implemented  # noqa: E402

TRL_VERSION = "1.13.0"
EPS_ADV = 1e-4  # grpo_trainer.py:2813
CLIP_EPS = 0.2  # SPEC v1.8 §9


def trl_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """grpo_trainer.py:2791-2813 + utils.py:864-887 for scale_rewards='group'. rewards [B, G] → [B*G]."""
    g = rewards.shape[1]
    flat = rewards.reshape(-1)
    grouped = flat.view(-1, g)
    mean = torch.nanmean(grouped, dim=1)  # :2791
    # nanstd (utils.py:879-887): population variance × count/(count-1); nan when count == 1
    m = torch.nanmean(grouped, dim=1, keepdim=True)
    var = torch.nanmean((grouped - m) ** 2, dim=1, keepdim=True)
    count = torch.sum(~torch.isnan(grouped), dim=1, keepdim=True)
    corr = torch.where(
        count > 1, count / (count - 1), torch.full_like(count, float("nan"), dtype=var.dtype)
    )
    std = torch.sqrt(var * corr).squeeze(1)
    mean = mean.repeat_interleave(g, dim=0)  # :2792
    std = std.repeat_interleave(g, dim=0)  # :2796
    adv = (flat - mean) / (std + EPS_ADV)  # :2811-2813
    return torch.nan_to_num(adv, nan=0.0)  # :2836


def trl_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    advantages: torch.Tensor,  # [N]
    mask: torch.Tensor,
    *,
    beta: float = 0.0,
    clip_eps: float = CLIP_EPS,
    bias_correction_kl: bool = False,
) -> torch.Tensor:
    """grpo_trainer.py:3148-3266 for loss_type='dapo', importance_sampling_level='token', delta=None."""
    adv = advantages.unsqueeze(1)  # :3152
    log_ratio = logp_new - logp_old  # :3176
    coef_1 = torch.exp(log_ratio)  # :3178, :3188
    coef_2 = torch.clamp(coef_1, 1 - clip_eps, 1 + clip_eps)  # :3206 (epsilon_low == epsilon_high)
    per_token_loss = -torch.min(coef_1 * adv, coef_2 * adv)  # :3211-3213
    if beta != 0.0:  # :3191
        d = logp_ref - logp_new
        kl = torch.exp(d) - d - 1  # :3193-3195
        if bias_correction_kl:  # :3197-3198 (TRL default True)
            kl = kl * coef_1
        per_token_loss = per_token_loss + beta * kl  # :3242-3243
    num_items_in_batch = mask.sum()  # :2520 (one process, one generation batch)
    return (per_token_loss * mask).sum() / num_items_in_batch.clamp(min=1.0)  # :3261-3266


def _synthetic(seed: int, b: int = 6, g: int = 4, t: int = 7):
    gen = torch.Generator().manual_seed(seed)
    rewards = (torch.rand(b, g, generator=gen) < 0.4).float()
    rewards[0] = 0.0  # a constant-zero group
    rewards[1] = 1.0  # a constant-one group
    n = b * g
    logp_old = -torch.rand(n, t, generator=gen) * 3
    logp_new = logp_old + 0.4 * torch.randn(n, t, generator=gen)  # some ratios outside [0.8, 1.2]
    logp_ref = logp_old + 0.2 * torch.randn(n, t, generator=gen)
    lengths = torch.randint(1, t + 1, (n,), generator=gen)
    mask = (torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return rewards, logp_new, logp_old, logp_ref, mask


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_advantages_match_trl(seed: int) -> None:
    rewards, *_ = _synthetic(seed)
    skip_unless_implemented(core_grpo.group_advantages, rewards)
    ours = core_grpo.group_advantages(rewards, eps=EPS_ADV).reshape(-1)
    theirs = trl_advantages(rewards)
    assert torch.allclose(ours, theirs, atol=1e-6), (ours - theirs).abs().max()
    assert torch.all(ours[:8] == 0), "constant groups must have zero advantage (implicit filtering)"


def test_group_advantages_eps_is_the_only_free_choice() -> None:
    """With the module default eps=1e-6 the numbers differ from TRL's 1e-4 by ~1e-4 relative — document it."""
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    skip_unless_implemented(core_grpo.group_advantages, r)
    a_trl = trl_advantages(r)[0]
    a_ours = core_grpo.group_advantages(r)[0, 0]
    assert abs(a_ours - a_trl) < 2e-4 and abs(a_ours - a_trl) > 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_loss_matches_trl_pinned_config(seed: int) -> None:
    """beta=0, symmetric clip 0.2, dapo normalization: value AND gradient agree to 1e-6."""
    rewards, logp_new, logp_old, logp_ref, mask = _synthetic(seed)
    skip_unless_implemented(
        core_grpo.grpo_loss, logp_new, logp_old, logp_ref, torch.zeros(rewards.numel()), mask
    )
    adv = trl_advantages(rewards)

    ours_in = logp_new.clone().requires_grad_(True)
    ours, stats = core_grpo.grpo_loss(
        ours_in, logp_old, logp_ref, adv, mask, clip_eps=CLIP_EPS, beta=0.0
    )
    theirs_in = logp_new.clone().requires_grad_(True)
    theirs = trl_loss(theirs_in, logp_old, logp_ref, adv, mask, beta=0.0)

    assert torch.allclose(ours, theirs, atol=1e-6), (ours.item(), theirs.item())
    ours.backward()
    theirs.backward()
    assert torch.allclose(ours_in.grad, theirs_in.grad, atol=1e-6)
    assert 0.0 < stats["clip_fraction"] < 1.0, "synthetic ratios should exercise both clip branches"
    assert stats["n_tokens"] == mask.sum().item()


def test_loss_ratio_one_first_iteration() -> None:
    """num_iterations=1: TRL uses old = new.detach() (:3159) → ratio 1, no clipping, pure REINFORCE."""
    rewards, logp_new, _, logp_ref, mask = _synthetic(7)
    adv = trl_advantages(rewards)
    skip_unless_implemented(core_grpo.grpo_loss, logp_new, logp_new, logp_ref, adv, mask)
    ours, stats = core_grpo.grpo_loss(
        logp_new, logp_new.detach(), logp_ref, adv, mask, clip_eps=CLIP_EPS, beta=0.0
    )
    theirs = trl_loss(logp_new, logp_new.detach(), logp_ref, adv, mask, beta=0.0)
    assert torch.allclose(ours, theirs, atol=1e-6)
    assert stats["clip_fraction"] == 0.0 and abs(stats["mean_ratio"] - 1.0) < 1e-6


def test_kl_term_matches_trl_without_bias_correction_only() -> None:
    """beta>0 (NOT our primary config): core matches TRL with use_bias_correction_kl=False; TRL's
    default multiplies the KL by the ratio (:3197-3198), which core does not. Recorded, not fixed:
    SPEC v1.8 pins beta=0 for every result-bearing run."""
    rewards, logp_new, logp_old, logp_ref, mask = _synthetic(3)
    adv = trl_advantages(rewards)
    skip_unless_implemented(core_grpo.grpo_loss, logp_new, logp_old, logp_ref, adv, mask)
    ours, _ = core_grpo.grpo_loss(
        logp_new, logp_old, logp_ref, adv, mask, clip_eps=CLIP_EPS, beta=0.04
    )
    no_bc = trl_loss(logp_new, logp_old, logp_ref, adv, mask, beta=0.04, bias_correction_kl=False)
    with_bc = trl_loss(logp_new, logp_old, logp_ref, adv, mask, beta=0.04, bias_correction_kl=True)
    assert torch.allclose(ours, no_bc, atol=1e-6)
    assert not torch.allclose(ours, with_bc, atol=1e-6)
