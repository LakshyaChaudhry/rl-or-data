"""Test CASES authored by Laksh. Compute expected values by hand BEFORE encoding them."""

import pytest

torch = pytest.importorskip("torch")

from rlordata.core import grpo as grpo_mod  # noqa: E402
from tests.conftest import skip_unless_implemented  # noqa: E402

pytestmark = pytest.mark.core


def test_constant_group_has_zero_advantage():
    r = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    skip_unless_implemented(grpo_mod.group_advantages, r)
    assert torch.allclose(grpo_mod.group_advantages(r), torch.zeros_like(r))


def test_mixed_group_by_hand():
    # [1,0,0,1]: mean 0.5; unbiased std = sqrt(1/3); A = ±0.5/sqrt(1/3) = ±sqrt(3)/2
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    skip_unless_implemented(grpo_mod.group_advantages, r)
    a = 0.5 / (1.0 / 3.0) ** 0.5  # sqrt(3)/2
    expected = torch.tensor([[a, -a, -a, a]])
    assert torch.allclose(grpo_mod.group_advantages(r), expected, atol=1e-5)


def test_ratio_one_reduces_to_reinforce():
    n, t = 4, 5
    logp = torch.randn(n, t)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(n, t)
    skip_unless_implemented(grpo_mod.grpo_loss, logp, logp, logp, adv, mask)
    loss, stats = grpo_mod.grpo_loss(logp, logp.clone(), logp.clone(), adv, mask, beta=0.0)
    expected = -(adv[:, None] * mask).sum() / mask.sum()  # ratio=1, kl=0
    assert torch.allclose(loss, expected, atol=1e-6)


def test_zero_advantage_zero_loss_and_grad():
    n, t = 2, 3
    logp = torch.randn(n, t, requires_grad=True)
    adv = torch.zeros(n)
    mask = torch.ones(n, t)
    loss, _ = grpo_mod.grpo_loss(logp, logp.detach(), logp.detach(), adv, mask, beta=0.0)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-8)
    loss.backward()
    assert torch.allclose(logp.grad, torch.zeros_like(logp))


def test_kl_nonnegative_and_zero_when_equal():
    n, t = 2, 4
    logp = torch.randn(n, t)
    adv = torch.ones(n)
    mask = torch.ones(n, t)
    _, stats = grpo_mod.grpo_loss(logp, logp, logp, adv, mask, beta=1.0)
    assert stats["mean_kl"] == 0.0 or abs(stats["mean_kl"]) < 1e-6
