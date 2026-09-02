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
    # rewards [1,0,0,1]: mean 0.5, std (ddof=?) -> decide, document, match TRL. Fill EXPECTED by hand.
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    skip_unless_implemented(grpo_mod.group_advantages, r)
    expected = None  # TODO(Laksh): e.g. torch.tensor([[a, -a, -a, a]]) with a computed on paper
    assert expected is not None, "fill in the hand-computed expectation"
    assert torch.allclose(grpo_mod.group_advantages(r), expected, atol=1e-5)


def test_ratio_one_reduces_to_reinforce():
    n, t = 4, 5
    logp = torch.randn(n, t)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(n, t)
    skip_unless_implemented(grpo_mod.grpo_loss, logp, logp, logp, adv, mask)
    loss, stats = grpo_mod.grpo_loss(logp, logp.clone(), logp.clone(), adv, mask, beta=0.0)
    expected = None  # TODO(Laksh): -mean over all tokens of adv broadcast (ratio == 1, kl == 0)
    assert expected is not None
    assert torch.allclose(loss, expected, atol=1e-6)
