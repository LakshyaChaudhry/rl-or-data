import pytest
import torch

from rlordata.core.sft_loss import sft_loss

pytestmark = pytest.mark.core


def test_all_masked_raises():
    logp = torch.randn(2, 4)
    mask = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="all-masked"):
        sft_loss(logp, mask)


def test_equals_negative_mean_of_masked():
    torch.manual_seed(0)
    logp = torch.randn(2, 5)
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    loss = sft_loss(logp, mask)
    expected = -logp[mask.bool()].mean()
    assert torch.allclose(loss, expected)


def test_invariant_to_right_padding():
    torch.manual_seed(1)
    logp = torch.tensor([[-0.5, -1.0, -2.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    base = sft_loss(logp, mask)

    logp_pad = torch.nn.functional.pad(logp, (0, 3))  # [1, 6]
    mask_pad = torch.nn.functional.pad(mask, (0, 3))
    assert torch.allclose(sft_loss(logp_pad, mask_pad), base)
