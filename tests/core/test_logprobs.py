import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from rlordata.core.logprobs import completion_logprobs

pytestmark = pytest.mark.core


def _tiny_model():
    cfg = GPT2Config(n_layer=2, n_embd=64, n_head=2, n_positions=64, vocab_size=128)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


def test_shape_and_nonpositive():
    model = _tiny_model()
    B, T, P = 2, 8, 5  # prompt length 5, completion 3
    input_ids = torch.randint(0, 128, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    completion_mask = torch.zeros(B, T, dtype=torch.long)
    completion_mask[:, P:] = 1

    out = completion_logprobs(model, input_ids, attention_mask, completion_mask)
    assert out.shape == (B, T)
    # prompt + first position stay 0; completion positions (except index 0) <= 0
    assert torch.all(out[:, :P] == 0)
    assert torch.all(out[:, P:] <= 0)


def test_matches_cross_entropy():
    model = _tiny_model()
    torch.manual_seed(0)
    B, T, P = 1, 6, 3
    input_ids = torch.randint(0, 128, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    completion_mask = torch.zeros(B, T, dtype=torch.long)
    completion_mask[:, P:] = 1

    out = completion_logprobs(model, input_ids, attention_mask, completion_mask)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # CE at positions that predict completion tokens P..T-1 → logits indices P-1..T-2
    ce = F.cross_entropy(
        logits[0, P - 1 : T - 1].float(),
        input_ids[0, P:T],
        reduction="none",
    )
    assert torch.allclose(out[0, P:T], -ce, atol=1e-5)
