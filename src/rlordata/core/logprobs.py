"""logprobs.py — the atom of SFT, RFT and GRPO. Hand-written (Laksh).

INTUITION
    A language model is a function from "everything so far" to a probability over the next token.
    Training methods differ only in WHICH tokens' probabilities they push up or down, and by HOW MUCH.
    So the one primitive every method needs is: "given prompt + completion, what log-probability did
    the model assign to each completion token?"

CONNECTION TO WHAT YOU KNOW
    forward pass -> logits [T, V]  (V = vocab size)
    log_softmax over V -> log-probs [T, V]
    gather the column of the token that actually came next -> [T]
    Cross-entropy loss is just  -mean(those gathered values)  over the tokens you care about.

SHAPES (single sequence; batched version is the same with a leading B)
    input_ids      [T_total] = prompt_ids ++ completion_ids
    logits         [T_total, V]
    logits[t] predicts token t+1, so the log-prob of completion token j (at absolute position
    P + j, P = prompt length) comes from logits[P + j - 1].
    output         [T_c]  where T_c = len(completion_ids)

PRECISE
    logp_j = log softmax(logits[P + j - 1])[completion_ids[j]]       for j in 0..T_c-1

TESTS YOU WRITE (tests/core/test_logprobs.py)
    - a tiny random model (or Qwen3-0.6B-Base on CPU) with a 3-token completion: values <= 0, shape [3]
    - agreement with torch.nn.functional.cross_entropy(reduction='none') on the same tokens (sign flip)
    - padding in a batch does not change the un-padded sequence's values
"""

from __future__ import annotations

import torch


def completion_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,  # [B, T_total] (right-padded)
    attention_mask: torch.Tensor,  # [B, T_total]
    completion_mask: torch.Tensor,  # [B, T_total] 1 where the token is a completion token
) -> torch.Tensor:  # [B, T_total] log-probs; entries outside completion_mask are 0
    """Per-token log-probabilities of the completion tokens under `model`.

    Notes for implementation:
      * Run the model once: logits = model(input_ids=..., attention_mask=...).logits  # [B, T, V]
      * Shift: logits[:, :-1] predicts input_ids[:, 1:]
      * Use log_softmax in float32 for stability, then gather.
      * Multiply by the (shifted) completion mask so prompt and pad positions contribute 0.
    """
    assert input_ids.ndim == 2 and input_ids.shape == attention_mask.shape == completion_mask.shape
    raise NotImplementedError("Laksh: implement completion_logprobs")
