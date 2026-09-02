"""HAND-WRITTEN CORE (Laksh). Agents: read and import only. See CLAUDE.md.

Build order (matches the fundamentals track):
  1. verify.py        — pure Python; no ML needed. Week 1.
  2. logprobs.py      — tokenizer -> forward -> logits -> log_softmax -> gather. Week 1.
  3. evaluate.py      — accuracy, pass@k, bootstrap CI. Week 1.
  4. sft_loss.py      — cross-entropy with completion-only masking. Week 2.
  5. rft_select.py    — rejection sampling + curation. Week 2.
  6. grpo.py          — advantages + clipped loss + KL. Week 3.
Each file's docstring is a mini-lesson: intuition -> connection to what you know -> shapes -> the formula.
"""
