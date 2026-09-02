"""Mechanical checks that run before any result is reported (CLAUDE.md 'Scientific standards').

Checks to implement (tasks/02 onward):
  - split disjointness (problem_id and structure) across train/val/test/ood
  - identical prompt template and cap across all runs being compared (compare resolved configs)
  - truncation and extraction-failure rates below thresholds; flag otherwise
  - LoRA adapter actually loaded (parameter count delta, adapter hash)
  - tokenizer identity between vLLM and trainer
  - eval checkpoint == final checkpoint (no early stopping on test)
"""

from __future__ import annotations
