import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.core

# TODO(Laksh): build a tiny random causal LM (transformers GPT2Config with 2 layers) or load
# Qwen/Qwen3-0.6B-Base on CPU; check shape [B,T], values <= 0, agreement with F.cross_entropy,
# and padding invariance. Keep it fast (< 5 s).
