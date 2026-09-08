from __future__ import annotations

from pathlib import Path

import pytest

from rlordata.core.verify import verify
from rlordata.sampling.cap import PROVISIONAL_CAP, CapError, write_cap_yaml
from rlordata.sampling.prompts import format_prompt
from rlordata.sampling.stub_sampler import FakeChatTokenizer, StubSampler
from tests.helpers import small_pool

CAP = 1024


def _sampler(problems, tmp_path: Path, **kw) -> StubSampler:
    cap_path = tmp_path / "cap.yaml"
    if not cap_path.exists():
        write_cap_yaml(cap_path, {"max_completion_tokens": CAP})
    kw.setdefault(
        "mean_len", 100.0
    )  # keep natural lengths far below CAP so cap truncation is negligible
    return StubSampler.from_problems(
        problems,
        model_id="stub/base",
        model_kind="base",
        max_completion_tokens=CAP,
        seed=kw.pop("seed", 3),
        cap_path=cap_path,
        **kw,
    )


def test_shapes_determinism_and_order_independence(tmp_path: Path) -> None:
    probs = small_pool(n=40)
    prompts = [format_prompt(p, "base") for p in probs]
    s = _sampler(probs, tmp_path)
    a = s.sample(prompts, n=4, temperature=1.0)
    assert len(a) == len(prompts) and all(len(row) == 4 for row in a)
    b = s.sample(prompts, n=4, temperature=1.0)
    assert a == b
    # per-prompt results do not depend on batch composition or position
    rev = s.sample(list(reversed(prompts)), n=4, temperature=1.0)
    assert rev == list(reversed(a))
    sub = s.sample(prompts[5:7], n=4, temperature=1.0)
    assert sub == a[5:7]
    # a different seed changes sampled outputs
    s2 = _sampler(probs, tmp_path, seed=4)
    assert s2.sample(prompts, n=4, temperature=1.0) != a
    # greedy is identical across calls and across n
    g1 = s.sample(prompts, n=1, temperature=0.0)
    g2 = s.sample(prompts, n=3, temperature=0.0)
    assert all(row2 == [row1[0]] * 3 for row1, row2 in zip(g1, g2, strict=True))


def test_outcome_fractions_are_controllable(tmp_path: Path) -> None:
    probs = small_pool(n=60)
    prompts = [format_prompt(p, "base") for p in probs]

    def verdicts(sampler):
        out = sampler.sample(prompts, n=2, temperature=1.0)
        return [(verify(p, c.text), c) for p, row in zip(probs, out, strict=True) for c in row]

    all_trunc = _sampler(probs, tmp_path, p_truncated=1.0)
    for v, c in verdicts(all_trunc):
        assert (
            c.truncated
            and c.n_tokens == CAP
            and c.finish_reason == "length"
            and v.extraction_failed
        )
    all_unparse = _sampler(probs, tmp_path, p_truncated=0.0, p_unparseable=1.0)
    for v, c in verdicts(all_unparse):
        assert v.extraction_failed and not c.truncated
    all_correct = _sampler(
        probs, tmp_path, p_truncated=0.0, p_unparseable=0.0, p_correct=1.0, difficulty_spread=False
    )
    for v, c in verdicts(all_correct):
        assert v.reward == 1.0 and not v.extraction_failed and c.n_tokens < CAP
    none_correct = _sampler(
        probs, tmp_path, p_truncated=0.0, p_unparseable=0.0, p_correct=0.0, difficulty_spread=False
    )
    for v, _ in verdicts(none_correct):
        assert v.reward == 0.0 and not v.extraction_failed
    # default spread: some correct, some not, some truncated, some unparseable (over 60 × 8 draws)
    mixed = _sampler(probs, tmp_path)
    out = mixed.sample(prompts, n=8, temperature=1.0)
    vs = [verify(p, c.text) for p, row in zip(probs, out, strict=True) for c in row]
    rewards = [v.reward for v in vs]
    assert 0.2 < sum(rewards) / len(rewards) < 0.8
    assert any(c.truncated for row in out for c in row)
    assert any(v.extraction_failed for v in vs)


def test_lookup_through_instruct_wrapping_and_unknown_problem(tmp_path: Path) -> None:
    probs = small_pool(n=10)
    s = _sampler(
        probs, tmp_path, p_truncated=0.0, p_unparseable=0.0, p_correct=1.0, difficulty_spread=False
    )
    tok = FakeChatTokenizer()
    prompts = [
        format_prompt(p, "instruct", tok, chat_kwargs={"enable_thinking": False}) for p in probs
    ]
    out = s.sample(prompts, n=1, temperature=0.0)
    assert all(verify(p, row[0].text).reward == 1.0 for p, row in zip(probs, out, strict=True))
    assert tok.calls and tok.calls[0] == {"enable_thinking": False}
    unknown = s.sample(["Problem: something the stub never saw\n"], n=1, temperature=0.0)
    assert unknown[0][0].text.endswith("Answer: 0")


def test_stub_obeys_cap_rule_and_prompt_limit(tmp_path: Path) -> None:
    cap_path = tmp_path / "cap.yaml"
    write_cap_yaml(cap_path, {"max_completion_tokens": CAP})
    with pytest.raises(CapError):
        StubSampler("m", "base", max_completion_tokens=CAP + 256, seed=0, cap_path=cap_path)
    with pytest.raises(CapError):
        StubSampler(
            "m",
            "base",
            max_completion_tokens=PROVISIONAL_CAP,
            seed=0,
            cap_path=tmp_path / "missing.yaml",
        )
    s = StubSampler(
        "m",
        "base",
        max_completion_tokens=PROVISIONAL_CAP,
        seed=0,
        allow_provisional_cap=True,
        cap_path=tmp_path / "missing.yaml",
    )
    assert s.cap_is_provisional and s.describe()["sampler"] == "stub"
    with pytest.raises(ValueError):
        s.sample(["word " * 4200], n=1, temperature=0.0)
