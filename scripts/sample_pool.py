"""Draw a random sample of pool problems for hand-review (tasks/01b §6). AGENT-OWNED.

Prints (and optionally writes as Markdown) each sampled problem with its text, answer, and the
per-step intermediate multiset sizes, and asserts the SPEC §4 v1.2 invariants on the sample:
no no-op step, no empty step, ≥3 values at the final op.

    uv run python scripts/sample_pool.py --pool data/pool/pool.jsonl --n 20 --seed 20260908 \
        --out notebook/samples/pool_v1.2_sample20_seed20260908.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlordata.data.generator import meta_path_for, read_jsonl, trace_pipeline
from rlordata.types import Problem


def _step_label(step: dict[str, Any]) -> str:
    params = ",".join(f"{k}={v}" for k, v in sorted(step.items()) if k != "name")
    return f"{step['name']}({params})" if params else step["name"]


def step_sizes(problem: Problem) -> list[tuple[str, int]]:
    """``[(label, size)]`` for the range and each filter / transform, in order."""
    trace = trace_pipeline(problem.pipeline)
    labels = ["range"]
    labels += [_step_label(f) for f in problem.pipeline["filters"]]
    labels += [_step_label(t) for t in problem.pipeline["transforms"]]
    assert len(labels) == len(trace)
    return list(zip(labels, [len(v) for v in trace], strict=True))


def check_invariants(problem: Problem, *, min_values_at_final_op: int = 3) -> None:
    trace = trace_pipeline(problem.pipeline)
    for i in range(1, len(trace)):
        assert trace[i], f"{problem.problem_id}: step {i} emptied the set"
        assert sorted(trace[i - 1]) != sorted(trace[i]), (
            f"{problem.problem_id}: step {i} is a no-op"
        )
    assert len(trace[-1]) >= min_values_at_final_op, (
        f"{problem.problem_id}: only {len(trace[-1])} values at the final op"
    )


def sample(problems: list[Problem], n: int, seed: int) -> list[Problem]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(problems), size=n, replace=False)
    return [problems[int(i)] for i in sorted(idx)]


def format_markdown(
    problems: list[Problem], *, pool: Path, seed: int, meta: dict[str, Any] | None
) -> str:
    lines = [f"# Pool sample — {len(problems)} problems, seed {seed}", ""]
    lines.append(f"- Pool: `{pool}`")
    if meta:
        lines.append(
            f"- Pool seed: {meta.get('seed')}  ·  config hash: `{meta.get('config_hash')}`"
        )
        lines.append(
            f"- Generated at: {meta.get('generated_at')}  ·  git SHA: `{meta.get('git_sha')}`"
        )
        cells = meta.get("cells", [])
        if cells:
            cell_str = ", ".join(
                f"{c['range_scale']}{c['total_steps']}={c['accepted']}" for c in cells
            )
            lines.append(f"- Cell counts: {cell_str}")
    lines.append(
        "- Invariants checked on every sampled problem: no no-op step, no empty step, ≥3 values at the final op."
    )
    lines.append("")
    for i, p in enumerate(problems, 1):
        sizes = " → ".join(f"{label}: {size}" for label, size in step_sizes(p))
        lines.append(f"## {i}. `{p.problem_id[:12]}` — {p.range_scale}, {p.total_steps} steps")
        lines.append("")
        lines.append(f"> {p.text}")
        lines.append("")
        lines.append(f"- **Answer:** {p.answer}")
        lines.append(f"- **Set sizes:** {sizes}")
        lines.append(
            f"- Pipeline: `{json.dumps(p.pipeline, sort_keys=True, separators=(',', ':'))}`"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pool", default="data/pool/pool.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", default=None, help="write Markdown here (also printed)")
    args = parser.parse_args(argv)

    pool = Path(args.pool)
    problems = read_jsonl(pool)
    meta_file = meta_path_for(pool)
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None
    chosen = sample(problems, args.n, args.seed)
    for p in chosen:
        check_invariants(p)
    text = format_markdown(chosen, pool=pool, seed=args.seed, meta=meta)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
