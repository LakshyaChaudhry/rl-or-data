"""Procedural counting-problem generator (SPEC §4). AGENT-OWNED; implemented in tasks/01.

Contract (frozen):
    generate_pool(config: dict, seed: int) -> list[Problem]
    execute_pipeline(pipeline: dict) -> int            # the ground-truth executor
    render(pipeline: dict, rng) -> str                 # NL rendering with >=3 paraphrases per operator
    canonical_id(pipeline: dict) -> str                # sha256 of canonical JSON
    write_jsonl(problems, path) / read_jsonl(path)
    cli_main(args) -> int                              # `rlordata gen --config ...`

Tests (tests/data/test_generator.py) include an INDEPENDENT reference executor written separately
from execute_pipeline; both must agree on every generated problem.
"""

from __future__ import annotations


def cli_main(args) -> int:  # noqa: ANN001
    raise NotImplementedError("tasks/01")
