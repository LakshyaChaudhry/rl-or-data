"""Local end-to-end dry run of tasks/03 with the stub sampler and a tiny random model. Never a result.

Builds a tiny world (pool, stub tiering, cap), draws 12 samples per prompt with the stub sampler,
writes a 2-layer random Qwen3 with the real Qwen tokenizer, runs a 2-config sweep on CPU, the
"finals" for two seeds, and the per-arm report. Requires ``core.sft_loss`` and ``core.rft_select``
to be implemented (it stops and says so otherwise). ``make rft-dry``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: tests.helpers, scripts.*

from rlordata.train.rft_pipeline_dry import run_dry_pipeline, run_iter_dry_pipeline  # noqa: E402


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(tempfile.mkdtemp(prefix="rlordata_rft_dry_"))
    print(f"dry-run world: {root}")
    if "--iterated" in sys.argv[1:]:  # tasks/06b: `make iter-rft-dry`
        return run_iter_dry_pipeline(root, verbose=True)
    return run_dry_pipeline(root, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
