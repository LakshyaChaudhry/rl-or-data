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

from rlordata.train.rft_pipeline_dry import run_dry_pipeline


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(tempfile.mkdtemp(prefix="rlordata_rft_dry_"))
    print(f"dry-run world: {root}")
    return run_dry_pipeline(root, verbose=True)


if __name__ == "__main__":
    sys.exit(main())
