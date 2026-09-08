"""Regression test for setup/idle_shutdown.sh install: the cron entry must survive an empty root crontab.

`grep -v` returns 1 when its input is empty; under `set -euo pipefail` that used to abort the subshell
before the guard line was echoed, so `crontab -` installed an EMPTY crontab and the box had no idle
guard at all (observed 2026-09-08 on the Lambda H100). Runs the extracted pipeline with a fake `sudo`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "setup" / "idle_shutdown.sh"
GUARD = "* * * * * /usr/local/bin/gpu_idle_check.sh"


def _crontab_pipeline() -> str:
    lines = [ln for ln in SCRIPT.read_text().splitlines() if "| sudo crontab -" in ln]
    assert len(lines) == 1, lines
    return lines[0].strip()


def _run_install_pipeline(tmp_path: Path, existing: str) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    store = tmp_path / "crontab.txt"
    store.write_text(existing)
    # `sudo crontab -l` prints the store (exit 1 when empty, like real crontab); `sudo crontab -` replaces it.
    (fake_bin / "sudo").write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1 $2" = "crontab -l" ]; then [ -s "$CRONSTORE" ] || exit 1; cat "$CRONSTORE"; exit 0; fi\n'
        'if [ "$1 $2" = "crontab -" ]; then t=$(mktemp); cat > "$t"; mv "$t" "$CRONSTORE"; exit 0; fi\n'
        'echo "unexpected sudo $*" >&2; exit 99\n'
    )
    (fake_bin / "sudo").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "CRONSTORE": str(store)}
    subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + _crontab_pipeline()], env=env, check=True, timeout=30
    )
    return store.read_text()


@pytest.mark.parametrize("existing", ["", "0 3 * * * /usr/bin/other-job\n", GUARD + "\n"])
def test_install_writes_guard_entry(tmp_path: Path, existing: str) -> None:
    result = _run_install_pipeline(tmp_path, existing)
    entries = [ln for ln in result.splitlines() if ln.strip()]
    assert entries.count(GUARD) == 1, result
    for ln in existing.splitlines():
        if ln.strip() and "gpu_idle_check" not in ln:
            assert ln in entries, result  # unrelated entries preserved


def test_pipeline_is_guarded_against_empty_grep() -> None:
    assert re.search(r"grep -v gpu_idle_check \|\| true", _crontab_pipeline())
