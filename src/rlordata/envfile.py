"""Minimal ``.env`` loader (no third-party dependency). AGENT-OWNED; tasks/02a.

``uv run`` does not load ``.env`` by itself. Every CLI entry point calls :func:`load_env` once so
``RLORDATA_ARTIFACTS``, ``RLORDATA_GPU_RATE_USD_PER_HOUR``, ``HF_TOKEN`` and ``LAMBDA_API_KEY``
behave the same whether they were exported in the shell or written to ``.env``. Existing
environment variables always win; the file never overrides them.
"""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Ignores blanks, ``#`` comments and inline ``# ...`` trailers."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value and value[0] in "\"'" and value[-1] == value[0] and len(value) >= 2:
            value = value[1:-1]
        elif value.startswith("#"):
            # `KEY=   # comment` — blank value, the rest is a comment.
            value = ""
        else:
            # Unquoted: drop an inline comment.
            hash_pos = value.find(" #")
            if hash_pos != -1:
                value = value[:hash_pos].rstrip()
        if key:
            out[key] = value
    return out


def load_env(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load ``path`` into ``os.environ`` (missing keys only unless ``override``). Returns what was set."""
    p = Path(path)
    if not p.exists():
        return {}
    parsed = parse_env_text(p.read_text(encoding="utf-8"))
    applied: dict[str, str] = {}
    for key, value in parsed.items():
        if override or key not in os.environ:
            if value == "" and key not in os.environ:
                # Empty placeholder lines in .env.example style files: leave unset.
                continue
            os.environ[key] = value
            applied[key] = value
    return applied


def gpu_rate_usd_per_hour() -> float | None:
    """``RLORDATA_GPU_RATE_USD_PER_HOUR`` as a float, or None when unset / unparsable."""
    raw = os.environ.get("RLORDATA_GPU_RATE_USD_PER_HOUR", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
