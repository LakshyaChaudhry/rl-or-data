"""Print the environment facts every run must record. `python -m rlordata.env_check`."""

from __future__ import annotations

import importlib.metadata as m
import platform
import subprocess
import sys


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def versions(
    pkgs: tuple[str, ...] = ("torch", "transformers", "peft", "trl", "vllm", "numpy"),
) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pkgs:
        try:
            out[p] = m.version(p)
        except m.PackageNotFoundError:
            out[p] = "missing"
    return out


def main() -> None:
    print(f"python   {sys.version.split()[0]}  platform {platform.platform()}")
    print(f"git      {git_sha()}")
    for k, v in versions().items():
        print(f"{k:12s} {v}")
    try:
        import torch

        dev = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        print(f"device   {dev}")
        if dev == "cuda":
            print(f"gpu      {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("device   torch not installed")


if __name__ == "__main__":
    main()
