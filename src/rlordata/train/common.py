"""Shared training plumbing for the RFT (tasks/03) and GRPO (tasks/04) arms. AGENT-OWNED.

What lives here and nowhere else:

- config loading: an arm config (``configs/rft/*.yaml``) + the locked training constants
  (``configs/locked/training.yaml``) + the locked cap (``configs/locked/cap.yaml``)
- seeding (``torch``, ``numpy``, ``random``) from the single run ``seed``
- the PEFT ``LoraConfig`` built from ``training.yaml`` (never from literals in a trainer)
- tokenizer fingerprinting, so the trainer's tokenizer can be asserted equal to vLLM's
- adapter merge into the base model for the single generation path (``VLLMSampler``)
- cost estimate / actual printouts (CLAUDE.md "GPU etiquette") and the artifact sync at exit

Nothing here computes a loss or selects data; those are ``core/`` functions called by the
trainers.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.artifacts import sync_run_dir
from rlordata.envfile import gpu_rate_usd_per_hour
from rlordata.run_dir import format_cost
from rlordata.sampling.cap import DEFAULT_CAP_PATH, load_locked_cap

DEFAULT_TRAINING_PATH = Path("configs/locked/training.yaml")
# Rough LoRA-SFT throughput of a 4B model on one H100 with gradient checkpointing; only for the
# START estimate. The END printout uses the measured wall-clock.
DEFAULT_EST_TRAIN_TOKENS_PER_SEC = 4000.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_arm_config(path: str | Path) -> dict[str, Any]:
    """Arm config with the locked training constants and the locked cap resolved into it.

    Returns a dict with keys ``arm``, ``model_id``, ``data_condition``, ``seed``, ``select``,
    ``training`` (the full locked block), ``max_completion_tokens`` and the paths used.
    Raises if the cap is not locked: no training run may start before Phase 1 (tasks/03 prereq 1).
    """
    cfg = load_yaml(path)
    training_path = Path(cfg.get("training", DEFAULT_TRAINING_PATH))
    cap_path = Path(cfg.get("cap_yaml", DEFAULT_CAP_PATH))
    cap = load_locked_cap(cap_path)
    if cap is None:
        raise RuntimeError(
            f"{cap_path} does not exist: the token cap is not locked, so no training run may start "
            "(SPEC §7, tasks/03 prerequisite 1)"
        )
    resolved = dict(cfg)
    resolved["source_config"] = str(path)
    resolved["training"] = load_yaml(training_path)
    resolved["training_yaml"] = str(training_path)
    resolved["cap_yaml"] = str(cap_path)
    resolved["max_completion_tokens"] = int(cap)
    return resolved


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + every CUDA device) from one integer."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is an optional extra locally
        pass


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def lora_kwargs_from_training(training: dict[str, Any]) -> dict[str, Any]:
    """The ``LoraConfig`` fields from the locked ``training.yaml`` (SPEC §9), as plain values."""
    lora = training["lora"]
    return {
        "r": int(lora["r"]),
        "lora_alpha": int(lora["alpha"]),
        "lora_dropout": float(lora["dropout"]),
        "target_modules": list(lora["target_modules"]),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def lora_config_from_training(training: dict[str, Any]):  # noqa: ANN201 — peft.LoraConfig
    from peft import LoraConfig

    return LoraConfig(**lora_kwargs_from_training(training))


# ---------------------------------------------------------------------------
# Tokenizer fingerprint
# ---------------------------------------------------------------------------


def tokenizer_hash(tokenizer: Any) -> str:
    """sha256 over the vocabulary, the added tokens and the special-token map.

    The same function is applied to the trainer's ``AutoTokenizer`` and to vLLM's
    ``LLM.get_tokenizer()`` so a mismatch (different revision, different added tokens) is caught
    before any number is written (tasks/03 §3).
    """
    vocab = tokenizer.get_vocab()
    payload = {
        "vocab": sorted(vocab.items(), key=lambda kv: (kv[1], kv[0])),
        "added": sorted(str(t) for t in getattr(tokenizer, "get_added_vocab", dict)()),
        "special": {
            k: str(v) for k, v in sorted(getattr(tokenizer, "special_tokens_map", {}).items())
        },
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Adapter merge (for eval through vLLM)
# ---------------------------------------------------------------------------


def merge_adapter(
    model_id: str,
    adapter_dir: str | Path,
    out_dir: str | Path,
    *,
    dtype: str = "bfloat16",
) -> Path:
    """Merge a LoRA adapter into ``model_id`` and save the merged weights + tokenizer to ``out_dir``.

    NOT used by any eval any more (notebook 2026-09-14): in bf16, ``W + ΔW`` rounds ~90 % of ΔW
    entries back to ``W`` (|ΔW| ≪ half-ulp), so a bf16-merged model is mostly the base model.
    Evals apply the adapter as a native vLLM LoRA (``VLLMSampler(lora_path=…)``). Kept only for
    an fp32 merge where a standalone checkpoint is explicitly wanted.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, dtype)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(str(out))
    del merged, model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def remove_dir(path: str | Path) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)


# ---------------------------------------------------------------------------
# Cost + sync
# ---------------------------------------------------------------------------


def estimate_train_gpu_hours(
    training_tokens: int, tokens_per_s: float = DEFAULT_EST_TRAIN_TOKENS_PER_SEC
) -> float:
    return float(training_tokens) / max(tokens_per_s, 1e-9) / 3600.0


def print_cost(label: str, gpu_hours: float) -> str:
    line = f"[cost] {label}: {format_cost(gpu_hours, gpu_rate_usd_per_hour())}"
    print(line, flush=True)
    return line


def sync_run(run_dir: str | Path) -> str | None:
    """Sync a run directory to the artifact store; a warned no-op when the store is unset."""
    return sync_run_dir(run_dir, quiet=False)


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def make_read_only(path: str | Path) -> None:
    """``chmod 444`` (tasks/03 acceptance: the 192-sample files are read-only after the draw)."""
    os.chmod(path, 0o444)
