"""The tasks/03 pipeline end to end on a tiny world: shared by ``scripts/rft_dry_run.py`` and the
integration test. Stub sampler + a 2-layer random Qwen3 with the real tokenizer. Never a result.
"""

from __future__ import annotations

import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rlordata.cli import main as cli_main
from rlordata.sampling.cap import PROVISIONAL_CAP, write_cap_yaml

TOKENIZER_ID = "Qwen/Qwen3-4B"  # same tokenizer as Qwen3-4B-Base; cached locally for tests


def core_implemented() -> bool:
    """Both hand-written functions the trainer calls (tasks/03 prerequisite 2)."""
    import torch

    from rlordata.core.rft_select import rft_select
    from rlordata.core.sft_loss import sft_loss

    try:
        sft_loss(torch.zeros(1, 2), torch.ones(1, 2))
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001 - implemented (behaviour is Laksh's tests' business)
        pass
    try:
        rft_select([], {}, mode="all")
    except NotImplementedError:
        return False
    except Exception:  # noqa: BLE001
        pass
    return True


def tokenizer_available() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TOKENIZER_ID)
        return True
    except Exception:  # noqa: BLE001 - not cached / offline
        return False


def write_tiny_model(path: Path, *, seed: int = 0) -> Path:
    """A 2-layer random Qwen3 (real vocab, hidden 64) saved with the tokenizer; loads by path."""
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(str(path), safe_serialization=True)
    tok.save_pretrained(str(path))
    return path


def make_rft_world(root: Path, *, n_total: int = 12) -> dict[str, Any]:
    """Tiny pool → stub tiering → cap → stub draw → tiny model → arm configs. Returns the paths."""
    from tests.helpers import make_world

    world = make_world(root)
    assert world.tier_with_stub() == 0
    write_cap_yaml(
        world.cap_path, {"max_completion_tokens": PROVISIONAL_CAP, "note": "dry-run cap"}
    )
    model_dir = write_tiny_model(root / "tiny_model")
    configs = {}
    for arm in ("easy", "mixed", "curated"):
        with (Path("configs/rft") / f"{arm}.yaml").open(encoding="utf-8") as f:
            cfg = deepcopy(yaml.safe_load(f))
        cfg.update(
            {
                "model_id": str(model_dir),
                "cap_yaml": str(world.cap_path),
                "splits_dir": str(world.splits_dir),
                "samples_dir": str(root / "data" / "samples"),
                "tiering_samples": str(world.samples_path),
                "output_dir": str(root / "runs" / "rft"),
                "base_eval_dir": str(root / "runs" / "eval_base"),
                "transfer_yaml": str(root / "transfer.yaml"),
                "samples_per_prompt": n_total,
                "pool": str(world.pool_path),
                "ood_path": str(world.ood_path),
                "micro_batch_size": 4,
                "gradient_checkpointing": False,
                "dtype": "float32",
            }
        )
        p = root / "configs" / f"rft_{arm}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        configs[arm] = p
    with (root / "transfer.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"splits": []}, f)
    return {
        "world": world,
        "configs": configs,
        "model_dir": model_dir,
        "root": root,
        "n_total": n_total,
    }


def run_dry_pipeline(root: Path, *, verbose: bool = False, arm: str = "mixed") -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not tokenizer_available():
        print(f"{TOKENIZER_ID} tokenizer not cached; cannot dry-run")
        return 3
    if not core_implemented():
        print("core.sft_loss / core.rft_select are not implemented yet (Laksh); dry run stops here")
        return 4
    w = make_rft_world(root)
    cfg = str(w["configs"][arm])
    rc = cli_main(
        ["rft", "--config", cfg, "--stage", "draw", "--stub", "--n-total", str(w["n_total"])]
    )
    if rc != 0:
        return rc
    rc = cli_main(["rft", "--config", cfg, "--stage", "select"])
    if rc != 0:
        return rc
    from scripts.rft_sweep import main as sweep_main

    rc = sweep_main(
        ["--config", cfg, "--stub", "--allow-cpu", "--only", "1e-4:2,5e-5:2", "--n-problems", "6"]
    )
    if rc != 0:
        return rc
    from scripts.rft_finals import main as finals_main

    rc = finals_main(
        ["--config", cfg, "--stub", "--allow-cpu", "--seeds", "1,2", "--n-problems", "6"]
    )
    if verbose:
        print(
            (
                Path(w["configs"][arm]).parent.parent / "runs" / "rft" / f"rft_{arm}" / "report.md"
            ).read_text()
        )
    shutil.rmtree(root, ignore_errors=not verbose)
    return rc
