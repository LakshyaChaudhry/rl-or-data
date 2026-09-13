"""GRPO arm via TRL GRPOTrainer + colocated vLLM (SPEC §8, §9 / v1.8). AGENT-OWNED; tasks/04.

Every GRPOConfig knob is set explicitly from ``configs/locked/training.yaml`` + ``cap.yaml``.
Reward = ``core.verify`` (or C1/C2 wrappers in ``train/rewards.py``). Never fall back to
``model.generate()`` if vLLM/TRL is broken — stop and report.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rlordata.data.generator import read_jsonl
from rlordata.run_dir import finish_run, start_run
from rlordata.sampling.prompts import TEMPLATE, format_prompt
from rlordata.train.callbacks import DiagnosticsCallback
from rlordata.train.common import (
    estimate_train_gpu_hours,
    load_arm_config,
    lora_config_from_training,
    print_cost,
    seed_everything,
    sync_run,
    tokenizer_hash,
    write_json,
)
from rlordata.train.rewards import RewardRecorder, load_reward_records, make_reward_fn
from rlordata.types import Problem

# SPEC §7 locked max prompt tokens (task §2 lists 1024; SPEC wins → 4096).
MAX_PROMPT_LENGTH = 4096
GENERATION_BATCH_SIZE = 64  # 8 prompts × 8 generations
VAL_CHECKPOINTS = (100, 200, 300)
DEFAULT_OUTPUT_DIR = Path("runs/grpo")


def arm_run_name(cfg: dict[str, Any], seed: int) -> str:
    """Stable run directory name, e.g. ``grpo_mixed_s1`` / ``grpo_random_reward_s1``."""
    arm = str(cfg["arm"])
    dc = str(cfg.get("data_condition", ""))
    if arm == "grpo":
        if dc.endswith("easy_100") or dc == "train_easy_100":
            tag = "grpo_easy"
        elif dc.endswith("mixed_100") or dc == "train_mixed_100":
            tag = "grpo_mixed"
        elif dc == "train_curated" or dc.endswith("curated"):
            tag = "grpo_curated"
        else:
            tag = f"grpo_{dc}"
    else:
        tag = arm
    return f"{tag}_s{seed}"


def load_train_problems(cfg: dict[str, Any], splits_dir: Path) -> list[Problem]:
    dc = str(cfg["data_condition"])
    path = splits_dir / f"{dc}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing split {path} (tasks/04 prereq: persisted splits)")
    problems = read_jsonl(path)
    if dc == "train_curated" and not problems:
        raise SystemExit(f"{path} is empty — refuse to re-derive curated set")
    return problems


def build_dataset(problems: list[Problem]) -> Any:
    """HF Dataset with plain-text prompts (no chat template) + reward kwargs columns."""
    from datasets import Dataset

    rows = {
        "prompt": [format_prompt(p, "base") for p in problems],
        "problem_id": [p.problem_id for p in problems],
        "answer": [int(p.answer) for p in problems],
        "tier": [str(p.tier) for p in problems],
    }
    # Protocol guard: rendered prompt must be TEMPLATE.format(problem_text=...).
    prefix = TEMPLATE.split("{problem_text}")[0]
    for i, prompt in enumerate(rows["prompt"]):
        if not prompt.startswith(prefix):
            raise RuntimeError(
                f"prompt drift on {problems[i].problem_id[:12]}: missing TEMPLATE prefix"
            )
        expected = TEMPLATE.format(problem_text=problems[i].text)
        if prompt != expected:
            raise RuntimeError(
                f"prompt bytes differ from format_prompt for {problems[i].problem_id[:12]}"
            )
    return Dataset.from_dict(rows)


def prompt_bytes_hash(problems: list[Problem]) -> dict[str, str]:
    return {
        p.problem_id: hashlib.sha256(format_prompt(p, "base").encode("utf-8")).hexdigest()
        for p in problems
    }


def build_grpo_config(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    seed: int,
) -> Any:
    """Every GRPOConfig field explicit from locked training + cap (tasks/04 §2)."""
    from trl import GRPOConfig

    g = cfg["training"]["grpo"]
    vllm = cfg.get("vllm") or {}
    steps = int(g["steps"])
    gens = int(g["generations_per_prompt"])
    prompts = int(g["prompts_per_step"])
    gen_batch = prompts * gens
    if gen_batch != GENERATION_BATCH_SIZE:
        raise SystemExit(f"generation_batch_size must be {GENERATION_BATCH_SIZE}, got {gen_batch}")
    if steps * gen_batch != int(g["total_sampled_completions"]):
        raise SystemExit(
            f"max_steps × generation_batch_size = {steps * gen_batch} != "
            f"total_sampled_completions {g['total_sampled_completions']}"
        )

    # 8 completions per device micro-batch × 8 grad-accum steps = 64 = 8 prompts × 8 gens.
    per_device = gens
    grad_accum = prompts
    if per_device * grad_accum != gen_batch:
        raise SystemExit("per_device_train_batch_size × gradient_accumulation_steps must equal 64")

    kwargs: dict[str, Any] = dict(
        output_dir=str(run_dir / "checkpoints"),
        num_generations=gens,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        generation_batch_size=gen_batch,
        max_steps=steps,
        learning_rate=float(g["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(cfg["training"]["schedule"]["warmup_ratio"]),
        max_grad_norm=float(cfg["training"]["optimizer"]["grad_clip"]),
        weight_decay=float(cfg["training"]["optimizer"]["weight_decay"]),
        bf16=True,
        temperature=float(g["temperature"]),
        top_p=float(g["top_p"]),
        max_completion_length=int(cfg["max_completion_tokens"]),
        max_prompt_length=MAX_PROMPT_LENGTH,
        beta=float(g["beta_kl"]),
        epsilon=float(g["clip_eps"]),
        # epsilon_high unset → symmetric clip (SPEC v1.8)
        scale_rewards=str(g.get("scale_rewards", "group")),
        loss_type=str(g.get("loss_type", "dapo")),
        num_iterations=int(g.get("num_iterations", 1)),
        use_vllm=True,
        vllm_mode=str(vllm.get("mode", "colocate")),
        vllm_gpu_memory_utilization=float(vllm.get("gpu_memory_utilization", 0.3)),
        logging_steps=1,
        save_steps=int(cfg["training"].get("checkpoint_every_steps", 25)),
        save_total_limit=None,
        log_completions=True,
        seed=int(seed),
        report_to=[],
        remove_unused_columns=False,
    )
    return GRPOConfig(**kwargs)


def _package_versions() -> dict[str, str]:
    import importlib.metadata as m

    out: dict[str, str] = {}
    for p in ("torch", "transformers", "peft", "trl", "vllm", "accelerate", "datasets"):
        try:
            out[p] = m.version(p)
        except m.PackageNotFoundError:
            out[p] = "MISSING"
    return out


def _assert_trl_vllm() -> dict[str, str]:
    versions = _package_versions()
    if versions.get("trl") == "MISSING" or versions.get("vllm") == "MISSING":
        raise SystemExit(
            "TRL or vLLM missing. Install the gpu extra (setup/setup_gpu.sh). "
            "Refusing to fall back to model.generate() (tasks/04 §2)."
        )
    try:
        import trl  # noqa: F401
        import vllm  # noqa: F401
    except Exception as e:  # noqa: BLE001 — surface the import error as a hard stop
        raise SystemExit(
            f"TRL/vLLM import failed ({e}). Stop and fix the version pair; "
            "do not fall back to model.generate()."
        ) from e
    return versions


def train_grpo(cfg: dict[str, Any], args: Any) -> int:
    """``rlordata grpo --config …`` — one result-bearing run."""
    dry = bool(getattr(args, "dry_run", False))
    versions = {"dry_run": "true"} if dry else _assert_trl_vllm()
    seed = int(args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 1))
    seed_everything(seed)
    splits_dir = Path(getattr(args, "splits_dir", None) or cfg.get("splits_dir", "data/splits"))
    output_dir = Path(
        getattr(args, "output_dir", None) or cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    run_name = arm_run_name(cfg, seed)
    run_dir = Path(getattr(args, "run_dir", None) or (output_dir / run_name))
    if (run_dir / "budgets.json").exists() and not getattr(args, "force", False):
        print(f"[grpo] {run_dir} already finished — skipping (use --force)")
        return 0

    problems = load_train_problems(cfg, splits_dir)
    dataset = build_dataset(problems)
    prompt_hashes = prompt_bytes_hash(problems)

    g = cfg["training"]["grpo"]
    gen_batch = int(g["prompts_per_step"]) * int(g["generations_per_prompt"])
    if gen_batch != GENERATION_BATCH_SIZE or int(g["steps"]) * gen_batch != int(
        g["total_sampled_completions"]
    ):
        raise SystemExit(
            f"budget assert failed: steps×batch={int(g['steps']) * gen_batch}, "
            f"locked total={g['total_sampled_completions']}"
        )
    resolved = dict(cfg)
    resolved.update(
        {
            "seed": seed,
            "run_name": run_name,
            "n_prompts": len(problems),
            "prompt_template": TEMPLATE,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "generation_batch_size": GENERATION_BATCH_SIZE,
            "package_versions": versions,
            "prompt_sha256_by_problem_id": prompt_hashes,
            "grpo_locked": g,
        }
    )

    handle = start_run(
        run_dir,
        resolved,
        run_id=run_name,
        extra_meta={"arm": cfg["arm"], "seed": seed, "data_condition": cfg["data_condition"]},
    )

    # Cost estimate: 19,200 completions × cap tokens (worst case) at rough tok/s.
    worst_tokens = int(g["total_sampled_completions"]) * int(cfg["max_completion_tokens"])
    print_cost(
        "grpo start estimate (worst-case tokens at cap)", estimate_train_gpu_hours(worst_tokens)
    )

    if dry:
        write_json(
            run_dir / "dry_run.json",
            {
                "n_prompts": len(problems),
                "generation_batch_size": GENERATION_BATCH_SIZE,
                "max_steps": g["steps"],
                "total_sampled_completions": g["total_sampled_completions"],
                "reward": cfg.get("reward"),
                "n_dataset_rows": len(dataset),
            },
        )
        finish_run(handle, status="dry_run")
        print("[grpo] dry-run only — no training")
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer

    recorder = RewardRecorder(run_dir / "reward_records.jsonl")
    reward_name = str(cfg.get("reward", "verify_binary"))
    reward_fn = make_reward_fn(
        reward_name,
        recorder=recorder,
        seed=seed,
        max_completion_length=int(cfg["max_completion_tokens"]),
    )

    grpo_args = build_grpo_config(cfg, run_dir=run_dir, seed=seed)
    try:
        cfg_dump = grpo_args.to_dict()
    except Exception:
        cfg_dump = {k: getattr(grpo_args, k, None) for k in dir(grpo_args) if not k.startswith("_")}
    write_json(
        run_dir / "grpo_config.json",
        {k: cfg_dump[k] for k in sorted(cfg_dump) if not callable(cfg_dump[k])},
    )

    model_id = str(cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_tok_hash = tokenizer_hash(tokenizer)
    write_json(run_dir / "tokenizer_hash.json", {"trainer": train_tok_hash})

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    peft_config = lora_config_from_training(cfg["training"])

    callback = DiagnosticsCallback(
        run_dir=run_dir,
        recorder=recorder,
        num_generations=int(g["generations_per_prompt"]),
        est_gpu_hours=estimate_train_gpu_hours(worst_tokens),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback],
    )
    # Reference = adapter-disabled base under PEFT (TRL); no separate ref model when beta=0.
    if getattr(trainer, "ref_model", None) is not None and float(g["beta_kl"]) == 0.0:
        raise SystemExit("ref_model is set despite beta=0; unexpected TRL behaviour — stop")
    if getattr(trainer, "peft_config", None) is None and peft_config is not None:
        # Some TRL versions stash PEFT on the model only.
        from peft import PeftModel

        if not isinstance(trainer.model, PeftModel):
            raise SystemExit("PEFT not active on trainer.model — refuse to train without LoRA")

    t0 = time.monotonic()
    resume = None
    if getattr(args, "resume", False):
        ckpts = sorted(
            (run_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if ckpts:
            resume = str(ckpts[-1])
            print(f"[grpo] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    wall_h = (time.monotonic() - t0) / 3600.0
    print_cost("grpo train actual", wall_h)

    # Save final adapter + copy step checkpoints we care about.
    final_adapter = run_dir / "adapter" / "final"
    final_adapter.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_adapter))

    step_adapters: dict[int, str] = {}
    ckpt_root = run_dir / "checkpoints"
    for step in VAL_CHECKPOINTS:
        src = ckpt_root / f"checkpoint-{step}"
        if src.exists():
            dst = run_dir / "adapter" / f"step_{step}"
            if not dst.exists():
                import shutil

                shutil.copytree(src, dst)
            step_adapters[step] = str(dst)
    if 300 not in step_adapters:
        step_adapters[300] = str(final_adapter)

    n_rewards = recorder.n_records
    if n_rewards == 0:
        n_rewards = len(load_reward_records(recorder.path))
    budgets = {
        "prompts": len(problems),
        "completions_available": int(g["total_sampled_completions"]),
        "completions_consumed": n_rewards,
        "training_tokens": callback.cumulative_tokens,
        "optimizer_steps": int(g["steps"]),
        "final_adapter": str(final_adapter),
        "step_adapters": step_adapters,
        "checkpoint_every_steps": int(cfg["training"].get("checkpoint_every_steps", 25)),
        "reward": reward_name,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "max_steps": int(g["steps"]),
        "tokenizer_sha256": train_tok_hash,
        "gpu_hours_train": round(wall_h, 6),
    }
    write_json(run_dir / "budgets.json", budgets)

    # Training-reward curve from train_log.jsonl (val curve filled by --stage eval).
    curves = {
        "train_reward": _curve_from_train_log(run_dir / "train_log.jsonl"),
        "val_greedy": {},
    }
    write_json(run_dir / "curves.json", curves)

    finish_run(handle, status="finished", gpu_hours_train=wall_h)
    sync_run(run_dir)
    print(f"[grpo] done {run_name}: {n_rewards} reward records, {wall_h:.3f} GPU-h")
    return 0


def _curve_from_train_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.append(
                {
                    "step": row.get("step"),
                    "reward_mean": row.get("reward_mean"),
                    "frac_reward_zero_std": row.get("frac_reward_zero_std"),
                    "completions_mean_length": row.get("completions_mean_length"),
                    "completions_clipped_ratio": row.get("completions_clipped_ratio"),
                    "step_time_s": row.get("step_time_s"),
                }
            )
    return out


def evaluate_grpo_checkpoints(cfg: dict[str, Any], args: Any) -> int:
    """Offline val at {100,200,300}; final (300) gets test/ood/transfer like RFT."""
    from rlordata.train.rft_eval import evaluate_run

    run_dir = Path(getattr(args, "run_dir", None) or "")
    if not run_dir:
        raise SystemExit("--run-dir is required for grpo eval")
    budgets = json.loads((run_dir / "budgets.json").read_text(encoding="utf-8"))
    step_adapters: dict[str, str] = {
        str(k): v for k, v in (budgets.get("step_adapters") or {}).items()
    }
    curves_path = run_dir / "curves.json"
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else {}
    val_curve = dict(curves.get("val_greedy") or {})

    for step in VAL_CHECKPOINTS:
        adapter = step_adapters.get(str(step))
        if not adapter:
            print(f"[grpo-eval] no adapter for step {step}; skip")
            continue
        # Temporarily point budgets.final_adapter at this checkpoint for evaluate_run.
        budgets["final_adapter"] = adapter
        write_json(run_dir / "budgets.json", budgets)
        eval_set = "final" if int(step) == 300 else "val"
        # Stash metrics under eval/step_{n}/ by overriding output via a symlink layout:
        # evaluate_run writes under run_dir/eval/... — move after.
        args.eval_set = eval_set
        rc = evaluate_run(cfg, run_dir, args)
        if rc != 0:
            return rc
        val_metrics = run_dir / "eval" / "val_mixed_100" / "greedy" / "metrics.json"
        if val_metrics.exists():
            m = json.loads(val_metrics.read_text(encoding="utf-8"))
            val_curve[str(step)] = {
                "accuracy": m.get("accuracy"),
                "ci": m.get("ci") or m.get("bootstrap_ci"),
                "truncation_rate": m.get("truncation_rate"),
                "extraction_failure_rate": m.get("extraction_failure_rate"),
            }
            # Keep step-specific copy.
            dest = run_dir / "eval_checkpoints" / f"step_{step}" / "val_mixed_100" / "greedy"
            dest.mkdir(parents=True, exist_ok=True)
            import shutil

            for name in ("metrics.json", "samples.jsonl", "config.yaml", "meta.json"):
                src = val_metrics.parent / name
                if src.exists():
                    shutil.copy2(src, dest / name)
        if int(step) != 300:
            # Remove non-final test artifacts if any slipped through — val-only for mid checkpoints.
            pass

    # Restore final adapter pointer to step 300 / final.
    budgets["final_adapter"] = step_adapters.get("300") or budgets.get("final_adapter")
    write_json(run_dir / "budgets.json", budgets)
    curves["val_greedy"] = val_curve
    write_json(curves_path, curves)
    sync_run(run_dir)
    return 0


def cli_main(args: Any) -> int:
    cfg = load_arm_config(args.config)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = int(args.seed)
    stage = getattr(args, "stage", None) or "train"
    if stage == "train":
        return train_grpo(cfg, args)
    if stage == "eval":
        return evaluate_grpo_checkpoints(cfg, args)
    raise SystemExit(f"unknown grpo stage {stage!r} (train|eval)")
