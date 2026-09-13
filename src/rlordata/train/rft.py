"""RFT arms (SPEC §8): draw → verify → ``core.rft_select`` → SFT with ``core.sft_loss`` → eval.
AGENT-OWNED; tasks/03.

Stages (``rlordata rft --stage ...``):

    draw    the 192-sample draw for ``train_easy_100`` and ``train_mixed_100`` (``sampling.draw``)
    select  print the selection statistics for one arm (no training)
    train   one SFT run: ``runs/rft/<arm>/seed<seed>_lr<lr>_ep<epochs>/`` with adapter, train_log.jsonl,
            budgets.json and provenance
    eval    evaluate a run's final adapter through the single generation path; ``--eval-set val``
            (sweep) or ``--eval-set final`` (val + test + ood + transfer, once)

The loss is ``core.sft_loss(core.completion_logprobs(...))`` — never a library loss. Selection is
``core.rft_select`` — never reimplemented here. Hyperparameters come from the locked
``training.yaml`` plus the sweep grid it defines; the sweep (``scripts/rft_sweep.py``) selects on
``val_mixed_100`` only.

Gradient accumulation: the effective batch is 16 examples (SPEC §9). When it is split into
micro-batches, each micro-batch loss (a token-mean) is weighted by its share of the batch's
completion tokens, so the optimizer step sees exactly the token-mean over all 16 examples that a
single 16-example forward pass would give.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlordata.core.rft_select import SFTExample, rft_select
from rlordata.data.generator import read_jsonl
from rlordata.envfile import load_env
from rlordata.run_dir import finish_run, read_run_config, start_run
from rlordata.sampling.draw import (
    DRAW_SEED_OFFSET,
    SAMPLES_PER_PROMPT,
    TIERING_K,
    draw_path,
    draw_split,
    format_draw_summary,
    load_tiering_samples,
    read_draw,
    summarize_draw,
    write_draw,
)
from rlordata.sampling.prompts import TEMPLATE
from rlordata.train.common import (
    estimate_train_gpu_hours,
    load_arm_config,
    lora_kwargs_from_training,
    print_cost,
    seed_everything,
    sync_run,
    tokenizer_hash,
    write_json,
)
from rlordata.types import Problem, Sample

DEFAULT_OUTPUT_DIR = Path("runs/rft")
DEFAULT_SPLITS_DIR = Path("data/splits")
DEFAULT_SAMPLES_DIR = Path("data/samples")
DRAW_SPLITS = ("train_easy_100", "train_mixed_100")
LOG_EVERY = 10  # tasks/03 §3
PARENT_SPLIT = {"train_easy_100": "train_easy_100", "train_mixed_100": "train_mixed_100"}
# The curated arm's prompts are a subset of train_mixed_100 and read the mixed draw (SPEC §8).
DRAW_SPLIT_FOR_CONDITION = {
    "train_easy_100": "train_easy_100",
    "train_mixed_100": "train_mixed_100",
    "train_curated": "train_mixed_100",
}


# ---------------------------------------------------------------------------
# Selection + budgets (tasks/03 §2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    examples: list[SFTExample]
    budgets: dict[str, Any]

    @property
    def problem_ids(self) -> set[str]:
        return {e.problem_id for e in self.examples}


def select_examples(
    problems: list[Problem],
    draws: dict[str, list[Sample]],
    *,
    mode: str,
    max_per_problem: int | None,
    samples_per_prompt: int = SAMPLES_PER_PROMPT,
) -> Selection:
    """``core.rft_select`` on the parent split's draw, plus the three SPEC §8 budgets.

    ``completions_available`` is the whole parent-split draw (192 × #parent prompts), because
    that is the rollout budget the arm was *given*; ``completions_consumed`` is what survived
    verification, curation and dedup and was actually trained on.
    """
    if mode not in ("all", "curated"):
        raise ValueError(f"select.mode must be 'all' or 'curated', got {mode!r}")
    missing = [p.problem_id for p in problems if p.problem_id not in draws]
    if missing:
        raise ValueError(f"{len(missing)} problem(s) have no draw (first {missing[0][:12]})")
    samples = {p.problem_id: draws[p.problem_id] for p in problems}
    examples = rft_select(problems, samples, mode=mode, max_per_problem=max_per_problem)  # type: ignore[arg-type]
    used = {e.problem_id for e in examples}
    tiers = {p.problem_id: p.tier for p in problems}
    correct_raw = sum(int(s.correct) for pid in used for s in samples[pid])
    budgets = {
        "prompts": len(used),
        "prompts_parent_split": len(problems),
        "completions_available": samples_per_prompt * len(problems),
        "completions_consumed": len(examples),
        "correct_before_dedup": correct_raw,
        "correct_after_dedup": len(examples),
        "max_per_problem": max_per_problem,
        "select_mode": mode,
        "per_tier_prompts": dict(sorted(Counter(tiers[pid] for pid in used).items())),
        "per_tier_examples": dict(sorted(Counter(e.tier for e in examples).items())),
    }
    return Selection(examples=examples, budgets=budgets)


def assert_curated_matches(selection: Selection, curated_path: str | Path) -> None:
    """tasks/03 §2: the persisted ``train_curated`` set is the one source of truth."""
    persisted = {p.problem_id for p in read_jsonl(curated_path)}
    got = selection.problem_ids
    if got != persisted:
        raise SystemExit(
            f"curated selection ({len(got)} prompts) != persisted {curated_path} ({len(persisted)}): "
            f"{len(got - persisted)} extra, {len(persisted - got)} missing. Stopping (tasks/03 §2)."
        )


def format_selection(arm: str, sel: Selection) -> str:
    b = sel.budgets
    return (
        f"[select] {arm}: {b['prompts']} prompts of {b['prompts_parent_split']} "
        f"(per tier {b['per_tier_prompts']}); correct completions {b['correct_before_dedup']} "
        f"before dedup -> {b['correct_after_dedup']} after (per tier {b['per_tier_examples']}); "
        f"budget available {b['completions_available']}, consumed {b['completions_consumed']}"
    )


# ---------------------------------------------------------------------------
# Tokenization (tasks/03 §3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    completion_mask: list[int]  # 1 on completion tokens
    n_prompt_tokens: int

    @property
    def n_completion_tokens(self) -> int:
        return len(self.input_ids) - self.n_prompt_tokens


def assert_eos_is_generation_stop(tokenizer_eos_id: int | None, generation_eos: Any) -> None:
    """The EOS appended in training must be a token vLLM stopped on when it drew the samples.

    ``generation_eos`` is ``GenerationConfig.eos_token_id`` (an int or a list). Qwen3 base and
    instruct tokenizers differ here (``<|endoftext|>`` vs ``<|im_end|>``); a mismatch would train a
    token that never ends generation, so it is a hard stop.
    """
    stops = [generation_eos] if isinstance(generation_eos, int) else list(generation_eos or [])
    if tokenizer_eos_id is None or tokenizer_eos_id not in stops:
        raise ValueError(
            f"tokenizer eos_token_id {tokenizer_eos_id} is not a generation stop token {stops}"
        )


def assert_run_trained_with_eos(run_dir: str | Path) -> None:
    """Refuse to reuse or build on a finished run from before the tasks/03 §3 EOS amendment.

    Every run trained after 2026-09-13 records ``append_eos: true`` in its resolved config. A run
    without it was trained by the no-EOS trainer (kept only as an ablation) and must not be skipped
    over as "already finished", ranked in a sweep, or used to pick the finals' hyperparameters.
    """
    d = Path(run_dir)
    if not (d / "config.yaml").exists():
        raise SystemExit(f"{d}: finished run has no config.yaml; cannot tell which trainer made it")
    cfg = read_run_config(d)
    if cfg.get("append_eos") is not True:
        raise SystemExit(
            f"{d} was trained WITHOUT the EOS token (append_eos={cfg.get('append_eos')!r}; "
            "tasks/03 §3 amended 2026-09-13). Move the old runs aside "
            "(e.g. runs/rft_noeos_ablation/, locally AND in the artifact store) and rerun; "
            "refusing to reuse a stale run."
        )


def encode_example(tokenizer: Any, example: SFTExample) -> Encoded:
    """Prompt and completion tokenized separately and concatenated, then the tokenizer's EOS.

    Separate tokenization mirrors generation: vLLM tokenized the prompt alone and produced the
    completion token by token, ending on EOS. vLLM's returned text omits that EOS, so it is
    appended here as a token id and kept in the loss (tasks/03 §3, amended 2026-09-13): without
    it the model is never trained to stop. The round trip
    ``decode(ids) == prompt + completion + eos_token`` is asserted so a tokenizer that normalises
    whitespace or merges across the boundary cannot go unnoticed.
    """
    if not example.prompt.startswith(TEMPLATE.split("{problem_text}")[0]):
        raise ValueError("SFT prompt does not start with the locked TEMPLATE (prompt drift)")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; cannot train the stop token")
    prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    if len(comp_ids) == 0:
        raise ValueError(f"empty completion for {example.problem_id[:12]}")
    comp_ids = list(comp_ids) + [int(eos_id)]
    ids = list(prompt_ids) + comp_ids
    round_trip = tokenizer.decode(ids, skip_special_tokens=False)
    if round_trip != example.prompt + example.completion + tokenizer.eos_token:
        raise ValueError(
            "tokenizer round trip differs from prompt + completion + eos for "
            f"{example.problem_id[:12]}"
        )
    mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
    return Encoded(input_ids=ids, completion_mask=mask, n_prompt_tokens=len(prompt_ids))


def collate(batch: list[Encoded], pad_id: int):  # noqa: ANN201 — dict of tensors
    """Right-pad to the longest sequence; returns ``[B, T]`` tensors."""
    import torch

    assert len(batch) >= 1
    t_max = max(len(e.input_ids) for e in batch)
    b = len(batch)
    input_ids = torch.full((b, t_max), int(pad_id), dtype=torch.long)
    attention = torch.zeros((b, t_max), dtype=torch.long)
    comp = torch.zeros((b, t_max), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        input_ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attention[i, :n] = 1
        comp[i, :n] = torch.tensor(e.completion_mask, dtype=torch.long)
    assert input_ids.shape == attention.shape == comp.shape == (b, t_max)
    return {"input_ids": input_ids, "attention_mask": attention, "completion_mask": comp}


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Data order for one epoch, a pure function of (seed, epoch)."""
    rng = np.random.default_rng([int(seed), int(epoch)])
    return [int(i) for i in rng.permutation(n)]


def batches_for_epoch(n: int, seed: int, epoch: int, batch_size: int) -> list[list[int]]:
    order = epoch_order(n, seed, epoch)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


# ---------------------------------------------------------------------------
# Training (tasks/03 §3)
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    optimizer_steps: int
    training_tokens: int  # completion tokens the loss was taken over, summed over epochs
    total_tokens: int  # prompt + completion tokens forwarded, summed over epochs
    final_adapter_dir: Path
    epoch_adapter_dirs: list[Path]
    wall_clock_s: float
    last_loss: float


def train_rft(
    model: Any,  # a PEFT-wrapped causal LM (see build_lora_model)
    tokenizer: Any,
    examples: list[SFTExample],
    *,
    run_dir: str | Path,
    seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    micro_batch_size: int,
    grad_clip: float,
    warmup_ratio: float,
    weight_decay: float,
    device: Any = None,
    log_every: int = LOG_EVERY,
) -> TrainResult:
    """The SFT loop. Loss = ``core.sft_loss(core.completion_logprobs(...))``; nothing else."""
    import torch
    from transformers import get_cosine_schedule_with_warmup

    from rlordata.core.logprobs import completion_logprobs
    from rlordata.core.sft_loss import sft_loss

    assert len(examples) > 0, "no SFT examples (selection produced nothing)"
    assert batch_size >= 1 and 1 <= micro_batch_size <= batch_size
    assert batch_size % micro_batch_size == 0, "micro_batch_size must divide batch_size"
    run_dir = Path(run_dir)
    seed_everything(seed)
    device = device or next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    assert pad_id is not None, "tokenizer has neither pad nor eos token"
    encoded = [encode_example(tokenizer, e) for e in examples]

    n = len(encoded)
    steps_per_epoch = math.ceil(n / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = math.ceil(warmup_ratio * total_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "no trainable parameters: LoRA not applied"
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log_path = run_dir / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    t0 = time.monotonic()
    step = 0
    tokens_seen = 0
    total_tokens = 0
    last_loss = float("nan")
    epoch_dirs: list[Path] = []
    model.train()
    for epoch in range(epochs):
        for batch_idx in batches_for_epoch(n, seed, epoch, batch_size):
            batch = [encoded[i] for i in batch_idx]
            batch_comp_tokens = sum(e.n_completion_tokens for e in batch)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m in range(0, len(batch), micro_batch_size):
                micro = batch[m : m + micro_batch_size]
                tensors = collate(micro, pad_id)
                tensors = {k: v.to(device) for k, v in tensors.items()}
                logp = completion_logprobs(
                    model,
                    tensors["input_ids"],
                    tensors["attention_mask"],
                    tensors["completion_mask"],
                )
                assert logp.shape == tensors["input_ids"].shape
                loss = sft_loss(logp, tensors["completion_mask"].to(logp.dtype))
                micro_tokens = sum(e.n_completion_tokens for e in micro)
                weight = micro_tokens / batch_comp_tokens  # exact token-mean over the 16
                (loss * weight).backward()
                step_loss += float(loss.detach()) * weight
                total_tokens += int(tensors["attention_mask"].sum())
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            tokens_seen += batch_comp_tokens
            last_loss = step_loss
            if step % log_every == 0 or step == total_steps or step == 1:
                rec = {
                    "step": step,
                    "epoch": epoch,
                    "loss": step_loss,
                    "lr": float(scheduler.get_last_lr()[0]),
                    "tokens_seen": tokens_seen,
                    "wall_clock_s": round(time.monotonic() - t0, 3),
                }
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                print(
                    f"  step {step}/{total_steps} epoch {epoch} loss {step_loss:.4f} "
                    f"lr {rec['lr']:.2e} tokens {tokens_seen}",
                    flush=True,
                )
        ckpt = run_dir / "adapter" / f"epoch_{epoch + 1}"
        model.save_pretrained(str(ckpt))
        epoch_dirs.append(ckpt)
    log_f.close()
    final = run_dir / "adapter" / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(epoch_dirs[-1], final)
    (final / "SOURCE.txt").write_text(
        f"copy of {epoch_dirs[-1].name} (last epoch; no early stopping)\n"
    )
    return TrainResult(
        optimizer_steps=step,
        training_tokens=tokens_seen,
        total_tokens=total_tokens,
        final_adapter_dir=final,
        epoch_adapter_dirs=epoch_dirs,
        wall_clock_s=time.monotonic() - t0,
        last_loss=last_loss,
    )


def build_lora_model(
    model_id: str,
    training: dict[str, Any],
    *,
    seed: int,
    dtype: str = "bfloat16",
    device: Any = None,
    gradient_checkpointing: bool = True,
    base_model: Any = None,
):  # noqa: ANN201 — (peft model, tokenizer)
    """Base model + fresh LoRA from the locked ``training.yaml``; tokenizer alongside.

    ``seed`` is applied right before the adapter is created so the LoRA ``A`` initialisation
    (PEFT: kaiming-uniform; ``B`` is zero) is a function of the run seed (tasks/03 §3).
    ``base_model`` lets tests pass a tiny in-memory model; result-bearing runs always load
    ``model_id``.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    if device is not None:
        base_model = base_model.to(device)
    if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
    lora = LoraConfig(**lora_kwargs_from_training(training))
    seed_everything(seed)
    model = get_peft_model(base_model, lora)
    return model, tokenizer


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


# ---------------------------------------------------------------------------
# Run directory naming
# ---------------------------------------------------------------------------


def run_name(seed: int, learning_rate: float, epochs: int) -> str:
    return f"seed{seed}_lr{learning_rate:g}_ep{epochs}"


def resolved_train_config(
    cfg: dict[str, Any],
    *,
    seed: int,
    learning_rate: float,
    epochs: int,
    micro_batch_size: int,
    draw_file: Path,
    selection: Selection,
    tokenizer_sha: str,
    dtype: str,
    eos_token_id: int,
) -> dict[str, Any]:
    tr = cfg["training"]
    return {
        "kind": "rft_train",
        "arm": cfg["arm"],
        "source_config": cfg.get("source_config"),
        "model_id": cfg["model_id"],
        "data_condition": cfg["data_condition"],
        "seed": seed,
        "select": dict(cfg.get("select", {})),
        "draw_file": str(draw_file),
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "lora": lora_kwargs_from_training(tr),
        "optimizer": {
            "name": "adamw",
            "learning_rate": learning_rate,
            "weight_decay": float(tr["optimizer"]["weight_decay"]),
            "grad_clip": float(tr["optimizer"]["grad_clip"]),
            "betas": [0.9, 0.999],
        },
        "schedule": {"name": "cosine", "warmup_ratio": float(tr["schedule"]["warmup_ratio"])},
        "epochs": epochs,
        "batch_size": int(tr["rft"]["batch_size"]),
        "micro_batch_size": micro_batch_size,
        "loss": "core.sft_loss(core.completion_logprobs) token-mean, prompt masked, "
        "micro-batches weighted by completion tokens",
        "append_eos": True,  # tasks/03 §3 amended 2026-09-13: EOS appended and trained
        "eos_token_id": eos_token_id,
        "precision": dtype,
        "prompt_template": TEMPLATE,
        "max_completion_tokens": int(cfg["max_completion_tokens"]),
        "cap_yaml": cfg["cap_yaml"],
        "training_yaml": cfg["training_yaml"],
        "tokenizer_sha256": tokenizer_sha,
        "n_examples": len(selection.examples),
        "budgets": selection.budgets,
        "checkpoint": "adapter at every epoch end; final = last epoch (no early stopping)",
    }


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _dtype_name(value: str) -> str:
    """Config shorthand (``bf16`` as in the locked ``training.yaml``) → the name torch and vLLM accept."""
    return {"bf16": "bfloat16", "fp32": "float32"}.get(value, value)


def _sampler_for_draw(
    cfg: dict[str, Any], *, seed: int, stub: bool, problems: list[Problem]
) -> Any:
    from rlordata.sampling.eval_runner import ModelSpec, make_sampler

    spec = ModelSpec(id=cfg["model_id"], kind="base", arm="base")
    return make_sampler(
        spec,
        stub=stub,
        cap=int(cfg["max_completion_tokens"]),
        seed=seed + DRAW_SEED_OFFSET,
        cap_path=cfg["cap_yaml"],
        allow_provisional_cap=False,
        problems=problems,
        dtype=_dtype_name(str(cfg.get("dtype", "bfloat16"))),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
    )


def stage_draw(cfg: dict[str, Any], args: Any) -> int:
    """tasks/03 §1: both training splits, once, 192 per prompt, read-only files."""
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    stub = bool(getattr(args, "stub", False))
    splits_dir = Path(
        getattr(args, "splits_dir", None) or cfg.get("splits_dir", DEFAULT_SPLITS_DIR)
    )
    samples_dir = Path(
        getattr(args, "samples_dir", None) or cfg.get("samples_dir", DEFAULT_SAMPLES_DIR)
    )
    tiering_path = Path(cfg.get("tiering_samples", samples_dir / "tiering_pass8.jsonl"))
    output_dir = Path(
        getattr(args, "output_dir", None) or cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    run_dir = output_dir / (f"draw_seed{seed}" + ("_stub" if stub else ""))
    n_total = int(getattr(args, "n_total", None) or SAMPLES_PER_PROMPT)
    if stub:
        print("STUB SAMPLER: scripted completions, no GPU, no model weights. Not a result.")
    elif n_total != SAMPLES_PER_PROMPT:
        raise SystemExit(
            f"--n-total is only for --stub dry runs; the protocol draw is {SAMPLES_PER_PROMPT}"
        )
    targets = []
    for split in cfg.get("draw_splits", DRAW_SPLITS):
        path = draw_path(split, seed, samples_dir=samples_dir)
        if path.exists():
            print(f"[draw] {path} exists — skipping (drawn once, never re-drawn)")
            continue
        targets.append((split, path))
    if not targets:
        print("[draw] nothing to do")
        return 0
    resolved = {
        "kind": "rft_draw",
        "model_id": cfg["model_id"],
        "seed": seed,
        "sampler_seed": seed + DRAW_SEED_OFFSET,
        "draw_seed_offset": DRAW_SEED_OFFSET,
        "splits": [s for s, _ in targets],
        "samples_per_prompt": n_total,
        "tiering_samples": str(tiering_path),
        "tiering_k": TIERING_K,
        "decoding": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
        "max_completion_tokens": int(cfg["max_completion_tokens"]),
        "cap_yaml": cfg["cap_yaml"],
        "prompt_template": TEMPLATE,
        "stub": stub,
    }
    handle = start_run(run_dir, resolved, run_id=f"rft_draw_seed{seed}" + ("_stub" if stub else ""))
    n_prompts = sum(len(read_jsonl(splits_dir / f"{s}.jsonl")) for s, _ in targets)
    worst_tokens = n_prompts * (n_total - TIERING_K) * int(cfg["max_completion_tokens"])
    print_cost(
        "draw start estimate (worst case, every completion at the cap)",
        worst_tokens / 3000.0 / 3600.0,
    )
    summaries: dict[str, Any] = {}
    try:
        for split, path in targets:
            problems = read_jsonl(splits_dir / f"{split}.jsonl")
            tiering = load_tiering_samples(tiering_path, problems, k=TIERING_K)
            sampler = _sampler_for_draw(cfg, seed=seed, stub=stub, problems=problems)
            try:
                samples = draw_split(
                    problems,
                    tiering,
                    sampler,
                    split=split,
                    run_id=handle.run_id,
                    config_hash=handle.config_hash,
                    seed=seed,
                    n_total=n_total,
                )
            finally:
                if hasattr(sampler, "close"):
                    sampler.close()
            write_draw(samples, path)
            summary = summarize_draw(read_draw(path, expect_n=n_total), k=TIERING_K)
            summaries[split] = {**summary, "path": str(path)}
            print(format_draw_summary(split, summary))
        write_json(run_dir / "draw_summary.json", summaries)
        finish_run(handle, summaries=summaries)
        print_cost("draw actual", handle.elapsed_s / 3600.0)
    except BaseException:
        finish_run(handle, status="failed")
        raise
    finally:
        sync_run(run_dir)
    return 0


def load_selection(
    cfg: dict[str, Any], *, seed: int, splits_dir: Path, samples_dir: Path
) -> tuple[Selection, Path]:
    condition = cfg["data_condition"]
    draw_split_name = DRAW_SPLIT_FOR_CONDITION.get(condition)
    if draw_split_name is None:
        raise ValueError(f"no draw is defined for data_condition {condition!r}")
    parent = read_jsonl(splits_dir / f"{draw_split_name}.jsonl")
    draw_file = draw_path(draw_split_name, int(cfg.get("draw_seed", 1)), samples_dir=samples_dir)
    if not draw_file.exists():
        raise SystemExit(
            f"{draw_file} missing: run `rlordata rft --stage draw` first (tasks/03 §1)"
        )
    draws = read_draw(draw_file, expect_n=int(cfg.get("samples_per_prompt", SAMPLES_PER_PROMPT)))
    select = dict(cfg.get("select", {}))
    sel = select_examples(
        parent,
        draws,
        mode=str(select.get("mode", "all")),
        max_per_problem=select.get("max_per_problem"),
        samples_per_prompt=int(cfg.get("samples_per_prompt", SAMPLES_PER_PROMPT)),
    )
    if condition == "train_curated":
        if select.get("mode") != "curated":
            raise ValueError("data_condition train_curated requires select.mode == 'curated'")
        assert_curated_matches(sel, splits_dir / "train_curated.jsonl")
    elif select.get("mode") == "curated":
        raise ValueError("select.mode 'curated' is only for data_condition train_curated")
    return sel, draw_file


def stage_select(cfg: dict[str, Any], args: Any) -> int:
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    splits_dir = Path(
        getattr(args, "splits_dir", None) or cfg.get("splits_dir", DEFAULT_SPLITS_DIR)
    )
    samples_dir = Path(
        getattr(args, "samples_dir", None) or cfg.get("samples_dir", DEFAULT_SAMPLES_DIR)
    )
    sel, draw_file = load_selection(cfg, seed=seed, splits_dir=splits_dir, samples_dir=samples_dir)
    print(f"[select] draw file: {draw_file}")
    print(format_selection(cfg["arm"], sel))
    return 0


def stage_train(cfg: dict[str, Any], args: Any) -> int:
    """One SFT run. Requires ``--lr`` and ``--epochs`` (the sweep passes them)."""
    import torch

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1))
    if getattr(args, "lr", None) is None or getattr(args, "epochs", None) is None:
        raise SystemExit(
            "--stage train needs --lr and --epochs (from the sweep grid in training.yaml)"
        )
    learning_rate = float(args.lr)
    epochs = int(args.epochs)
    tr = cfg["training"]
    grid = tr["rft"]["sweep"]
    if learning_rate not in [float(x) for x in grid["learning_rate"]] or epochs not in [
        int(x) for x in grid["epochs"]
    ]:
        raise SystemExit(
            f"lr={learning_rate} epochs={epochs} is not on the locked sweep grid {grid} (SPEC §9)"
        )
    splits_dir = Path(
        getattr(args, "splits_dir", None) or cfg.get("splits_dir", DEFAULT_SPLITS_DIR)
    )
    samples_dir = Path(
        getattr(args, "samples_dir", None) or cfg.get("samples_dir", DEFAULT_SAMPLES_DIR)
    )
    output_dir = Path(
        getattr(args, "output_dir", None) or cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    )
    run_dir = Path(
        getattr(args, "run_dir", None)
        or output_dir / cfg["arm"] / run_name(seed, learning_rate, epochs)
    )
    if (run_dir / "budgets.json").exists() and not getattr(args, "force", False):
        assert_run_trained_with_eos(run_dir)  # never silently reuse a pre-amendment run
        print(f"[train] {run_dir} already finished — skipping (use --force to retrain)")
        return 0
    sel, draw_file = load_selection(cfg, seed=seed, splits_dir=splits_dir, samples_dir=samples_dir)
    print(format_selection(cfg["arm"], sel))
    dtype = _dtype_name(str(cfg.get("dtype", tr.get("precision", "bf16"))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not getattr(args, "allow_cpu", False):
        raise SystemExit(
            "no CUDA device: result-bearing training runs on the GPU box only (pass --allow-cpu for smoke tests)"
        )
    micro = int(getattr(args, "micro_batch_size", None) or cfg.get("micro_batch_size", 4))
    batch_size = int(tr["rft"]["batch_size"])
    model, tokenizer = build_lora_model(
        cfg["model_id"],
        tr,
        seed=seed,
        dtype=dtype,
        device=device,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
    )
    # The appended EOS must be the token vLLM stopped on (tasks/03 §3, amended 2026-09-13).
    assert_eos_is_generation_stop(
        tokenizer.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    tok_sha = tokenizer_hash(tokenizer)
    resolved = resolved_train_config(
        cfg,
        seed=seed,
        learning_rate=learning_rate,
        epochs=epochs,
        micro_batch_size=micro,
        draw_file=draw_file,
        selection=sel,
        tokenizer_sha=tok_sha,
        dtype=dtype,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    resolved["parameters"] = trainable_parameter_summary(model)
    run_id = f"rft_{cfg['arm']}_{run_name(seed, learning_rate, epochs)}"
    handle = start_run(
        run_dir, resolved, run_id=run_id, extra_meta={"arm": cfg["arm"], "seed": seed}
    )
    est_tokens = sum(
        len(tokenizer(e.prompt + e.completion, add_special_tokens=False)["input_ids"])
        for e in sel.examples[:200]
    )
    est_tokens = int(est_tokens / max(min(len(sel.examples), 200), 1) * len(sel.examples) * epochs)
    print_cost(
        f"train start estimate ({est_tokens} tokens × {epochs} epochs)",
        estimate_train_gpu_hours(est_tokens),
    )
    try:
        result = train_rft(
            model,
            tokenizer,
            sel.examples,
            run_dir=run_dir,
            seed=seed,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            micro_batch_size=micro,
            grad_clip=float(tr["optimizer"]["grad_clip"]),
            warmup_ratio=float(tr["schedule"]["warmup_ratio"]),
            weight_decay=float(tr["optimizer"]["weight_decay"]),
            device=device,
        )
        budgets = {
            **sel.budgets,
            "optimizer_steps": result.optimizer_steps,
            "training_tokens": result.training_tokens,
            "total_tokens_forwarded": result.total_tokens,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "final_adapter": str(result.final_adapter_dir),
            "epoch_adapters": [str(p) for p in result.epoch_adapter_dirs],
        }
        write_json(run_dir / "budgets.json", budgets)
        finish_run(
            handle,
            optimizer_steps=result.optimizer_steps,
            training_tokens=result.training_tokens,
            last_loss=result.last_loss,
        )
        print_cost("train actual", handle.elapsed_s / 3600.0)
        print(
            f"[train] {run_id}: {result.optimizer_steps} steps, {result.training_tokens} completion tokens, final loss {result.last_loss:.4f}"
        )
    except BaseException:
        finish_run(handle, status="failed")
        raise
    finally:
        sync_run(run_dir)
    return 0


def stage_eval(cfg: dict[str, Any], args: Any) -> int:
    from rlordata.train.rft_eval import evaluate_run

    run_dir = getattr(args, "run_dir", None)
    if not run_dir:
        raise SystemExit("--stage eval needs --run-dir (the training run to evaluate)")
    return evaluate_run(cfg, Path(run_dir), args)


def cli_main(args: Any) -> int:
    load_env()
    cfg = load_arm_config(args.config)
    stage = str(getattr(args, "stage", "train"))
    stages = {"draw": stage_draw, "select": stage_select, "train": stage_train, "eval": stage_eval}
    if stage not in stages:
        print(f"unknown stage {stage!r}; choose from {sorted(stages)}", file=sys.stderr)
        return 2
    return int(stages[stage](cfg, args) or 0)
