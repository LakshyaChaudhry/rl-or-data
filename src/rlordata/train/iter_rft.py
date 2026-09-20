"""Iterated RFT on the frozen curated prompts (SPEC §8 Priority 2; tasks/06b). AGENT-OWNED.

Registered after the six primary arms' test results were seen (PREREGISTRATION §6, 2026-09-20): a
secondary arm. Everything that is not the round structure is tasks/03 unchanged — the same
verifier, ``core.rft_select``, ``core.sft_loss``, EOS rule, LoRA, optimizer, schedule and cap.

    round 1   the first ``samples_per_round`` (64) samples per prompt, in generation order, of the
              existing base-model draw (no new sampling; identical for every seed)
    round r   64 samples per prompt from the current policy π_{r−1} through vLLM native LoRA,
              T = 1.0, top_p = 1.0, locked cap and prompt; sampler seed = 1000 × seed + r
    training  round r *continues the previous round's adapter* on that round's kept samples only,
              with a fresh AdamW and a fresh cosine schedule; lr and epochs are RFT-Curated's
              val-chosen config, reused without a sweep (PREREGISTRATION §4)
    final     the adapter after the last round, whatever the val curve says

Layout: ``runs/rft/iter_rft_curated/seed{s}/round_{r}/`` is a complete tasks/03-style run dir
(so ``rlordata rft --stage eval --eval-set val --run-dir …`` evaluates it), and the seed dir
itself becomes one after ``finalize`` (so ``--eval-set final`` evaluates the final adapter once).

Stages (``rlordata iter-rft --stage …``): ``sample`` (rounds ≥ 2), ``train``, ``finalize``.
``scripts/iter_rft.py`` runs them in order. Nothing here reads a held-out split.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rlordata.core.rft_select import SFTExample
from rlordata.data.generator import read_jsonl
from rlordata.envfile import load_env
from rlordata.run_dir import finish_run, start_run
from rlordata.sampling.draw import draw_path, read_draw
from rlordata.sampling.eval_runner import (
    ModelSpec,
    build_samples,
    make_sampler,
    read_samples,
    write_samples,
)
from rlordata.sampling.prompts import format_prompt
from rlordata.train.common import (
    estimate_train_gpu_hours,
    load_arm_config,
    print_cost,
    read_json,
    release_cuda_cache,
    seed_everything,
    sync_run,
    tokenizer_hash,
    write_json,
)
from rlordata.train.rft import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLES_DIR,
    DEFAULT_SPLITS_DIR,
    DRAW_SPLIT_FOR_CONDITION,
    Selection,
    _dtype_name,
    assert_eos_is_generation_stop,
    build_lora_model,
    collate,
    encode_example,
    format_selection,
    resolved_train_config,
    select_examples,
    train_rft,
    trainable_parameter_summary,
)
from rlordata.types import Problem, Sample

SAMPLER_SEED_STRIDE = 1000
LOGPROB_DIAG_MAX_EXAMPLES = 256
CURATED = "train_curated"


# ---------------------------------------------------------------------------
# paths, seeds, guards
# ---------------------------------------------------------------------------


def seed_dir(output_dir: Path, arm: str, seed: int) -> Path:
    return output_dir / arm / f"seed{seed}"


def round_dir(run_dir: Path, r: int) -> Path:
    return run_dir / f"round_{r}"


def round_sampler_seed(seed: int, r: int) -> int:
    """vLLM seed for round ``r`` of training seed ``seed``; never equal to a training seed."""
    assert seed >= 1 and r >= 1
    return SAMPLER_SEED_STRIDE * int(seed) + int(r)


def _paths(cfg: dict[str, Any], args: Any) -> tuple[Path, Path, Path]:
    splits = Path(getattr(args, "splits_dir", None) or cfg.get("splits_dir", DEFAULT_SPLITS_DIR))
    samples = Path(
        getattr(args, "samples_dir", None) or cfg.get("samples_dir", DEFAULT_SAMPLES_DIR)
    )
    output = Path(getattr(args, "output_dir", None) or cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    return splits, samples, output


def validate_config(cfg: dict[str, Any]) -> None:
    """The arm's fixed design (tasks/06b A1–A5); anything else is a different experiment."""
    if cfg.get("data_condition") != CURATED:
        raise SystemExit("iterated RFT is registered on train_curated only (SPEC §8)")
    select = dict(cfg.get("select") or {})
    if select.get("mode") != "all" or select.get("max_per_problem") is not None:
        raise SystemExit(
            "select must be {mode: all, max_per_problem: null}: the curated set is frozen and is "
            "never re-curated from a later policy's samples (SPEC §6, tasks/06b A1)"
        )
    grid = cfg["training"]["rft"]["sweep"]
    lr, ep = float(cfg["learning_rate"]), int(cfg["epochs"])
    if lr not in [float(x) for x in grid["learning_rate"]] or ep not in [
        int(x) for x in grid["epochs"]
    ]:
        raise SystemExit(f"lr={lr} epochs={ep} is not on the locked sweep grid {grid} (SPEC §9)")
    if int(cfg["rounds"]) < 1 or int(cfg["samples_per_round"]) < 1:
        raise SystemExit("rounds and samples_per_round must be positive")


def assert_preregistered(cfg: dict[str, Any]) -> None:
    """tasks/06b prerequisite 1: HEAD must descend from the PREREGISTRATION §6 commit."""
    commit = cfg.get("preregistration_commit")
    if not commit:
        raise SystemExit("config lacks preregistration_commit (tasks/06b prerequisite 1)")
    try:
        rc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(commit), "HEAD"],
            capture_output=True,
            check=False,
        ).returncode
    except FileNotFoundError:
        rc = 127
    if rc != 0:
        raise SystemExit(
            f"HEAD does not descend from the pre-registration commit {commit}: refusing to start a "
            "result-bearing iterated-RFT stage (tasks/06b prerequisite 1)"
        )


def curated_problems(splits_dir: Path) -> list[Problem]:
    problems = read_jsonl(splits_dir / f"{CURATED}.jsonl")
    assert len({p.problem_id for p in problems}) == len(problems)
    return problems


# ---------------------------------------------------------------------------
# round data
# ---------------------------------------------------------------------------


def round1_samples(
    problems: list[Problem], base_draw: dict[str, list[Sample]], n: int
) -> list[Sample]:
    """The first ``n`` samples per curated prompt of the base draw, in generation order."""
    out: list[Sample] = []
    for p in problems:
        draws = base_draw.get(p.problem_id)
        if draws is None or len(draws) < n:
            raise SystemExit(f"base draw has no {n} samples for {p.problem_id[:12]}")
        if draws[0].prompt != format_prompt(p, "base"):
            raise SystemExit(
                f"prompt drift between the base draw and TEMPLATE ({p.problem_id[:12]})"
            )
        for s in draws[:n]:
            assert int(s.extra["sample_idx"]) < n
            out.append(
                Sample(
                    **{
                        **s.to_dict(),
                        "data_condition": CURATED,
                        "tier": p.tier,
                        "extra": {**s.extra, "round": 1, "round_source": "base_draw_first_n"},
                    }
                )
            )
    return out


def samples_path(rdir: Path) -> Path:
    return rdir / "samples.jsonl"


def round_diagnostics(
    problems: list[Problem], samples: list[Sample], sel: Selection
) -> dict[str, Any]:
    """What the round's samples looked like and what survived selection (tasks/06b A6)."""
    tiers = {p.problem_id: p.tier for p in problems}
    by_pid: dict[str, list[Sample]] = {}
    for s in samples:
        by_pid.setdefault(s.problem_id, []).append(s)
    pass_rate = {pid: float(np.mean([s.correct for s in ss])) for pid, ss in by_pid.items()}
    per_tier: dict[str, list[float]] = {}
    for pid, r in pass_rate.items():
        per_tier.setdefault(tiers[pid], []).append(r)
    kept = {(e.problem_id, e.completion) for e in sel.examples}
    kept_samples = [s for s in samples if s.correct and (s.problem_id, s.completion) in kept]
    correct_by_tier = Counter(tiers[s.problem_id] for s in samples if s.correct)

    def lengths(ss: list[Sample]) -> dict[str, float | None]:
        if not ss:
            return {"mean_tokens": None, "p99_tokens": None, "truncation_rate": None}
        toks = np.asarray([s.n_tokens for s in ss], dtype=np.float64)
        return {
            "mean_tokens": float(toks.mean()),
            "p99_tokens": float(np.quantile(toks, 0.99)),
            "truncation_rate": float(np.mean([s.truncated for s in ss])),
        }

    n_correct = sum(int(s.correct) for s in samples)
    return {
        "n_prompts": len(by_pid),
        "n_samples": len(samples),
        "pass_rate_mean": float(np.mean(list(pass_rate.values()))),
        "pass_rate_per_tier": {t: float(np.mean(v)) for t, v in sorted(per_tier.items())},
        "pass_rate_per_prompt": {pid: pass_rate[pid] for pid in sorted(pass_rate)},
        "n_prompts_all_correct": sum(1 for r in pass_rate.values() if r == 1.0),
        "n_prompts_zero_correct": sum(1 for r in pass_rate.values() if r == 0.0),
        "correct_before_dedup": n_correct,
        "correct_before_dedup_per_tier": dict(sorted(correct_by_tier.items())),
        "kept_after_dedup": len(sel.examples),
        "kept_after_dedup_per_tier": sel.budgets["per_tier_examples"],
        "dedup_rate": (1.0 - len(sel.examples) / n_correct) if n_correct else None,
        "unique_completion_fraction": len({(s.problem_id, s.completion) for s in samples})
        / len(samples),
        "extraction_failure_rate": float(np.mean([s.extraction_failed for s in samples])),
        "sampled": lengths(samples),
        "kept": lengths(kept_samples),
    }


def mean_completion_logprob(
    model: Any,
    tokenizer: Any,
    examples: list[SFTExample],
    *,
    device: Any,
    micro_batch_size: int,
    adapter_enabled: bool,
) -> float:
    """Token-mean log-prob of the completions (+EOS) under the model, adapter on or off."""
    import torch

    from rlordata.core.logprobs import completion_logprobs

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    encoded = [encode_example(tokenizer, e) for e in examples]
    total, n_tokens = 0.0, 0
    was_training = model.training
    model.eval()
    ctx = torch.no_grad()
    with ctx:
        for i in range(0, len(encoded), micro_batch_size):
            tensors = collate(encoded[i : i + micro_batch_size], pad_id)
            tensors = {k: v.to(device) for k, v in tensors.items()}  # each [B, T]
            args = (tensors["input_ids"], tensors["attention_mask"], tensors["completion_mask"])
            if adapter_enabled:
                logp = completion_logprobs(model, *args)
            else:
                with model.disable_adapter():
                    logp = completion_logprobs(model, *args)
            assert logp.shape == tensors["input_ids"].shape
            mask = tensors["completion_mask"].to(logp.dtype)
            total += float((logp * mask).sum())
            n_tokens += int(mask.sum())
    if was_training:
        model.train()
    return total / max(n_tokens, 1)


def onpolicy_diagnostic(
    model: Any,
    tokenizer: Any,
    examples: list[SFTExample],
    *,
    seed: int,
    r: int,
    device: Any,
    micro_batch_size: int,
) -> dict[str, Any]:
    """How on-policy the round's kept data is: mean log-prob under π_0 and under π_{r−1}."""
    rng = np.random.default_rng([int(seed), int(r), 7])
    idx = sorted(rng.permutation(len(examples))[:LOGPROB_DIAG_MAX_EXAMPLES].tolist())
    sub = [examples[i] for i in idx]
    kw = {"device": device, "micro_batch_size": micro_batch_size}
    prev = mean_completion_logprob(model, tokenizer, sub, adapter_enabled=True, **kw)
    base = (
        prev
        if r == 1
        else mean_completion_logprob(model, tokenizer, sub, adapter_enabled=False, **kw)
    )
    return {
        "n_examples": len(sub),
        "mean_token_logprob_under_pi0": base,
        "mean_token_logprob_under_prev_policy": prev,
        "note": "round 1: π_{r−1} = π_0 (fresh LoRA, B = 0)",
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def _common(cfg: dict[str, Any], args: Any) -> tuple[int, int, bool, Path, Path, Path]:
    validate_config(cfg)
    seed = int(args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 1))
    r = int(getattr(args, "round", None) or 0)
    stub = bool(getattr(args, "stub", False))
    if not stub and not getattr(args, "allow_cpu", False):
        assert_preregistered(cfg)
    splits_dir, samples_dir, output_dir = _paths(cfg, args)
    return seed, r, stub, splits_dir, samples_dir, seed_dir(output_dir, cfg["arm"], seed)


def stage_sample(cfg: dict[str, Any], args: Any) -> int:
    """Rounds ≥ 2: ``samples_per_round`` per curated prompt from the previous round's adapter."""
    seed, r, stub, splits_dir, _, run_dir = _common(cfg, args)
    if not 2 <= r <= int(cfg["rounds"]):
        raise SystemExit(
            f"--stage sample needs --round in 2..{cfg['rounds']} (round 1 reuses the base draw)"
        )
    rdir = round_dir(run_dir, r)
    if samples_path(rdir).exists() and not getattr(args, "force", False):
        print(f"[iter-rft] {samples_path(rdir)} exists — skipping")
        return 0
    prev = round_dir(run_dir, r - 1)
    adapter = Path(read_json(prev / "budgets.json")["final_adapter"])
    if not adapter.exists():
        raise SystemExit(f"{adapter} missing: train round {r - 1} first")
    problems = curated_problems(splits_dir)
    n = int(cfg["samples_per_round"])
    sampler_seed = round_sampler_seed(seed, r)
    resolved = {
        "kind": "iter_rft_sample",
        "arm": cfg["arm"],
        "model_id": cfg["model_id"],
        "data_condition": CURATED,
        "seed": seed,
        "round": r,
        "sampler_seed": sampler_seed,
        "policy_adapter": str(adapter),
        "samples_per_prompt": n,
        "n_prompts": len(problems),
        "decoding": {"temperature": 1.0, "top_p": 1.0, "repetition_penalty": 1.0},
        "max_completion_tokens": int(cfg["max_completion_tokens"]),
        "cap_yaml": cfg["cap_yaml"],
        "stub": stub,
    }
    handle = start_run(
        rdir / "sampling",
        resolved,
        run_id=f"iter_rft_sample_{cfg['arm']}_seed{seed}_round{r}",
        extra_meta={"arm": cfg["arm"], "seed": seed, "round": r},
    )
    print_cost(
        "sample start estimate (worst case, every completion at the cap)",
        len(problems) * n * int(cfg["max_completion_tokens"]) / 3000.0 / 3600.0,
    )
    if stub:
        print("STUB SAMPLER: scripted completions, no GPU, no model weights. Not a result.")
    sampler = None
    try:
        release_cuda_cache()
        sampler = make_sampler(
            ModelSpec(id=cfg["model_id"], kind="base", arm=cfg["arm"], lora_path=str(adapter)),
            stub=stub,
            cap=int(cfg["max_completion_tokens"]),
            seed=sampler_seed,
            cap_path=cfg["cap_yaml"],
            allow_provisional_cap=False,
            problems=problems,
            dtype=_dtype_name(str(cfg.get("dtype", "bfloat16"))),
            gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.9)),
        )
        prompts = [format_prompt(p, "base") for p in problems]
        completions = sampler.sample(prompts, n=n, temperature=1.0, top_p=1.0)
        samples = build_samples(
            problems,
            prompts,
            completions,
            run_id=handle.run_id,
            config_hash=handle.config_hash,
            seed=seed,
            arm=cfg["arm"],
            data_condition=CURATED,
            extra={
                "model_id": cfg["model_id"],
                "decoding": "iter_rft_round",
                "temperature": 1.0,
                "top_p": 1.0,
                "sampler_seed": sampler_seed,
                "round": r,
                "round_source": "policy",
                "policy_adapter": str(adapter),
            },
        )
        assert len(samples) == len(problems) * n
        write_samples(samples, samples_path(rdir))
        finish_run(handle, n_samples=len(samples))
        print_cost("sample actual", handle.elapsed_s / 3600.0)
    except BaseException:
        finish_run(handle, status="failed")
        raise
    finally:
        if sampler is not None and hasattr(sampler, "close"):
            sampler.close()
        sync_run(run_dir)
    return 0


def stage_train(cfg: dict[str, Any], args: Any) -> int:
    """One round of SFT on that round's kept samples, continuing the previous round's adapter."""
    import torch

    seed, r, stub, splits_dir, samples_dir, run_dir = _common(cfg, args)
    rounds, n = int(cfg["rounds"]), int(cfg["samples_per_round"])
    if not 1 <= r <= rounds:
        raise SystemExit(f"--stage train needs --round in 1..{rounds}")
    rdir = round_dir(run_dir, r)
    if (rdir / "budgets.json").exists() and not getattr(args, "force", False):
        print(f"[iter-rft] {rdir} already trained — skipping (use --force)")
        return 0
    problems = curated_problems(splits_dir)
    if r == 1:
        draw_file = draw_path(
            DRAW_SPLIT_FOR_CONDITION[CURATED], int(cfg.get("draw_seed", 1)), samples_dir=samples_dir
        )
        if not draw_file.exists():
            raise SystemExit(f"{draw_file} missing: restore the tasks/03 draw first")
        base_draw = read_draw(draw_file, expect_n=int(cfg.get("samples_per_prompt_base_draw", 192)))
        samples = round1_samples(problems, base_draw, n)
        write_samples(samples, samples_path(rdir))
        prev_adapter = None
    else:
        if not samples_path(rdir).exists():
            raise SystemExit(f"{samples_path(rdir)} missing: run --stage sample --round {r} first")
        samples = read_samples(samples_path(rdir))
        prev_adapter = Path(read_json(round_dir(run_dir, r - 1) / "budgets.json")["final_adapter"])
    by_pid: dict[str, list[Sample]] = {}
    for s in samples:
        by_pid.setdefault(s.problem_id, []).append(s)
    if set(by_pid) != {p.problem_id for p in problems} or any(len(v) != n for v in by_pid.values()):
        raise SystemExit(
            f"{samples_path(rdir)}: expected exactly {n} samples for each curated prompt"
        )
    sel = select_examples(problems, by_pid, mode="all", max_per_problem=None, samples_per_prompt=n)
    print(format_selection(f"{cfg['arm']} round {r}", sel))
    diagnostics = round_diagnostics(problems, samples, sel)

    tr = cfg["training"]
    lr, epochs = float(cfg["learning_rate"]), int(cfg["epochs"])
    dtype = _dtype_name(str(cfg.get("dtype", tr.get("precision", "bf16"))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not getattr(args, "allow_cpu", False):
        raise SystemExit("no CUDA device: result-bearing training runs on the GPU box only")
    micro = int(getattr(args, "micro_batch_size", None) or cfg.get("micro_batch_size", 4))
    release_cuda_cache()
    model, tokenizer = build_lora_model(
        cfg["model_id"],
        tr,
        seed=seed,
        dtype=dtype,
        device=device,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        adapter_path=prev_adapter,
    )
    assert_eos_is_generation_stop(
        tokenizer.eos_token_id,
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    resolved = resolved_train_config(
        cfg,
        seed=seed,
        learning_rate=lr,
        epochs=epochs,
        micro_batch_size=micro,
        draw_file=samples_path(rdir),
        selection=sel,
        tokenizer_sha=tokenizer_hash(tokenizer),
        dtype=dtype,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    resolved.update(
        {
            "kind": "iter_rft_round",
            "round": r,
            "rounds": rounds,
            "samples_per_prompt": n,
            "round_source": "base_draw_first_n" if r == 1 else "policy",
            "init_adapter": None if prev_adapter is None else str(prev_adapter),
            "round_training_rule": "continue previous adapter on this round's kept samples only; "
            "fresh AdamW and cosine schedule (PREREGISTRATION §6)",
            "hyperparameters_from": cfg.get("hyperparameters_from"),
            "preregistration_commit": cfg.get("preregistration_commit"),
            "parameters": trainable_parameter_summary(model),
        }
    )
    handle = start_run(
        rdir,
        resolved,
        run_id=f"iter_rft_{cfg['arm']}_seed{seed}_round{r}",
        extra_meta={"arm": cfg["arm"], "seed": seed, "round": r},
    )
    est = int(
        np.mean([s.n_tokens for s in samples if s.correct] or [0]) * len(sel.examples) * epochs
    )
    print_cost(
        f"round {r} train start estimate ({est} completion tokens)", estimate_train_gpu_hours(est)
    )
    try:
        diagnostics["onpolicy"] = onpolicy_diagnostic(
            model, tokenizer, sel.examples, seed=seed, r=r, device=device, micro_batch_size=micro
        )
        result = train_rft(
            model,
            tokenizer,
            sel.examples,
            run_dir=rdir,
            seed=seed,
            learning_rate=lr,
            epochs=epochs,
            batch_size=int(tr["rft"]["batch_size"]),
            micro_batch_size=micro,
            grad_clip=float(tr["optimizer"]["grad_clip"]),
            warmup_ratio=float(tr["schedule"]["warmup_ratio"]),
            weight_decay=float(tr["optimizer"]["weight_decay"]),
            device=device,
        )
        write_json(rdir / "diagnostics.json", diagnostics)
        write_json(
            rdir / "budgets.json",
            {
                **sel.budgets,
                "round": r,
                "optimizer_steps": result.optimizer_steps,
                "training_tokens": result.training_tokens,
                "total_tokens_forwarded": result.total_tokens,
                "epochs": epochs,
                "batch_size": int(tr["rft"]["batch_size"]),
                "learning_rate": lr,
                "final_adapter": str(result.final_adapter_dir),
                "epoch_adapters": [str(p) for p in result.epoch_adapter_dirs],
                "init_adapter": None if prev_adapter is None else str(prev_adapter),
            },
        )
        finish_run(
            handle,
            optimizer_steps=result.optimizer_steps,
            training_tokens=result.training_tokens,
            last_loss=result.last_loss,
        )
        print_cost(f"round {r} train actual", handle.elapsed_s / 3600.0)
    except BaseException:
        finish_run(handle, status="failed")
        raise
    finally:
        del model
        release_cuda_cache()
        sync_run(run_dir)
    return 0


def _val_metrics(rdir: Path) -> dict[str, Any] | None:
    m = rdir / "eval" / "val" / "val_mixed_100" / "greedy" / "metrics.json"
    return read_json(m) if m.exists() else None


def _val_changed_fraction(prev: Path, cur: Path) -> float | None:
    a = prev / "eval" / "val" / "val_mixed_100" / "greedy" / "samples.jsonl"
    b = cur / "eval" / "val" / "val_mixed_100" / "greedy" / "samples.jsonl"
    if not (a.exists() and b.exists()):
        return None
    pa = {s.problem_id: s.completion for s in read_samples(a)}
    pb = {s.problem_id: s.completion for s in read_samples(b)}
    common = sorted(set(pa) & set(pb))
    return float(np.mean([pa[p] != pb[p] for p in common])) if common else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_finalize(cfg: dict[str, Any], args: Any) -> int:
    """Make the seed dir a tasks/03-style run dir whose final adapter is the last round's."""
    seed, _, _, _, _, run_dir = _common(cfg, args)
    rounds, n = int(cfg["rounds"]), int(cfg["samples_per_round"])
    if (run_dir / "budgets.json").exists() and not getattr(args, "force", False):
        print(f"[iter-rft] {run_dir} already finalized — skipping")
        return 0
    rdirs = [round_dir(run_dir, r) for r in range(1, rounds + 1)]
    missing = [str(d) for d in rdirs if not (d / "budgets.json").exists()]
    if missing:
        raise SystemExit(f"rounds not trained yet: {missing}")
    budgets = [read_json(d / "budgets.json") for d in rdirs]
    last_cfg_path = rdirs[-1] / "config.yaml"
    import yaml

    with last_cfg_path.open(encoding="utf-8") as f:
        last_cfg = yaml.safe_load(f)
    per_round = []
    gpu_hours = 0.0
    for r, (d, b) in enumerate(zip(rdirs, budgets, strict=True), start=1):
        meta = read_json(d / "meta.json")
        smeta = (
            read_json(d / "sampling" / "meta.json")
            if (d / "sampling" / "meta.json").exists()
            else {}
        )
        gpu_hours += float(meta.get("gpu_hours_actual") or 0.0) + float(
            smeta.get("gpu_hours_actual") or 0.0
        )
        val = _val_metrics(d)
        per_round.append(
            {
                "round": r,
                "budgets": {
                    k: b[k]
                    for k in (
                        "prompts",
                        "completions_consumed",
                        "training_tokens",
                        "optimizer_steps",
                    )
                },
                "diagnostics": read_json(d / "diagnostics.json"),
                "train_gpu_hours": meta.get("gpu_hours_actual"),
                "sampling_gpu_hours": smeta.get("gpu_hours_actual"),
                "val_greedy": None
                if val is None
                else {
                    k: val.get(k)
                    for k in (
                        "accuracy",
                        "ci_low",
                        "ci_high",
                        "truncation_rate",
                        "mean_completion_tokens",
                        "n_problems",
                    )
                },
                "val_outputs_changed_vs_previous_round": None
                if r == 1
                else _val_changed_fraction(rdirs[r - 2], d),
            }
        )
    last_final = Path(budgets[-1]["final_adapter"])
    final = run_dir / "adapter" / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(last_final, final)
    last_epoch = Path(budgets[-1]["epoch_adapters"][-1])
    (final / "SOURCE.txt").write_text(
        f"copy of {rdirs[-1].name}/adapter/{last_epoch.name} (last epoch of the last round; no early stopping)\n"
    )
    weights = final / "adapter_model.safetensors"
    resolved = {
        **{
            k: v
            for k, v in last_cfg.items()
            if k
            not in ("round", "round_source", "init_adapter", "budgets", "n_examples", "draw_file")
        },
        "kind": "iter_rft_train",
        "round_dirs": [str(d) for d in rdirs],
        "samples_per_round": n,
        "final_adapter_sha256": _sha256(weights) if weights.exists() else None,
    }
    handle = start_run(
        run_dir,
        resolved,
        run_id=f"iter_rft_{cfg['arm']}_seed{seed}",
        extra_meta={"arm": cfg["arm"], "seed": seed},
    )
    prompts_used = max(b["prompts"] for b in budgets)
    write_json(run_dir / "rounds.json", {"arm": cfg["arm"], "seed": seed, "rounds": per_round})
    write_json(
        run_dir / "budgets.json",
        {
            "prompts": prompts_used,
            "prompts_parent_split": budgets[0]["prompts_parent_split"],
            "completions_available": budgets[0]["prompts_parent_split"] * n * rounds,
            "completions_consumed": sum(b["completions_consumed"] for b in budgets),
            "training_tokens": sum(b["training_tokens"] for b in budgets),
            "optimizer_steps": sum(b["optimizer_steps"] for b in budgets),
            "rounds": rounds,
            "samples_per_round": n,
            "epochs": int(cfg["epochs"]),  # per round; final = last epoch of the last round
            "learning_rate": float(cfg["learning_rate"]),
            "batch_size": budgets[-1]["batch_size"],
            "final_adapter": str(final),
            "epoch_adapters": budgets[-1]["epoch_adapters"],
            "per_round": [p["budgets"] for p in per_round],
            "select_mode": "all",
            "max_per_problem": None,
        },
    )
    finish_run(
        handle,
        gpu_hours_actual=round(gpu_hours, 6),
        gpu_hours_note="sum of the rounds' sampling and training",
    )
    sync_run(run_dir)
    for p in per_round:
        v = p["val_greedy"]
        print(
            f"[iter-rft] seed {seed} round {p['round']}: kept {p['budgets']['completions_consumed']}, "
            f"pass rate {p['diagnostics']['pass_rate_mean']:.3f}, val "
            + (
                "not evaluated"
                if v is None
                else f"{v['accuracy']:.3f} [{v['ci_low']:.3f},{v['ci_high']:.3f}] trunc {100 * v['truncation_rate']:.1f}%"
            )
        )
    return 0


def cli_main(args: Any) -> int:
    load_env()
    cfg = load_arm_config(args.config)
    stages = {"sample": stage_sample, "train": stage_train, "finalize": stage_finalize}
    stage = str(getattr(args, "stage", "train"))
    if stage not in stages:
        print(f"unknown stage {stage!r}; choose from {sorted(stages)}", file=sys.stderr)
        return 2
    seed_everything(
        int(args.seed if getattr(args, "seed", None) is not None else cfg.get("seed", 1))
    )
    return int(stages[stage](cfg, args) or 0)
