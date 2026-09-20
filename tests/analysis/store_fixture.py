"""A tiny synthetic run root laid out like the artifact store (test infrastructure only). AGENT-OWNED.

Six arms × 3 seeds, two single-seed controls, base, one reference model, plus the things that
must never load (stale units, a Gemma stub, a no-EOS ablation, a sweep run that was not chosen).
Correctness is drawn from a seeded RNG; no expected values live here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.data.generator import write_jsonl
from rlordata.data.tiers import build_splits, tier_from_pass8
from rlordata.sampling.eval_runner import metrics_for, problem_ids_digest, write_samples
from rlordata.train.rft import run_name
from rlordata.types import Problem, Sample
from tests.helpers import small_ood, small_pool

CAP = 4352  # must equal configs/locked/cap.yaml: the protocol check reads the locked file
GRPO_COMPLETIONS = 16
TOKENIZER = "t" * 64
SPLIT_SPEC: dict[str, Any] = {
    "train_easy_100": {"easy": 6},
    "train_mixed_100": {"easy": 2, "medium": 2, "hard": 2},
    "val_mixed_100": {"easy": 2, "medium": 2, "hard": 2},
    "test_300": {"easy": 4, "medium": 4, "hard": 4},
    "train_curated": {"derived_from": "train_mixed_100", "pass8_min": 1, "pass8_max": 7},
}
PROTOCOL = {
    "max_completion_tokens": CAP,
    "max_prompt_tokens": 4096,
    "prompt_template": "Solve.\n\nProblem: {problem_text}\n",
    "answer_regex": "layered",
    "extraction_rule": "v1.6",
    "thinking": False,
    "cap_is_provisional": False,
}
DECODINGS = {
    "greedy": {"name": "greedy", "n": 1, "ks": None, "temperature": 0.0},
    "mean_at_k": {"name": "mean_at_k", "n": 8, "ks": None, "temperature": 1.0},
    "pass_at_k": {"name": "pass_at_k", "n": 64, "ks": [1, 2, 4, 8, 16, 32, 64], "temperature": 1.0},
}
FINAL_UNITS = [
    ("val_mixed_100", "greedy"),
    ("val_mixed_100", "mean_at_k"),
    ("test_300", "greedy"),
    ("test_300", "mean_at_k"),
    ("test_300", "pass_at_k"),
    ("ood_hard_200", "greedy"),
    ("ood_hard_200", "mean_at_k"),
    ("gsm8k_500", "greedy"),
    ("gsm8k_500", "mean_at_k"),
]
ARMS = {
    "rft_easy": ("rft", "train_easy_100", 0.55),
    "rft_mixed": ("rft", "train_mixed_100", 0.60),
    "rft_curated": ("rft", "train_curated", 0.58),
    "grpo_easy": ("grpo", "train_easy_100", 0.65),
    "grpo_mixed": ("grpo", "train_mixed_100", 0.72),
    "grpo_curated": ("grpo", "train_curated", 0.70),
}
LABELS = {
    "rft_easy": "RFT-Easy",
    "rft_mixed": "RFT-Mixed",
    "rft_curated": "RFT-Curated",
    "grpo_easy": "GRPO-Easy",
    "grpo_mixed": "GRPO-Mixed",
    "grpo_curated": "GRPO-Curated",
}
CONTROLS = {"random_reward": 0.40, "format_only": 0.52}
CHOSEN = {"rft_easy": (5e-05, 8), "rft_mixed": (5e-05, 4), "rft_curated": (1e-05, 4)}
GRID = [(lr, ep) for lr in (1e-05, 5e-05, 1e-04) for ep in (2, 4, 8)]


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(obj, sort_keys=True), encoding="utf-8")


def make_splits(splits_dir: Path) -> dict[str, list[Problem]]:
    pool = [
        Problem(**{**p.to_dict(), "pass8": i % 9, "tier": tier_from_pass8(i % 9)})
        for i, p in enumerate(small_pool(800))
    ]
    splits = build_splits(pool, seed=3, split_spec=SPLIT_SPEC)
    splits["ood_hard_200"] = [
        Problem(
            **{
                **p.to_dict(),
                "pass8": i % 9,
                "tier": tier_from_pass8(i % 9),
                "split": "ood_hard_200",
            }
        )
        for i, p in enumerate(small_ood(6))
    ]
    for name, problems in splits.items():
        write_jsonl(problems, splits_dir / f"{name}.jsonl")
    return splits


def _gsm8k(n: int = 5) -> list[Problem]:
    return [
        Problem(
            problem_id=hashlib.sha256(f"gsm{i}".encode()).hexdigest(),
            text=f"q{i}",
            answer=i,
            pipeline={"source": "gsm8k", "index": i},
            range_scale="S",
            n_filters=0,
            n_transforms=0,
            total_steps=0,
        )
        for i in range(n)
    ]


def write_unit(
    unit_dir: Path,
    problems: list[Problem],
    *,
    split: str,
    decoding: str,
    seed: int,
    who: str,
    p_correct: float,
    p_trunc: float,
    model_kind: str = "base",
    sampler: str = "vllm",
    extra_config: dict[str, Any] | None = None,
    sanity: dict[str, Any] | None = None,
) -> None:
    dec = DECODINGS[decoding]
    subset = "test_300_first100" if decoding == "pass_at_k" else None
    config = {
        **PROTOCOL,
        "kind": "eval",
        "split": split,
        "subset": subset,
        "seed": seed,
        "decoding": dec,
        "model": {"id": "toy", "kind": model_kind},
        "sampler": {"sampler": sampler},
        "problem_ids_sha256": problem_ids_digest(problems),
        "n_problems": len(problems),
        "tokenizer_sha256_trainer": TOKENIZER,
        "tokenizer_sha256_vllm": TOKENIZER,
        **(extra_config or {}),
    }
    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    rng = np.random.default_rng(
        int(hashlib.sha256(f"{who}|{split}|{decoding}".encode()).hexdigest()[:8], 16)
    )
    template = PROTOCOL["prompt_template"]
    if model_kind != "base":
        template = "<chat>" + template + "</chat>"
    samples = []
    for i, p in enumerate(problems):
        bump = {"easy": 0.25, "medium": 0.0, "hard": -0.25}.get(p.tier, 0.0)
        for j in range(dec["n"]):
            truncated = bool(rng.random() < p_trunc)
            correct = bool(not truncated and rng.random() < min(max(p_correct + bump, 0.02), 0.98))
            samples.append(
                Sample(
                    run_id=f"{who}_{split}_{decoding}",
                    config_hash=cfg_hash,
                    seed=seed,
                    arm=who.split("/")[0],
                    data_condition=split,
                    problem_id=p.problem_id,
                    tier=p.tier,
                    prompt=template.format(problem_text=p.text),
                    completion=f"[{who} p{i} s{j}] "
                    + ("…" if truncated else f"Answer: {p.answer if correct else p.answer + 1}"),
                    extracted_answer=None if truncated else (p.answer if correct else p.answer + 1),
                    correct=correct,
                    reward=float(correct),
                    n_tokens=CAP if truncated else int(200 + 3000 * rng.random()),
                    truncated=truncated,
                    extraction_failed=truncated,
                )
            )
    unit_dir.mkdir(parents=True, exist_ok=True)
    write_samples(samples, unit_dir / "samples.jsonl")
    metrics = metrics_for(samples, ks=dec["ks"], seed=seed, cap=CAP, split=split)
    metrics.update({"sampler": sampler, "split": split, "decoding": decoding, "seed": seed})
    _dump(unit_dir / "metrics.json", metrics)
    _dump(unit_dir / "config.yaml", config)
    (unit_dir / "config_hash.txt").write_text(cfg_hash + "\n", encoding="utf-8")
    _dump(
        unit_dir / "meta.json",
        {
            "status": "finished",
            "n_samples": len(samples),
            "n_samples_planned": len(samples),
            "sampler": sampler,
            "git_dirty": False,
            "git_sha": "abc1234",
            "gpu_hours_actual": 0.01,
            "gpu_rate_usd_per_hour": 4.29,
        },
    )
    if sanity is not None:
        _dump(unit_dir / "sanity.json", sanity)


def _problems_for(splits: dict[str, list[Problem]], split: str, decoding: str) -> list[Problem]:
    if split == "gsm8k_500":
        return _gsm8k()
    return splits[split][:100] if decoding == "pass_at_k" else splits[split]


def _trained_run(
    root: Path,
    rel: str,
    splits: dict[str, list[Problem]],
    *,
    who: str,
    method: str,
    data_condition: str,
    seed: int,
    p_correct: float,
    p_trunc: float,
    lr: float | None = None,
    epochs: int | None = None,
) -> None:
    run_dir = root / "runs" / rel
    final_rel = f"runs/{rel}/adapter/" + ("final" if method == "rft" else "step_300")
    weights_rel = f"runs/{rel}/adapter/final"
    (root / weights_rel).mkdir(parents=True, exist_ok=True)
    (root / weights_rel / "adapter_model.safetensors").write_bytes(f"weights of {who}".encode())
    lora = {"r": 64, "lora_alpha": 16, "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "use_rslora": False, "use_dora": False, "target_modules": ["q_proj", "v_proj"]}  # fmt: skip
    _dump(root / weights_rel / "adapter_config.json", lora)
    n_prompts = len(splits[data_condition])
    budgets: dict[str, Any] = {
        "prompts": n_prompts,
        "completions_available": GRPO_COMPLETIONS,
        "training_tokens": 1000 * (seed + 5),
        "final_adapter": final_rel,
    }
    if method == "rft":
        epoch_adapters = [f"runs/{rel}/adapter/epoch_{e + 1}" for e in range(int(epochs or 1))]
        (root / weights_rel / "SOURCE.txt").write_text(
            f"copy of epoch_{epochs}\n", encoding="utf-8"
        )
        budgets |= {
            "completions_consumed": GRPO_COMPLETIONS // 2,
            "optimizer_steps": 40 * int(epochs or 1),
            "learning_rate": lr,
            "epochs": epochs,
            "epoch_adapters": epoch_adapters,
            "prompts_parent_split": n_prompts,
            "per_tier_prompts": {"easy": n_prompts},
            "per_tier_examples": {"easy": GRPO_COMPLETIONS // 2},
        }
        _dump(
            run_dir / "config.yaml",
            {"append_eos": True, "seed": seed, "data_condition": data_condition},
        )
    else:
        step = root / final_rel
        step.mkdir(parents=True, exist_ok=True)
        _dump(step / "trainer_state.json", {"global_step": 300})
        _dump(step / "adapter_config.json", lora)
        budgets |= {
            "completions_consumed": GRPO_COMPLETIONS,
            "optimizer_steps": 300,
            "max_steps": 300,
            "eval_step": 300,
            "result_bearing": True,
            "final_adapter_trained": weights_rel,
            "step_adapters": {
                "100": f"runs/{rel}/adapter/step_100",
                "200": f"runs/{rel}/adapter/step_200",
                "300": final_rel,
            },
        }
        _dump(
            run_dir / "grpo_config.json",
            {"beta": 0.0, "max_steps": 300, "seed": seed, "output_dir": rel},
        )
        name = "random_bernoulli" if "random_reward" in rel else "verify_binary"
        (run_dir / "reward_records.jsonl").write_text(
            "".join(
                json.dumps({"reward": float(i % 2), "reward_name": name, "step": i}) + "\n"
                for i in range(GRPO_COMPLETIONS)
            ),
            encoding="utf-8",
        )
        rows = [
            {"step": s, "summary": False, "reward_mean": 0.4 + 0.001 * s, "frac_reward_zero_std": 0.2,
             "completions_mean_length": 500.0 + s, "completions_clipped_ratio": 0.01,
             "per_tier": {"easy": {"n": 16, "frac_reward_zero_std": 0.5, "reward_mean": 0.8}}}
            for s in range(1, 300)
        ]  # fmt: skip
        rows.append({"step": 300, "summary": True, "reward_mean": None})
        (run_dir / "train_log.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        _dump(
            run_dir / "curves.json",
            {
                "val_greedy": {
                    str(s): {"accuracy": 0.6, "ci": [0.5, 0.7], "truncation_rate": 0.02}
                    for s in (100, 200, 300)
                }
            },
        )
    _dump(run_dir / "budgets.json", budgets)
    _dump(
        run_dir / "meta.json",
        {
            "status": "finished",
            "git_dirty": False,
            "git_sha": "abc1234",
            "gpu_hours_actual": 1.5,
            "seed": seed,
        },
    )
    (run_dir / "config_hash.txt").write_text(
        hashlib.sha256(rel.encode()).hexdigest() + "\n", encoding="utf-8"
    )
    for split, decoding in FINAL_UNITS:
        write_unit(
            run_dir / "eval" / "final" / split / decoding,
            _problems_for(splits, split, decoding),
            split=split,
            decoding=decoding,
            seed=seed,
            who=who,
            p_correct=p_correct + 0.01 * seed - (0.3 if split == "ood_hard_200" else 0.0),
            p_trunc=p_trunc * (4 if split == "ood_hard_200" else 1),
            extra_config={
                "adapter": final_rel,
                "eval_set": "final",
                "train_run_dir": f"runs/{rel}",
            },
            sanity={"adapter": final_rel, "eval_set": "final", "protocol_vs_base": "ok"},
        )


def build_store(root: Path) -> dict[str, Any]:
    """Write <root>/{runs,splits}/ and return an analysis config pointing at them."""
    splits = make_splits(root / "splits")
    runs = root / "runs"
    for model, kind, p in (("toy__base", "base", 0.5), ("toy__ref", "instruct", 0.62)):
        for split, decoding in FINAL_UNITS:
            if split == "gsm8k_500":
                continue
            write_unit(runs / "eval" / model / split / decoding, _problems_for(splits, split, decoding),
                       split=split, decoding=decoding, seed=1, who=model, p_correct=p, p_trunc=0.03, model_kind=kind)  # fmt: skip
    write_unit(runs / "transfer_pick" / "toy__base" / "gsm8k_500" / "greedy", _gsm8k(), split="gsm8k_500",
               decoding="greedy", seed=1, who="toy__base", p_correct=0.8, p_trunc=0.0)  # fmt: skip
    for arm, (method, dc, p) in ARMS.items():
        for seed in (1, 2, 3):
            if method == "rft":
                lr, ep = CHOSEN[arm]
                rel = f"rft/{arm}/{run_name(seed, lr, ep)}"
                _trained_run(root, rel, splits, who=f"{arm}/seed{seed}", method=method, data_condition=dc,
                             seed=seed, p_correct=p, p_trunc=0.02, lr=lr, epochs=ep)  # fmt: skip
            else:
                rel = f"grpo/{arm}_s{seed}"
                _trained_run(root, rel, splits, who=f"{arm}/seed{seed}", method=method, data_condition=dc,
                             seed=seed, p_correct=p, p_trunc=0.08)  # fmt: skip
        if method == "rft":
            lr, ep = CHOSEN[arm]
            results = [
                {"learning_rate": g_lr, "epochs": g_ep, "selection_split": "val_mixed_100", "status": "finished",
                 "val_accuracy": 0.67 if (g_lr, g_ep) == (lr, ep) else 0.60, "run_dir": f"runs/rft/{arm}/{run_name(1, g_lr, g_ep)}"}
                for g_lr, g_ep in GRID
            ]  # fmt: skip
            _dump(
                runs / "rft" / arm / "sweep.json",
                {"arm": arm, "seed": 1, "grid": GRID, "results": results},
            )
            _dump(runs / "rft" / arm / "chosen.json",
                  {"arm": arm, "learning_rate": lr, "epochs": ep, "selected_on": "val_mixed_100", "val_accuracy": 0.67,
                   "sweep_run_dir": f"runs/rft/{arm}/{run_name(1, lr, ep)}", "tie_break": "fewer epochs, then lower learning rate"})  # fmt: skip
    for control, p in CONTROLS.items():
        _trained_run(root, f"grpo/grpo_{control}_s1", splits, who=f"{control}/seed1", method="grpo",
                     data_condition="train_mixed_100", seed=1, p_correct=p, p_trunc=0.02)  # fmt: skip

    # ---- things that must never load -------------------------------------------------------
    poison = dict(split="test_300", decoding="greedy", seed=1, p_correct=0.99, p_trunc=0.0)
    test = splits["test_300"]
    lr, ep = CHOSEN["rft_mixed"]
    chosen_run = runs / "rft" / "rft_mixed" / run_name(1, lr, ep)
    write_unit(
        chosen_run / "eval" / "val_bf16merge_stale" / "test_300" / "greedy",
        test,
        who="stale",
        **poison,
    )
    write_unit(
        runs / "eval" / "google__gemma-4-E4B-it" / "test_300" / "greedy",
        test,
        who="gemma",
        sampler="stub",
        **poison,
    )
    write_unit(
        runs
        / "rft_noeos_ablation"
        / "rft_mixed"
        / run_name(1, lr, ep)
        / "eval"
        / "final"
        / "test_300"
        / "greedy",
        test,
        who="noeos",
        **poison,
    )
    write_unit(
        runs / "grpo" / "_failed" / "grpo_mixed_s1" / "eval" / "final" / "test_300" / "greedy",
        test,
        who="failed",
        **poison,
    )
    write_unit(runs / "dev" / "x" / "test_300" / "greedy", test, who="dev", **poison)
    # a sweep run that was not chosen, with a (hypothetical) final eval: must not be read either
    other = runs / "rft" / "rft_mixed" / run_name(1, 1e-04, 8)
    write_unit(other / "eval" / "final" / "test_300" / "greedy", test, who="unchosen", **poison)
    _dump(other / "meta.json", {"status": "finished", "gpu_hours_actual": 2.0})

    return {
        "run_root": str(runs),
        "splits_dir": str(root / "splits"),
        "out_dir": str(root / "out"),
        "store_checksums": None,
        "seed": 0,
        "n_boot": 200,
        "seeds": [1, 2, 3],
        "grpo_completions": GRPO_COMPLETIONS,
        "gpu_rate": {"recorded_usd_per_hour": 4.29, "billed_usd_per_hour": 3.29},
        "base": {"label": "Base", "model_id": "toy", "dir": "eval/toy__base", "transfer_dir": "transfer_pick/toy__base"},
        "references": [{"label": "Toy-Ref", "dir": "eval/toy__ref"}],
        "arms": {
            arm: {"label": LABELS[arm],
                  "method": method, "data_condition": dc,
                  **({"dir": f"rft/{arm}"} if method == "rft" else {"run_pattern": f"grpo/{arm}_s{{seed}}"})}
            for arm, (method, dc, _) in ARMS.items()
        },
        "controls": {
            "c1_random_reward": {"label": "C1 random reward", "method": "grpo", "data_condition": "train_mixed_100",
                                 "run_pattern": "grpo/grpo_random_reward_s{seed}", "seeds": [1]},
            "c2_format_only": {"label": "C2 format only", "method": "grpo", "data_condition": "train_mixed_100",
                               "run_pattern": "grpo/grpo_format_only_s{seed}", "seeds": [1]},
        },
        "final_units": [list(u) for u in FINAL_UNITS],
        "known_missing": [{"who": "base", "split": "gsm8k_500", "decoding": "mean_at_k", "why": "never evaluated"}],
        "never_results": [
            {"fragment": "_bf16merge_stale", "why": "stale"},
            {"fragment": "/rft_noeos_ablation/", "why": "no EOS"},
            {"fragment": "/grpo/_failed/", "why": "failed"},
            {"fragment": "/dev/", "why": "dev"},
            {"fragment": "google__gemma", "why": "stub"},
        ],
    }  # fmt: skip
