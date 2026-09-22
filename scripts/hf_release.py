"""tasks/07: stage (and, only with --push, upload) the final LoRA adapters with their model cards.

    uv run python scripts/hf_release.py                          # stage under outputs/hf_release/, upload nothing
    uv run python scripts/hf_release.py --push --namespace <hf-user-or-org> [--private]

Without --push nothing leaves the machine. One repo per trained run, named
`rl-or-data-<arm>-seed<k>`; every card states base model, data condition, seed and config hash, the
training budgets, and the run's own evaluation numbers with n, CI and truncation. Numbers come from
the same loader and sanity checks as `make analysis`; nothing is selected on any result.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from rlordata.analysis import loader, report, sanity
from rlordata.analysis.loader import Dataset, Run

REPO_URL = "https://github.com/LakshyaChaudhry/rl-or-data"
COLLECTION_TITLE = "rl-or-data: is it the RL or the data? (LoRA adapters)"  # Hub limit: 60 chars
COLLECTION_DESCRIPTION = (
    "RFT vs GRPO on Qwen3-4B-Base at matched budgets: 3 seeds per arm, controls, iterated RFT. "
    f"{REPO_URL}"
)
# Hub limits (a 400 at create_collection otherwise): title <= 60 chars, description <= 150.
assert len(COLLECTION_TITLE) <= 60 and len(COLLECTION_DESCRIPTION) <= 150, (
    len(COLLECTION_TITLE),
    len(COLLECTION_DESCRIPTION),
)
METRICS = (
    ("val_greedy", "val_mixed_100 greedy"),
    ("test_greedy", "test_300 greedy"),
    ("ood_greedy", "ood_hard_200 greedy"),
    ("test_mean8", "test_300 mean@8"),
    ("gsm8k_greedy", "gsm8k_500 greedy"),
)
ROLE = {
    "rft": "rejection-sampling fine-tuning (RFT): SFT on verified-correct completions sampled once from the base model",
    "grpo": "GRPO with a binary verifiable reward (TRL), 300 steps × 8 prompts × 8 generations",
    "iter_rft": "iterated RFT: 3 rounds × 64 samples per prompt, each round sampled from the current policy and "
    "continuing the previous adapter. **Secondary arm, registered after the primary results were seen** "
    "(PREREGISTRATION §6)",
}
CONTROL_NOTE = {
    "c1_random_reward": "**Control, not a model to use.** Trained with a random Bernoulli(0.5) reward to measure what "
    "GRPO does without a correctness signal; it is worse than the base model by design of the experiment.",
    "c2_format_only": "**Control, not a model to use.** Rewarded only for producing a parseable answer line, never for "
    "being right.",
}


def repo_name(run: Run) -> str:
    return f"rl-or-data-{run.key.replace('_', '-')}-seed{run.seed}"


def card(run: Run, ds: Dataset, results: dict[str, Any], adapter_sha: str, license_id: str) -> str:
    who = f"{run.key}/seed{run.seed}"
    cells = results["runs"][who]["cells"]
    b = results["budgets"][who]
    unit = next(iter(run.units.values()))
    lora = yaml.safe_load(
        (run.run_dir / "adapter" / "final" / "adapter_config.json").read_text(encoding="utf-8")
    )
    rows = []
    for key, label in METRICS:
        c = cells.get(key)
        if c:
            rows.append(
                f"| {label} | {c['value']:.3f} | [{c['ci_low']:.3f}, {c['ci_high']:.3f}] | {c['n_problems']} | "
                f"{100 * c['truncation_rate']:.1f} % | {100 * c['extraction_failure_rate']:.1f} % | `{c['config_hash'][:12]}` |"
            )
    base_cells = results["runs"]["base"]["cells"]
    base_txt = ", ".join(
        f"{label} {base_cells[k]['value']:.3f}" for k, label in METRICS[:3] if base_cells.get(k)
    )
    flag = cells["test_greedy"] and cells["test_greedy"]["truncation_flag"]
    notes = [CONTROL_NOTE[run.key]] if run.key in CONTROL_NOTE else []
    if flag:
        notes.append(
            "**Truncation flag (SPEC §7):** more than 5 % of this run's test_300 greedy completions hit the "
            f"{unit.config['max_completion_tokens']:,}-token cap and were scored wrong by rule; the protocol "
            "does not treat its test number as a headline number as it stands."
        )
    if run.method == "grpo":
        notes.append(
            "The evaluated checkpoint is `adapter/step_300`; this upload is `adapter/final`, saved by the trainer "
            "right after step 300 (byte-identical for the curated arm by direct comparison, by construction for the others)."
        )
    lr, ep = b.get("learning_rate"), b.get("epochs")
    recipe = (
        f"lr {lr:g}, {ep} epochs"
        if isinstance(lr, float) and ep
        else "lr 5e-5, one pass (fixed recipe, SPEC §9)"
    )
    return "\n".join(
        [
            "---",
            f"base_model: {ds.cfg['base']['model_id']}",
            "library_name: peft",
            f"license: {license_id}",
            "tags: [lora, peft, rlvr, grpo, rejection-sampling, procedural-counting]",
            "---",
            "",
            f"# {run.label}, seed {run.seed} — LoRA adapter for `{ds.cfg['base']['model_id']}`",
            "",
            f"One run of the controlled study **“Is it the RL or the data?”** ({REPO_URL}): under matched prompt "
            "and rollout budgets, how much of low-data RLVR's gain comes from data selection and how much from the "
            "RL objective, on procedurally generated counting problems. This is a research artifact for "
            "reproducing that comparison, not a general-purpose model.",
            "",
            *[n + "\n" for n in notes],
            "| | |",
            "|---|---|",
            f"| base model | `{ds.cfg['base']['model_id']}` |",
            f"| method | {ROLE[str(run.method)]} |",
            f"| data condition | `{run.data_condition}` ({b['prompts']} prompts used) |",
            f"| seed | {run.seed} |",
            f"| run config hash | `{run.config_hash}` |",
            f"| git SHA of the training code | `{(run.meta or {}).get('git_sha', 'unknown')}` |",
            f"| adapter sha256 | `{adapter_sha}` |",
            f"| LoRA | r={lora['r']}, alpha={lora['lora_alpha']}, dropout={lora['lora_dropout']}, targets {', '.join(sorted(lora['target_modules']))} |",
            f"| recipe | {recipe}; AdamW, weight decay 0, grad clip 1.0, cosine schedule, bf16 |",
            f"| budgets | completions available {b['completions_available']:,}, consumed {b['completions_consumed']:,}; "
            f"training tokens {b['training_tokens']:,}; optimizer steps {b['optimizer_steps']:,}; "
            f"{b['gpu_hours_train']:.2f} GPU-h on 1× H100 PCIe |",
            f"| completion cap | {unit.config['max_completion_tokens']:,} tokens, identical for every arm, training and evaluation |",
            "",
            "## Evaluation of this run",
            "",
            "vLLM, no thinking mode, one evaluation per unit; bootstrap 95 % CI over problems (10,000 resamples). "
            "A completion cut at the cap is scored wrong whatever it contains.",
            "",
            "| unit | accuracy | 95 % CI | n problems | truncation | extraction failure | config hash |",
            "|---|---|---|---|---|---|---|",
            *rows,
            "",
            f"Base model, same protocol: {base_txt}. Arm-level results (three seeds, paired contrasts, the "
            f"pre-registered criterion and its caveats) are in `results/` of the repository; a single seed is not a result. "
            "Greedy numbers re-generated on a different host differ by a few problems in a hundred (see the repository's "
            "Phase 5 README).",
            "",
            "## Use",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'base = AutoModelForCausalLM.from_pretrained("{ds.cfg["base"]["model_id"]}", torch_dtype="bfloat16")',
            f'model = PeftModel.from_pretrained(base, "<namespace>/{repo_name(run)}")',
            f'tok = AutoTokenizer.from_pretrained("{ds.cfg["base"]["model_id"]}")',
            "```",
            "",
            "Prompt template used for training and evaluation (no chat template; the base model is not instruction-tuned):",
            "",
            "```",
            str(unit.config["prompt_template"]).rstrip(),
            "```",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(loader.DEFAULT_CONFIG))
    ap.add_argument("--out", default="outputs/hf_release")
    ap.add_argument("--license", default="apache-2.0", help="license id written into every card")
    ap.add_argument("--skip-secondary", action="store_true", help="leave the iterated-RFT arm out")
    ap.add_argument(
        "--push", action="store_true", help="UPLOAD. Without it nothing leaves the machine."
    )
    ap.add_argument("--namespace", default=None, help="HF user or org (required with --push)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)
    if args.push and not args.namespace:
        raise SystemExit("--push needs --namespace")

    cfg = loader.load_config(args.config)
    cache = Path(cfg["out_dir"]) / ".cache"
    ds = loader.load_dataset(cfg, cache_dir=cache / "units", jobs=4)
    rep = sanity.cross_run_checks(ds, splits_dir=cfg["splits_dir"], cache_dir=cache)
    if rep.failures:
        raise SystemExit(f"sanity failed, nothing staged: {rep.failures[:3]}")
    results = report.build_results(ds)
    hashes = sanity.adapter_hashes(ds, cache_path=cache / "adapter_sha256.json")
    secondary = {k for k, a in cfg["arms"].items() if a.get("secondary")}
    out = Path(args.out)
    staged = []
    for run in ds.trained_runs():
        if args.skip_secondary and run.key in secondary:
            continue
        src = run.run_dir / "adapter" / "final"
        dst = out / repo_name(run)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (dst / name).exists():
                shutil.copy2(src / name, dst / name)
        who = f"{run.key}/seed{run.seed}"
        (dst / "README.md").write_text(
            card(run, ds, results, hashes[who]["sha256"], args.license), encoding="utf-8"
        )
        staged.append((repo_name(run), dst))
    (out / "manifest.json").write_text(
        json.dumps({"collection": COLLECTION_TITLE, "repos": [n for n, _ in staged]}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for _, d in staged for f in d.iterdir()) / 1e9
    print(f"[hf_release] staged {len(staged)} adapters ({size:.1f} GB) under {out}/")
    if not args.push:
        print(
            "[hf_release] dry run: nothing uploaded. Re-run with --push --namespace <name> to publish."
        )
        return 0

    from huggingface_hub import (
        HfApi,
    )  # imported here so a dry run needs neither the package nor a token

    api = HfApi()
    collection = api.create_collection(
        COLLECTION_TITLE,
        namespace=args.namespace,
        description=COLLECTION_DESCRIPTION,
        private=args.private,
        exists_ok=True,
    )
    for name, folder in staged:
        repo_id = f"{args.namespace}/{name}"
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, folder_path=str(folder), commit_message="rl-or-data release"
        )
        api.add_collection_item(collection.slug, item_id=repo_id, item_type="model", exists_ok=True)
        print(f"[hf_release] pushed {repo_id}")
    print(f"[hf_release] collection: https://huggingface.co/collections/{collection.slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
