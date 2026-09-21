"""tasks/05: loader allow-list, cross-run sanity, deterministic outputs, blank verdicts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from rlordata.analysis import loader, report, sanity
from rlordata.core import evaluate
from tests.analysis.store_fixture import CHOSEN, build_store
from tests.helpers import REPO


@pytest.fixture(scope="module", autouse=True)
def _fast_core_bootstrap() -> Any:
    """compute_metrics always runs 10,000 resamples; 220 fixture units do not need that many.

    The stand-in is the same function with a smaller default, used both when the fixture's
    metrics.json files are written and when they are recomputed, so they still have to agree.
    """
    real = evaluate.bootstrap_ci

    def quick(
        scores: Any, n_boot: int = 200, alpha: float = 0.05, seed: int = 0
    ) -> tuple[float, float]:
        return real(scores, n_boot=n_boot, alpha=alpha, seed=seed)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(evaluate, "bootstrap_ci", quick)
        yield


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("store")
    return root, build_store(root)


def _copy(store: tuple[Path, dict[str, Any]], tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A private, mutable copy of the store with the config re-pointed at it."""
    src, cfg = store
    dst = tmp_path / "store"
    shutil.copytree(src, dst)
    text = json.dumps(cfg).replace(str(src), str(dst))
    return dst, json.loads(text)


def _checks(cfg: dict[str, Any]) -> sanity.CrossRunReport:
    ds = loader.load_dataset(cfg)
    return sanity.cross_run_checks(ds, splits_dir=cfg["splits_dir"])


def _run(cfg: dict[str, Any], tmp_path: Path, out: str, *extra: str) -> int:
    path = tmp_path / f"{out}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return report.main(["--config", str(path), "--out", str(tmp_path / out), "--jobs", "1", *extra])


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_only_allow_listed_final_units_load(store: tuple[Path, dict[str, Any]]) -> None:
    _, cfg = store
    ds = loader.load_dataset(cfg)
    assert {k: [r.seed for r in v] for k, v in ds.arms.items()} == {
        k: [1, 2, 3] for k in cfg["arms"]
    }
    rels = [u.rel for r in ds.all_runs() for u in r.units.values()]
    for fragment in ("stale", "gemma", "noeos", "_failed", "dev/", "lr0.0001_ep8"):
        assert not any(fragment in rel for rel in rels), fragment
    for arm, (lr, ep) in CHOSEN.items():
        assert all(f"lr{lr:g}_ep{ep}" in r.rel for r in ds.arms[arm])
    for run in ds.trained_runs():
        assert all("/eval/final/" in u.rel for u in run.units.values())
    assert [(m["who"], m["split"], m["decoding"]) for m in ds.missing] == [
        ("base", "gsm8k_500", "mean_at_k")
    ]


def test_guard_refuses_never_results(store: tuple[Path, dict[str, Any]]) -> None:
    root, cfg = store
    runs = root / "runs"
    for rel in (
        "eval/google__gemma-4-E4B-it/test_300/greedy",
        "rft_noeos_ablation/rft_mixed",
        "grpo/_failed/grpo_mixed_s1",
        "dev/x/test_300/greedy",
        "rft/rft_mixed/seed1_lr5e-05_ep4/eval/val_bf16merge_stale/test_300/greedy",
    ):
        with pytest.raises(loader.ExcludedPathError):
            loader.load_units([runs / rel], runs, cfg)
    present = {e["fragment"]: e["paths"] for e in loader.excluded_present(runs, cfg)}
    assert all(
        present[f]
        for f in (
            "_bf16merge_stale",
            "/rft_noeos_ablation/",
            "/grpo/_failed/",
            "/dev/",
            "google__gemma",
        )
    )


def test_unit_cache_does_not_change_anything(
    store: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    _, cfg = store
    a = loader.load_dataset(cfg)
    loader.load_dataset(cfg, cache_dir=tmp_path / "cache")
    b = loader.load_dataset(cfg, cache_dir=tmp_path / "cache")  # second pass reads the cache
    ua, ub = a.base.units[("test_300", "pass_at_k")], b.base.units[("test_300", "pass_at_k")]
    assert (
        ua.metrics == ub.metrics
        and (ua.scores == ub.scores).all()
        and ua.prompt_digest == ub.prompt_digest
    )


# ---------------------------------------------------------------------------
# cross-run sanity: passes on a clean store, fails loudly on each corruption
# ---------------------------------------------------------------------------


def test_clean_store_passes_and_lists_known_gaps(store: tuple[Path, dict[str, Any]]) -> None:
    rep = _checks(store[1])
    assert rep.failures == []
    text = rep.to_markdown()
    assert "gsm8k_500/mean_at_k does not exist" in text and "google__gemma" in text
    assert "NOT VERIFIED here" in text  # GRPO step_300 weights are not in the fixture either
    assert "exceed 5 % truncation on test_300 greedy" in text


def _edit_yaml(path: Path, **updates: Any) -> None:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg.update(updates)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _edit_json(path: Path, **updates: Any) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj.update(updates)
    path.write_text(json.dumps(obj), encoding="utf-8")


UNIT = "grpo/grpo_mixed_s2/eval/final/test_300/greedy"


def _different_cap(runs: Path) -> None:
    _edit_yaml(runs / UNIT / "config.yaml", max_completion_tokens=2048)


def _different_template(runs: Path) -> None:
    _edit_yaml(runs / UNIT / "config.yaml", prompt_template="Answer quickly: {problem_text}")


def _same_adapter_for_two_seeds(runs: Path) -> None:
    src = runs / "grpo/grpo_mixed_s1/adapter/final/adapter_model.safetensors"
    shutil.copy(src, runs / "grpo/grpo_mixed_s2/adapter/final/adapter_model.safetensors")


def _no_budgets(runs: Path) -> None:
    (runs / "rft/rft_easy/seed3_lr5e-05_ep8/budgets.json").unlink()


def _not_the_final_checkpoint(runs: Path) -> None:
    _edit_json(runs / UNIT / "sanity.json", adapter="runs/grpo/grpo_mixed_s2/adapter/step_200")


def _stored_metrics_disagree(runs: Path) -> None:
    _edit_json(runs / UNIT / "metrics.json", accuracy=0.999)


def _stub_sampler(runs: Path) -> None:
    _edit_yaml(runs / UNIT / "config.yaml", sampler={"sampler": "stub"})


def _chosen_is_not_the_val_argmax(runs: Path) -> None:
    sweep = runs / "rft/rft_curated/sweep.json"
    obj = json.loads(sweep.read_text(encoding="utf-8"))
    obj["results"][-1]["val_accuracy"] = 0.99
    sweep.write_text(json.dumps(obj), encoding="utf-8")


def _selected_on_test(runs: Path) -> None:
    _edit_json(runs / "rft/rft_curated/chosen.json", selected_on="test_300")


def _unit_missing(runs: Path) -> None:
    shutil.rmtree(runs / "grpo/grpo_easy_s1/eval/final/ood_hard_200/greedy")


def _wrong_problems(runs: Path) -> None:
    _edit_yaml(runs / UNIT / "config.yaml", problem_ids_sha256="0" * 64)


def _no_eos(runs: Path) -> None:
    _edit_yaml(runs / "rft/rft_mixed/seed2_lr5e-05_ep4/config.yaml", append_eos=False)


def _grpo_recipe_differs(runs: Path) -> None:
    _edit_json(runs / "grpo/grpo_curated_s3/grpo_config.json", beta=0.04)


def _too_few_rollouts(runs: Path) -> None:
    path = runs / "grpo/grpo_easy_s2/reward_records.jsonl"
    path.write_text(
        "".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:-3]), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("corrupt", "expect"),
    [
        (_different_cap, "max_completion_tokens differs"),
        (_different_template, "prompt_template differs"),
        (_same_adapter_for_two_seeds, "byte-identical adapters"),
        (_no_budgets, "budgets.json"),
        (_not_the_final_checkpoint, "final adapter is"),
        (_stored_metrics_disagree, "stored accuracy"),
        (_stub_sampler, "only vLLM numbers are results"),
        (_chosen_is_not_the_val_argmax, "val argmax"),
        (_selected_on_test, "selected_on='test_300'"),
        (_unit_missing, "ood_hard_200/greedy does not exist"),
        (_wrong_problems, "differs from the committed split"),
        (_no_eos, "append_eos"),
        (_grpo_recipe_differs, "grpo_config differs"),
        (_too_few_rollouts, "reward records"),
    ],
)
def test_each_corruption_fails_loudly(
    store: tuple[Path, dict[str, Any]], tmp_path: Path, corrupt: Any, expect: str
) -> None:
    root, cfg = _copy(store, tmp_path)
    corrupt(root / "runs")
    failures = _checks(cfg).failures
    assert any(expect in f for f in failures), failures


def test_split_leak_fails(store: tuple[Path, dict[str, Any]], tmp_path: Path) -> None:
    root, cfg = _copy(store, tmp_path)
    test_line = (root / "splits/test_300.jsonl").read_text(encoding="utf-8").splitlines()[0]
    with (root / "splits/train_easy_100.jsonl").open("a", encoding="utf-8") as f:
        f.write(test_line + "\n")
    assert any("shared problem_id" in f for f in _checks(cfg).failures)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def outputs(
    store: tuple[Path, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    root, cfg = store
    tmp = tmp_path_factory.mktemp("outputs")
    before = _tree(root)
    assert _run(cfg, tmp, "a") == 0
    assert _run(cfg, tmp, "b", "--no-cache") == 0
    assert _tree(root) == before, "the run root was written to"
    return tmp / "a", tmp / "b"


def test_outputs_are_complete_and_byte_reproducible(outputs: tuple[Path, Path]) -> None:
    a, b = outputs
    names = sorted(
        str(p.relative_to(a)) for p in a.rglob("*") if p.is_file() and ".cache" not in p.parts
    )
    assert names == sorted(
        str(p.relative_to(b)) for p in b.rglob("*") if p.is_file() and ".cache" not in p.parts
    )
    for want in ("hypotheses.md", "how_could_this_be_wrong.md", "sanity.md", "sanity.json", "analysis_config.yaml",
                 "tables/results.md", "tables/contrasts.md", "tables/units.csv", "tables/arms.csv", "tables/per_tier.csv",
                 "tables/budgets.csv", "tables/truncation.csv", "tables/contrasts.csv"):  # fmt: skip
        assert want in names, want
    assert len([n for n in names if n.startswith("figures/") and n.endswith(".png")]) == 9
    for name in names:
        if name == "analysis_meta.json":  # git SHA, versions, wall-clock: provenance, not a result
            continue
        assert (a / name).read_bytes() == (b / name).read_bytes(), (
            f"{name} differs between two runs"
        )


def test_every_verdict_line_is_blank(outputs: tuple[Path, Path]) -> None:
    lines = (outputs[0] / "hypotheses.md").read_text(encoding="utf-8").splitlines()
    verdicts = [line for line in lines if line.startswith("**Verdict")]
    assert len(verdicts) == 6  # H1, H2, H3, P1, P2, controls
    assert all(re.fullmatch(r"\*\*Verdict \(Laksh\):\*\* _+", v) for v in verdicts)
    text = "\n".join(lines)
    for phrase in ("supported", "confirmed", "falsified.", "refuted"):
        assert phrase not in text.lower().replace("also falsified if", "").replace(
            "falsified if", ""
        )
    assert "RFT tried 9 configurations per arm" in text and "No numeric threshold exists" in text
    assert "not evaluable" in text  # single-seed controls


def test_no_accuracy_cell_lacks_ci_truncation_n_or_seed(outputs: tuple[Path, Path]) -> None:
    text = (outputs[0] / "tables/results.md").read_text(encoding="utf-8")
    header: list[str] = []
    n_cells = 0
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if any(h in cells[0] for h in ("arm", "seed", "model", "run")) and not re.search(
            r"\d\.\d{3}", line
        ):
            header = cells
            continue
        for title, cell in zip(header, cells, strict=False):
            if "(n=" not in title:
                continue
            n_cells += 1
            assert cell == "missing" or re.search(
                r"[-+]?\d\.\d{3}.* \[[-+]?\d\.\d{3}, [-+]?\d\.\d{3}\].*tr \d+\.\d%", cell
            ), (title, cell)
        if any("(n=" in h for h in header):  # the seed is a column, or is named inside the cell
            assert {"seed", "seeds", "model"} & set(header) or "seeds " in line, header
    assert n_cells > 100
    # the missing base unit is shown as missing, never as a zero
    base_row = next(
        line
        for line in text.splitlines()
        if line.startswith("| Base |") and "gsm8k" not in line and line.count("|") > 10
    )
    assert "missing" in base_row and "0.000 [" not in base_row


def test_contrasts_carry_truncation_and_flags(outputs: tuple[Path, Path]) -> None:
    text = (outputs[0] / "tables/contrasts.md").read_text(encoding="utf-8")
    for key in ("H1_rft", "H1_grpo", "H2_num", "H2_den", "H3", "C1", "C2"):
        assert f"## {key} " in text
    rows = [
        line for line in text.splitlines() if line.startswith(("| test greedy", "| ood greedy"))
    ]
    assert rows and all(
        re.search(r"\d+\.\d% .*vs .*\d+\.\d%", r) for r in rows
    )  # truncation A vs B
    assert all(" pp |" in r for r in rows if "(tr " not in r)  # the ratio table has no gap column
    assert "not a headline number as it stands" in text  # the fixture's GRPO arms are over 5 %
    wrong = (outputs[0] / "how_could_this_be_wrong.md").read_text(encoding="utf-8")
    for item in (
        "Truncation gap",
        "Extraction-failure gap",
        "Budget gap",
        "Tier-composition gap",
        "Seed range overlap",
    ):
        assert wrong.count(f"**{item}") == 5, item  # five headline contrasts


def test_sanity_failure_writes_no_tables(
    store: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    root, cfg = _copy(store, tmp_path)
    _different_cap(root / "runs")
    assert _run(cfg, tmp_path, "failed") == 1
    assert (tmp_path / "failed/sanity.md").exists()
    assert (
        not (tmp_path / "failed/tables").exists()
        and not (tmp_path / "failed/hypotheses.md").exists()
    )


def test_refuses_to_write_into_the_run_root(
    store: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    root, cfg = store
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit):
        report.main(["--config", str(path), "--out", str(root / "runs" / "analysis")])


def test_structure_does_not_depend_on_test_or_ood_results(
    store: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Flip every test/ood outcome: the same rows, columns and contrasts must come out."""
    root, cfg = _copy(store, tmp_path)
    first = report.build_results(loader.load_dataset(cfg))
    for samples in (root / "runs").rglob("samples.jsonl"):
        if (
            not any(s in str(samples) for s in ("/test_300/", "/ood_hard_200/"))
            or "/eval/final/" not in str(samples)
            and "/eval/toy" not in str(samples)
        ):
            continue
        rows = [json.loads(line) for line in samples.read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["correct"] = bool(not r["truncated"] and not r["correct"])
        samples.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
    second = report.build_results(loader.load_dataset(cfg))

    drop = ("flagged_seeds", "flagged_seeds_a", "flagged_seeds_b")

    def shape(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: shape(v) for k, v in x.items() if k not in drop}
        if isinstance(x, (list, tuple)):
            return len(x)
        # a rate with an empty denominator is None (e.g. no correct completion at all)
        return "number" if x is None or isinstance(x, float) else type(x).__name__

    a = first["runs"]["base"]["cells"]["test_greedy"]["value"]
    assert a != second["runs"]["base"]["cells"]["test_greedy"]["value"]
    assert shape(first) == shape(second)


def test_default_config_names_real_protocol_files() -> None:
    cfg = loader.load_config(REPO / "configs/analysis/default.yaml")
    assert cfg["n_boot"] == 10_000 and cfg["seeds"] == [1, 2, 3]
    confirmatory = {k for k, a in cfg["arms"].items() if not a.get("secondary")}
    assert confirmatory == {
        "rft_easy",
        "rft_mixed",
        "rft_curated",
        "grpo_easy",
        "grpo_mixed",
        "grpo_curated",
    }
    # tasks/06b: one secondary arm, registered after unblinding; its contrasts never join CONTRASTS
    assert set(cfg["arms"]) - confirmatory == {"iter_rft_curated"}
    assert not {c.key for c in report.CONTRASTS} & {c.key for c in report.SECONDARY_CONTRASTS}
    assert all({c.a, c.b} & {"iter_rft_curated"} for c in report.SECONDARY_CONTRASTS)
    assert {c.key for c in report.CONTRASTS} >= {
        "H1_rft",
        "H1_grpo",
        "H2_num",
        "H2_den",
        "H3",
        "C1",
        "C2",
    }
    fragments = {n["fragment"] for n in cfg["never_results"]}
    assert {
        "_bf16merge_stale",
        "/rft_noeos_ablation/",
        "/grpo/_failed/",
        "/dev/",
        "google__gemma",
    } <= fragments
