"""Read-only loader for the result-bearing runs (tasks/05). AGENT-OWNED.

The run root is laid out like the artifact store's ``runs/`` directory and is never written to.
Runs are found by an explicit allow-list built from ``configs/analysis/default.yaml`` — never by
globbing — so nothing that merely sits next to a result can enter a table:

  * RFT: ``<arm dir>/chosen.json`` (written by the val-only sweep) names the config; the
    result-bearing runs are that config at the configured seeds, and only ``eval/final`` units load.
  * GRPO and controls: one run dir per seed, ``eval/final`` units only.
  * Base and reference models: ``<model dir>/<split>/<decoding>``.

Every unit's metrics are recomputed from its ``samples.jsonl`` with
``sampling.eval_runner.metrics_for`` (a wrapper of ``core.evaluate.compute_metrics``) using the
unit's own seed, so they can be compared bit-for-bit with the stored ``metrics.json``
(``analysis.sanity``). Per-problem score vectors are kept for the paired bootstraps in
``analysis.stats``. Nothing in this module looks at an accuracy to decide anything.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rlordata.core.evaluate import pass_at_k
from rlordata.sampling.eval_runner import metrics_for, read_samples
from rlordata.train.rft import run_name

DEFAULT_CONFIG = Path("configs/analysis/default.yaml")
LONG_COMPLETION_TOKENS = 2048  # SPEC §7: report the fraction of correct completions above 2048
CACHE_VERSION = 1
PASS_KS = (1, 2, 4, 8, 16, 32, 64)  # SPEC §10

UnitKey = tuple[str, str]  # (split, decoding)


class ExcludedPathError(RuntimeError):
    """A path that SPEC/close-out marks as not-a-result was about to be read as one."""


@dataclass(frozen=True)
class Unit:
    """One eval unit = one (model/adapter, split, decoding) directory."""

    path: Path
    rel: str  # path relative to the run root (posix)
    split: str
    decoding: str  # greedy | mean_at_k | pass_at_k
    seed: int
    config: dict[str, Any]
    meta: dict[str, Any]
    stored_metrics: dict[str, Any]
    sanity: dict[str, Any]
    metrics: dict[str, Any]  # recomputed from samples.jsonl via metrics_for
    config_hash: str
    samples_sha256: str
    problem_ids: tuple[str, ...]  # sorted; every per-problem array below is in this order
    scores: np.ndarray  # [P] mean(correct) over the problem's samples
    truncated: np.ndarray  # [P] fraction of the problem's samples that hit the cap
    n_samples: np.ndarray  # [P] int
    n_correct: np.ndarray  # [P] int
    tiers: tuple[str, ...]  # [P]
    completion_sha: tuple[str, ...]  # [P] sha256 of the completion; greedy units only, else ()
    prompt_digest: str  # sha256 over sorted (problem_id, prompt) — prompt-template drift check
    frac_correct_over_2048: float | None  # None when no completion is correct
    n_truncated_correct: int  # must be 0 under SPEC §5 v1.6
    sample_field_values: dict[str, list[Any]]  # distinct config_hash / seed / arm seen in samples

    def pass_at(self, k: int) -> np.ndarray:
        """[P] unbiased per-problem pass@k (core.evaluate.pass_at_k)."""
        assert self.n_samples.ndim == 1 and self.n_samples.shape == self.n_correct.shape
        return np.asarray(
            [
                pass_at_k(int(n), int(c), k)
                for n, c in zip(self.n_samples, self.n_correct, strict=True)
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Run:
    """One model under evaluation: a trained seed, a control, the base model or a reference."""

    key: str  # arm key ("grpo_curated"), "base", or the reference's directory name
    label: str
    kind: str  # arm | control | base | reference
    method: str | None  # rft | grpo | None
    data_condition: str | None
    seed: int
    run_dir: Path
    rel: str
    units: dict[UnitKey, Unit]
    budgets: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    config_hash: str | None = None
    train_log: list[dict[str, Any]] = field(default_factory=list)
    curves: dict[str, Any] = field(default_factory=dict)
    chosen: dict[str, Any] | None = None


@dataclass(frozen=True)
class Dataset:
    cfg: dict[str, Any]
    run_root: Path
    base: Run
    references: list[Run]
    arms: dict[str, list[Run]]  # arm key -> runs ordered by seed
    controls: dict[str, list[Run]]
    missing: list[dict[str, str]]  # expected units that do not exist
    selection: dict[str, dict[str, Any]]  # RFT arm -> {chosen, sweep, sweep_gpu_hours, ...}

    def trained_runs(self) -> list[Run]:
        return [r for runs in (*self.arms.values(), *self.controls.values()) for r in runs]

    def all_runs(self) -> list[Run]:
        return [self.base, *self.references, *self.trained_runs()]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    run_root: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if run_root is not None:
        cfg["run_root"] = str(run_root)
    if out_dir is not None:
        cfg["out_dir"] = str(out_dir)
    return cfg


def analysis_config_hash(cfg: dict[str, Any]) -> str:
    """Hash of everything that can change a number; paths that only relocate I/O are left out."""
    skip = ("run_root", "out_dir", "splits_dir", "store_checksums")
    payload = {k: v for k, v in cfg.items() if k not in skip}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# exclusion guard
# ---------------------------------------------------------------------------


def _rel(path: Path, run_root: Path) -> str:
    return path.resolve().relative_to(run_root.resolve()).as_posix()


def excluded_reason(path: Path, run_root: Path, cfg: dict[str, Any]) -> str | None:
    probe = f"/{_rel(path, run_root)}/"
    for item in cfg.get("never_results") or []:
        if item["fragment"] in probe:
            return str(item["why"])
    return None


def guard(path: Path, run_root: Path, cfg: dict[str, Any]) -> Path:
    """Raise if ``path`` is one of the things that must never enter a table or contrast."""
    why = excluded_reason(path, run_root, cfg)
    if why is not None:
        raise ExcludedPathError(f"{path} is not a result: {why}")
    return path


def excluded_present(run_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Which never-results exist under the run root (listed in the sanity output, never loaded)."""
    out = []
    for item in cfg.get("never_results") or []:
        frag = item["fragment"].strip("/")
        hits = sorted(
            p.relative_to(run_root).as_posix()
            for p in run_root.glob(f"**/*{frag.split('/')[-1]}*")
            if item["fragment"] in f"/{p.relative_to(run_root).as_posix()}/"
        )
        # keep only the top-most hit of each subtree
        tops: list[str] = []
        for h in hits:
            if not any(h.startswith(t + "/") for t in tops):
                tops.append(h)
        out.append({"fragment": item["fragment"], "why": item["why"], "paths": tops})
    return out


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_unit_payload(unit_dir: str) -> dict[str, Any]:
    """Everything derived from one unit's samples.jsonl, as plain JSON (cacheable, picklable)."""
    d = Path(unit_dir)
    cfg = yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")) or {}
    samples = read_samples(d / "samples.jsonl")
    decoding = cfg.get("decoding") or {}
    metrics = metrics_for(
        samples,
        ks=decoding.get("ks"),
        seed=int(cfg["seed"]),
        cap=int(cfg["max_completion_tokens"]),
        split=cfg.get("split"),
    )
    by_problem: dict[str, list[Any]] = {}
    for s in samples:
        by_problem.setdefault(s.problem_id, []).append(s)
    pids = sorted(by_problem)
    greedy = decoding.get("name") == "greedy"
    correct = [s for s in samples if s.correct]
    prompt_h = hashlib.sha256()
    for pid in pids:
        prompt_h.update(pid.encode("utf-8"))
        prompt_h.update(b"\x00")
        prompt_h.update(by_problem[pid][0].prompt.encode("utf-8"))
        prompt_h.update(b"\x01")
    return {
        "metrics": metrics,
        "problem_ids": pids,
        "scores": [float(np.mean([float(s.correct) for s in by_problem[p]])) for p in pids],
        "truncated": [float(np.mean([float(s.truncated) for s in by_problem[p]])) for p in pids],
        "n_samples": [len(by_problem[p]) for p in pids],
        "n_correct": [sum(int(s.correct) for s in by_problem[p]) for p in pids],
        "tiers": [by_problem[p][0].tier for p in pids],
        "completion_sha": (
            [hashlib.sha256(by_problem[p][0].completion.encode("utf-8")).hexdigest() for p in pids]
            if greedy
            else []
        ),
        "prompt_digest": prompt_h.hexdigest(),
        "frac_correct_over_2048": (
            float(np.mean([s.n_tokens > LONG_COMPLETION_TOKENS for s in correct]))
            if correct
            else None
        ),
        "n_truncated_correct": sum(1 for s in samples if s.truncated and s.correct),
        "sample_field_values": {
            "config_hash": sorted({s.config_hash for s in samples}),
            "seed": sorted({int(s.seed) for s in samples}),
            "arm": sorted({str(s.arm) for s in samples}),
        },
    }


def _cached_payload(
    unit_dir: Path, cache_dir: Path | None
) -> tuple[dict[str, Any] | None, str, str]:
    """(payload or None, cache key, samples sha256).

    The key covers the samples, the unit config and this code's version.
    """
    samples_sha = _sha256_file(unit_dir / "samples.jsonl")
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}\n".encode())
    h.update(samples_sha.encode())
    h.update((unit_dir / "config.yaml").read_bytes())
    key = h.hexdigest()
    if cache_dir is not None:
        f = cache_dir / f"{key}.json"
        if f.exists():
            return _read_json(f), key, samples_sha
    return None, key, samples_sha


def _unit_from_payload(
    unit_dir: Path, run_root: Path, payload: dict[str, Any], samples_sha: str
) -> Unit:
    cfg = yaml.safe_load((unit_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    sanity_path = unit_dir / "sanity.json"
    return Unit(
        path=unit_dir,
        rel=_rel(unit_dir, run_root),
        split=str(cfg["split"]),
        decoding=str((cfg.get("decoding") or {}).get("name")),
        seed=int(cfg["seed"]),
        config=cfg,
        meta=_read_json(unit_dir / "meta.json"),
        stored_metrics=_read_json(unit_dir / "metrics.json"),
        sanity=_read_json(sanity_path) if sanity_path.exists() else {},
        metrics=payload["metrics"],
        config_hash=(unit_dir / "config_hash.txt").read_text(encoding="utf-8").strip(),
        samples_sha256=samples_sha,
        problem_ids=tuple(payload["problem_ids"]),
        scores=np.asarray(payload["scores"], dtype=np.float64),
        truncated=np.asarray(payload["truncated"], dtype=np.float64),
        n_samples=np.asarray(payload["n_samples"], dtype=np.int64),
        n_correct=np.asarray(payload["n_correct"], dtype=np.int64),
        tiers=tuple(payload["tiers"]),
        completion_sha=tuple(payload["completion_sha"]),
        prompt_digest=str(payload["prompt_digest"]),
        frac_correct_over_2048=payload["frac_correct_over_2048"],
        n_truncated_correct=int(payload["n_truncated_correct"]),
        sample_field_values=payload["sample_field_values"],
    )


def load_units(
    unit_dirs: list[Path],
    run_root: Path,
    cfg: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    jobs: int = 1,
) -> dict[Path, Unit]:
    """Load many units; samples are parsed at most once per (samples, config) thanks to the cache."""
    for d in unit_dirs:
        guard(d, run_root, cfg)
    payloads: dict[Path, dict[str, Any]] = {}
    keys: dict[Path, str] = {}
    shas: dict[Path, str] = {}
    todo: list[Path] = []
    for d in unit_dirs:
        payload, keys[d], shas[d] = _cached_payload(d, cache_dir)
        if payload is None:
            todo.append(d)
        else:
            payloads[d] = payload
    if todo:
        if jobs > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                computed = list(pool.map(compute_unit_payload, [str(d) for d in todo]))
        else:
            computed = [compute_unit_payload(str(d)) for d in todo]
        for d, payload in zip(todo, computed, strict=True):
            payloads[d] = payload
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"{keys[d]}.json").write_text(
                    json.dumps(payload, sort_keys=True), encoding="utf-8"
                )
    return {d: _unit_from_payload(d, run_root, payloads[d], shas[d]) for d in unit_dirs}


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def _read_train_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _final_unit_dirs(run_dir: Path, cfg: dict[str, Any]) -> dict[UnitKey, Path]:
    return {
        (split, dec): run_dir / "eval" / "final" / split / dec for split, dec in cfg["final_units"]
    }


def _model_unit_dirs(model_dir: Path, cfg: dict[str, Any]) -> dict[UnitKey, Path]:
    return {(split, dec): model_dir / split / dec for split, dec in cfg["final_units"]}


def _exists(unit_dir: Path) -> bool:
    return (unit_dir / "samples.jsonl").exists() and (unit_dir / "metrics.json").exists()


def rft_run_dirs(arm_dir: Path, seeds: list[int]) -> tuple[dict[str, Any], dict[int, Path]]:
    """(chosen.json, {seed: run dir of the val-chosen config})."""
    chosen = _read_json(arm_dir / "chosen.json")
    lr, ep = float(chosen["learning_rate"]), int(chosen["epochs"])
    return chosen, {s: arm_dir / run_name(s, lr, ep) for s in seeds}


def _selection_record(
    arm_dir: Path, chosen: dict[str, Any], run_root: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    """The val-only selection record of one RFT arm: sweep.json plus what the sweep cost.

    Only ``meta.json`` of the other sweep runs is read (GPU-hours); none of their evals load.
    """
    sweep = _read_json(arm_dir / "sweep.json") if (arm_dir / "sweep.json").exists() else {}
    chosen_name = Path(str(chosen.get("sweep_run_dir", ""))).name
    other_train = 0.0
    other_eval = 0.0
    n_other = 0
    for d in sorted(arm_dir.glob("seed1_lr*_ep*")):
        if d.name == chosen_name or excluded_reason(d, run_root, cfg):
            continue
        n_other += 1
        meta = d / "meta.json"
        if meta.exists():
            other_train += float(_read_json(meta).get("gpu_hours_actual") or 0.0)
        for m in sorted((d / "eval" / "val").glob("*/*/meta.json")):
            other_eval += float(_read_json(m).get("gpu_hours_actual") or 0.0)
    return {
        "chosen": chosen,
        "sweep": sweep,
        "n_configs_tried": n_other + 1,
        "other_configs_train_gpu_hours": other_train,
        "other_configs_val_eval_gpu_hours": other_eval,
    }


def load_dataset(cfg: dict[str, Any], *, cache_dir: Path | None = None, jobs: int = 1) -> Dataset:
    run_root = Path(cfg["run_root"])
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root {run_root} does not exist")
    seeds = [int(s) for s in cfg["seeds"]]

    plan: list[dict[str, Any]] = []  # one entry per Run
    base_dir = run_root / cfg["base"]["dir"]
    base_units = _model_unit_dirs(base_dir, cfg)
    transfer = cfg["base"].get("transfer_dir")
    if transfer:
        for key in list(base_units):
            alt = run_root / transfer / key[0] / key[1]
            if not _exists(base_units[key]) and _exists(alt):
                base_units[key] = alt
    plan.append(
        {
            "key": "base",
            "label": cfg["base"]["label"],
            "kind": "base",
            "method": None,
            "data_condition": None,
            "seed": 1,
            "run_dir": base_dir,
            "unit_dirs": base_units,
        }
    )
    for ref in cfg.get("references") or []:
        d = run_root / ref["dir"]
        units = _model_unit_dirs(d, cfg)
        # Reference models were evaluated on the counting splits only (SPEC §10).
        units = {k: v for k, v in units.items() if k[0] != "gsm8k_500"}
        plan.append(
            {
                "key": Path(ref["dir"]).name,
                "label": ref["label"],
                "kind": "reference",
                "method": None,
                "data_condition": None,
                "seed": 1,
                "run_dir": d,
                "unit_dirs": units,
            }
        )
    selection: dict[str, dict[str, Any]] = {}
    for kind, group in (("arm", cfg["arms"]), ("control", cfg.get("controls") or {})):
        for key, spec in group.items():
            arm_seeds = [int(s) for s in spec.get("seeds", seeds)]
            chosen = None
            if spec["method"] == "rft":
                arm_dir = guard(run_root / spec["dir"], run_root, cfg)
                chosen, dirs = rft_run_dirs(arm_dir, arm_seeds)
                selection[key] = _selection_record(arm_dir, chosen, run_root, cfg)
            else:
                dirs = {s: run_root / spec["run_pattern"].format(seed=s) for s in arm_seeds}
            for s in arm_seeds:
                plan.append(
                    {
                        "key": key,
                        "label": spec["label"],
                        "kind": kind,
                        "method": spec["method"],
                        "data_condition": spec["data_condition"],
                        "seed": s,
                        "run_dir": guard(dirs[s], run_root, cfg),
                        "unit_dirs": _final_unit_dirs(dirs[s], cfg),
                        "chosen": chosen,
                    }
                )

    missing: list[dict[str, str]] = []
    wanted: list[Path] = []
    for p in plan:
        for (split, dec), d in p["unit_dirs"].items():
            if _exists(d):
                wanted.append(d)
            else:
                missing.append(
                    {
                        "who": p["key"] if p["kind"] != "arm" else f"{p['key']}/seed{p['seed']}",
                        "split": split,
                        "decoding": dec,
                        "path": str(d),
                    }
                )
    loaded = load_units(wanted, run_root, cfg, cache_dir=cache_dir, jobs=jobs)

    runs: list[Run] = []
    for p in plan:
        rd: Path = p["run_dir"]
        trained = p["kind"] in ("arm", "control")
        runs.append(
            Run(
                key=p["key"],
                label=p["label"],
                kind=p["kind"],
                method=p["method"],
                data_condition=p["data_condition"],
                seed=p["seed"],
                run_dir=rd,
                rel=_rel(rd, run_root) if rd.exists() else str(rd),
                units={k: loaded[d] for k, d in p["unit_dirs"].items() if d in loaded},
                budgets=_read_json(rd / "budgets.json")
                if trained and (rd / "budgets.json").exists()
                else None,
                meta=_read_json(rd / "meta.json")
                if trained and (rd / "meta.json").exists()
                else None,
                config_hash=(rd / "config_hash.txt").read_text(encoding="utf-8").strip()
                if trained and (rd / "config_hash.txt").exists()
                else None,
                train_log=_read_train_log(rd / "train_log.jsonl") if trained else [],
                curves=_read_json(rd / "curves.json")
                if trained and (rd / "curves.json").exists()
                else {},
                chosen=p.get("chosen"),
            )
        )
    arms = {k: [r for r in runs if r.kind == "arm" and r.key == k] for k in cfg["arms"]}
    controls = {
        k: [r for r in runs if r.kind == "control" and r.key == k]
        for k in (cfg.get("controls") or {})
    }
    return Dataset(
        cfg=cfg,
        run_root=run_root,
        base=runs[0],
        references=[r for r in runs if r.kind == "reference"],
        arms=arms,
        controls=controls,
        missing=missing,
        selection=selection,
    )
