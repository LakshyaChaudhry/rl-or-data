"""Training diagnostics callback for GRPO (tasks/04 §4). AGENT-OWNED.

Logs every step to ``train_log.jsonl``: TRL metrics (reward mean/std, frac_reward_zero_std,
completion length/truncation, kl, clip, grad_norm, lr, …) plus our per-tier / per-prompt
rolling stats and cumulative completion/token counts.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlordata.train.common import print_cost
from rlordata.train.rewards import RewardRecorder, load_reward_records


class GrpoDiagnosticsCallback:
    """HuggingFace ``TrainerCallback`` that writes one JSONL line per logging step."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        recorder: RewardRecorder,
        num_generations: int,
        est_gpu_hours: float,
        gpu_hours_label: str = "grpo start estimate",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.recorder = recorder
        self.num_generations = int(num_generations)
        self.log_path = self.run_dir / "train_log.jsonl"
        self.est_gpu_hours = float(est_gpu_hours)
        self.gpu_hours_label = gpu_hours_label
        self._t0 = time.monotonic()
        self._last_log_t = self._t0
        self._printed_start = False
        # problem_id -> list of recent correct flags (rolling window)
        self._pass_hist: dict[str, list[int]] = defaultdict(list)
        self.cumulative_completions = 0
        self.cumulative_tokens = 0

    # TrainerCallback API -------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self._printed_start:
            print_cost(self.gpu_hours_label, self.est_gpu_hours)
            self._printed_start = True
        self._t0 = time.monotonic()
        self._last_log_t = self._t0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # HF fires on_step_begin once per optimizer step with global_step == N-1; the rewards for
        # step N are computed inside training_step, before global_step is incremented. Tag them N
        # so on_log (which fires after the increment, with global_step == N) can find them.
        self.recorder.set_step(int(getattr(state, "global_step", 0) or 0) + 1)

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any
    ) -> None:
        logs = dict(logs or {})
        step = int(getattr(state, "global_step", 0) or logs.get("step") or 0)
        now = time.monotonic()
        step_time = now - self._last_log_t
        self._last_log_t = now

        # Pull newly written reward records for this step (best-effort).
        records = [r for r in load_reward_records(self.recorder.path) if r.get("step") == step]
        if records:
            self.cumulative_completions += len(records)
            for r in records:
                n = r.get("n_tokens")
                if isinstance(n, int):
                    self.cumulative_tokens += n
                pid = str(r.get("problem_id", ""))
                self._pass_hist[pid].append(int(bool(r.get("correct"))))
                if len(self._pass_hist[pid]) > 32:
                    self._pass_hist[pid] = self._pass_hist[pid][-32:]

        per_tier = _per_tier_stats(records, self.num_generations)
        active = {
            pid: sum(hist) / max(len(hist), 1)
            for pid, hist in self._pass_hist.items()
            if hist and 0 < sum(hist) < len(hist)  # mixed recent outcomes ≈ "active"
        }

        row = {
            "step": step,
            "step_time_s": round(step_time, 4),
            "wall_clock_s": round(now - self._t0, 3),
            "reward_mean": _get(logs, "reward", "rewards/mean", "train/reward"),
            "reward_std": _get(logs, "reward_std", "rewards/std", "train/reward_std"),
            "frac_reward_zero_std": _get(
                logs, "frac_reward_zero_std", "rewards/frac_zero_std", "train/frac_reward_zero_std"
            ),
            "completions_mean_length": _get(
                logs, "completions/mean_length", "mean_length", "train/completions_mean_length"
            ),
            "completions_clipped_ratio": _get(
                logs, "completions/clipped_ratio", "clipped_ratio", "train/clipped_ratio"
            ),
            "kl": _get(logs, "kl", "train/kl", "objective/kl"),
            "clip_ratio": _get(logs, "clip_ratio", "clip_ratio/mean", "train/clip_ratio"),
            "grad_norm": _get(logs, "grad_norm", "train/grad_norm"),
            "lr": _get(logs, "learning_rate", "lr", "train/learning_rate"),
            "loss": _get(logs, "loss", "train/loss"),
            "num_tokens": _get(logs, "num_tokens", "train/num_tokens"),
            "per_tier": per_tier,
            "n_active_prompts": len(active),
            "cumulative_completions": self.cumulative_completions,
            "cumulative_training_tokens": self.cumulative_tokens,
            "trl_logs": {k: _jsonable(v) for k, v in logs.items()},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        wall_h = (time.monotonic() - self._t0) / 3600.0
        print_cost("grpo train actual", wall_h)


def _get(logs: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in logs:
            return _jsonable(logs[k])
    return None


def _jsonable(v: Any) -> Any:
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return (
                v.detach().float().cpu().item()
                if v.numel() == 1
                else v.detach().float().cpu().tolist()
            )
    except ImportError:
        pass
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def _per_tier_stats(records: list[dict[str, Any]], num_generations: int) -> dict[str, Any]:
    by_tier: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_tier[str(r.get("tier", "untiered"))].append(float(r.get("reward", 0.0)))
    out: dict[str, Any] = {}
    g = max(int(num_generations), 1)
    for tier, rewards in sorted(by_tier.items()):
        mean = sum(rewards) / max(len(rewards), 1)
        # Group into chunks of G in list order (generation order within the step).
        zero_std = 0
        n_groups = 0
        for i in range(0, len(rewards) - g + 1, g):
            chunk = rewards[i : i + g]
            n_groups += 1
            if max(chunk) - min(chunk) < 1e-12:
                zero_std += 1
        out[tier] = {
            "reward_mean": mean,
            "n": len(rewards),
            "frac_reward_zero_std": (zero_std / n_groups) if n_groups else None,
        }
    return out


# Register as a transformers TrainerCallback when transformers is available.
try:
    from transformers import TrainerCallback

    class _HFCallback(GrpoDiagnosticsCallback, TrainerCallback):  # type: ignore[misc, valid-type]
        pass

    DiagnosticsCallback = _HFCallback
except ImportError:  # local Mac without ml extra
    DiagnosticsCallback = GrpoDiagnosticsCallback  # type: ignore[misc, assignment]
