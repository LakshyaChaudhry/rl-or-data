"""Ship the PEFT adapter to colocated vLLM as a *native* LoRA (tasks/04; notebook 2026-09-14). AGENT-OWNED.

Why this exists. TRL 1.13's colocate sync (``trl/generation/vllm_generation.py``, ``sync_weights``)
does ``merge_adapter() -> load_weights() -> unmerge_adapter()``. The merge is
``base.weight.data += scaling * B @ A`` on a **bf16** tensor: per-entry |ΔW| is ~1e-6..1e-5 while
the bf16 half-ulp of a typical 0.014 weight is 3e-5, so 87–97 % of ΔW entries round to exactly
zero and only 10–60 % of ‖ΔW‖² reaches vLLM (grpo_mixed_s1 checkpoints 25–150). vLLM then samples
from ≈ base + a coarsely quantised adapter while the trainer differentiates base + the full adapter;
with ratio ≡ 1 and β = 0 nothing corrects it, and the run collapsed at step ~150. Merging in fp32
first does not help: the loss is at the bf16 cast, and vLLM's weights are bf16.

Fix. Keep vLLM's base weights untouched and pass the adapter as a ``LoRARequest``. vLLM applies the
low-rank branch separately (fp32 accumulation), like PEFT's forward, so no rounding against W
happens. Verified on the collapsed run's adapters: vLLM-native-LoRA log-probs match fp32
base+adapter within 0.003–0.08 nats/token at checkpoints 125/150
(``runs/grpo/_diag/nativelora_grpo_mixed_s1``), the same as the trainer's own bf16 path; the merged
path was off by 0.8–5 nats.

Three pieces, all against TRL's public trainer surface (colocate mode only):
  * :func:`vllm_llm_with_lora` — build the colocated ``LLM`` with ``enable_lora`` (it cannot be
    switched on after construction);
  * :class:`NativeLoraSync` — replace ``VLLMGeneration.sync_weights`` (save the adapter, hand vLLM
    a fresh ``LoRARequest`` id) and wrap ``llm.generate`` (pass the current request);
  * :func:`mismatch_logging_trainer_cls` — log |logp_trainer − logp_vLLM| every step, the metric
    that would have caught the merge on day one (TRL only computes it when its importance-sampling
    correction is on, which SPEC v1.8 keeps off).
"""

from __future__ import annotations

import contextlib
import functools
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rlordata.sampling.vllm_sampler import LORA_MAX_RANK

# Adapter versions kept on disk; vLLM's CPU LoRA cache is sized the same, older ones are deleted.
KEEP_VERSIONS = 2
POLICY_SYNC_NATIVE_LORA = "vllm_native_lora"


@contextlib.contextmanager
def vllm_llm_with_lora(
    *, max_lora_rank: int = LORA_MAX_RANK, keep: int = KEEP_VERSIONS
) -> Iterator[None]:
    """While active, TRL's colocated ``LLM(...)`` is constructed with native LoRA support.

    TRL binds ``LLM`` at import time (``from vllm import LLM`` in ``trl.generation.vllm_generation``),
    so the module attribute is what ``VLLMGeneration.__init__`` calls. Call-site kwargs win over the
    partial's, so nothing TRL passes is overridden.
    """
    from trl.generation import vllm_generation as vg

    orig = vg.LLM
    vg.LLM = functools.partial(  # type: ignore[assignment]
        orig,
        enable_lora=True,
        max_lora_rank=int(max_lora_rank),
        max_loras=1,
        max_cpu_loras=max(int(keep), 1),
    )
    try:
        yield
    finally:
        vg.LLM = orig


class NativeLoraSync:
    """Replace ``VLLMGeneration.sync_weights`` with save-adapter + new ``LoRARequest``.

    ``sync_weights`` is what TRL calls before generating whenever ``global_step`` changed
    (``GRPOTrainer._generate_single_turn``). Each sync writes the adapter to ``lora_dir/v<N>`` and
    hands vLLM ``LoRARequest(name="policy-v<N>", id=N, path)``: ids must be fresh, vLLM keys its LoRA
    cache by ``lora_int_id``. ``llm.generate`` is wrapped to pass the current request. The base
    weights inside vLLM are never touched.
    """

    def __init__(self, trainer: Any, lora_dir: str | Path, *, keep: int = KEEP_VERSIONS) -> None:
        vg = getattr(trainer, "vllm_generation", None)
        if vg is None:
            raise RuntimeError("trainer has no vllm_generation (use_vllm must be True)")
        if getattr(vg, "mode", None) != "colocate":
            raise RuntimeError(
                f"native LoRA sync supports vllm_mode='colocate', got {getattr(vg, 'mode', None)!r}"
            )
        if getattr(vg, "enable_sleep_mode", False):
            raise RuntimeError("native LoRA sync does not handle vllm_enable_sleep_mode=True")
        if not lora_enabled(vg.llm):
            raise RuntimeError(
                "vLLM was constructed without enable_lora; build the trainer inside vllm_llm_with_lora()"
            )
        self.trainer = trainer
        self.vg = vg
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.keep = max(int(keep), 1)
        self.version = 0
        self.request: Any = None
        self._orig_generate = vg.llm.generate
        vg.sync_weights = self.sync_weights  # type: ignore[method-assign]
        vg.llm.generate = self._generate  # type: ignore[method-assign]

    def sync_weights(self) -> None:
        """Save the current adapter and point vLLM at it (replaces TRL's merge → load → unmerge)."""
        from vllm.lora.request import LoRARequest

        self.version += 1
        path = self.lora_dir / f"v{self.version}"
        # PeftModel.save_pretrained: adapter_config.json + adapter_model.safetensors (LoRA A/B only).
        self.trainer.model.save_pretrained(str(path))
        self.request = LoRARequest(f"policy-v{self.version}", self.version, str(path))
        self.vg.llm.reset_prefix_cache()
        stale = self.lora_dir / f"v{self.version - self.keep}"
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.request is None:
            raise RuntimeError("vLLM generate called before the first adapter sync")
        kwargs.setdefault("lora_request", self.request)
        return self._orig_generate(*args, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            "vllm_policy_sync": POLICY_SYNC_NATIVE_LORA,
            "max_lora_rank": LORA_MAX_RANK,
            "lora_dir": str(self.lora_dir),
            "syncs": self.version,
        }

    def close(self) -> None:
        shutil.rmtree(self.lora_dir, ignore_errors=True)


def lora_enabled(llm: Any) -> bool:
    """True iff the vLLM engine was built with ``enable_lora`` (its ``lora_config`` is set)."""
    engine = getattr(llm, "llm_engine", None)
    cfg = getattr(engine, "vllm_config", None)
    return getattr(cfg, "lora_config", None) is not None


def mismatch_logging_trainer_cls() -> type:
    """``GRPOTrainer`` subclass logging mean/max |logp_trainer − logp_vLLM| over the sampled tokens.

    Reads the per-token log-probs the loss already computed (no extra forward) against the
    log-probs vLLM reported for the tokens it sampled. The loss is unchanged. A healthy colocated
    policy sits at ≈ 0.01–0.05 nats; the bf16-merge run was at 0.02 by step 100 and 5.2 at step 150.
    Logged as ``sampling/logp_absdiff_{mean,max}`` (averaged over the step by TRL's ``log``).
    """
    import torch
    from trl import GRPOTrainer

    class MismatchLoggingGRPOTrainer(GRPOTrainer):
        _last_per_token_logps: Any = None

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            out = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            try:
                self._last_per_token_logps = out[0].detach()
            except Exception:  # noqa: BLE001 — diagnostics never break training
                self._last_per_token_logps = None
            return out

        def _compute_loss(self, model: Any, inputs: dict[str, Any]) -> Any:
            self._last_per_token_logps = None
            loss = super()._compute_loss(model, inputs)
            samp = inputs.get("sampling_per_token_logps")
            lp = self._last_per_token_logps
            if samp is not None and lp is not None and tuple(samp.shape) == tuple(lp.shape):
                mask = inputs["completion_mask"].bool()
                if bool(mask.any()):
                    diff = torch.nan_to_num((lp.float() - samp.float()).abs(), nan=0.0)[mask]
                    mode = "train" if self.model.training else "eval"
                    self._metrics[mode]["sampling/logp_absdiff_mean"].append(float(diff.mean()))
                    self._metrics[mode]["sampling/logp_absdiff_max"].append(float(diff.max()))
            return loss

    return MismatchLoggingGRPOTrainer
