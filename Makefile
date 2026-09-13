.PHONY: setup-mac setup-gpu test test-all lint fmt gen gen-ood gen-check sample-pool cap-run cap tier tier-provisional tier-rescore rescore-cap-run tier-dry eval-base eval-dry sanity transfer-pick sync clean rft-draw rft-select rft-sweep rft-finals rft-eval-final rft-dry grpo-train grpo-eval

setup-mac:
	bash setup/setup_mac.sh

setup-gpu:
	bash setup/setup_gpu.sh

test:
	uv run pytest -m "not gpu and not slow"

test-all:
	uv run pytest

lint:
	uv run ruff check src tests scripts && uv run ruff format --check src tests scripts

fmt:
	uv run ruff format src tests scripts && uv run ruff check --fix src tests scripts

# Implemented by tasks/01 (generator + tiering needs a GPU for the pass@8 step; the pool itself is CPU-only)
gen:
	uv run rlordata gen --config configs/data/pool.yaml

# tasks/01b: the complexity-extrapolation set (SPEC §6), generated separately with its own seed
gen-ood:
	uv run rlordata gen --config configs/data/ood.yaml

# The pool is committed. Regenerate from configs into a scratch dir and diff against data/pool/ (box-side
# determinism check: numpy Generator streams are not guaranteed identical across numpy versions).
gen-check:
	uv run python scripts/check_pool.py

# tasks/01b: 20 random pool problems with answers and per-step set sizes, for hand-review
sample-pool:
	uv run python scripts/sample_pool.py --pool data/pool/pool.jsonl --n 20 --seed 20260908 \
		--out notebook/samples/pool_v1.2_sample20_seed20260908.md

# ---- tasks/02 (GPU box) — run in this order: cap-run -> cap -> tier -> eval-base -> transfer-pick ----
# Provisional cap run: Qwen3-4B-Base, T=1.0, n=8, cap 4096, val_candidates. Refuses once cap.yaml exists.
cap-run:
	uv run rlordata eval --config configs/eval/provisional_cap.yaml

# Writes configs/locked/cap.yaml (SPEC §7 v1.3: max(2048, per-cell 1.25×p99)) from the provisional run; refuses to overwrite. Commit it afterwards.
cap:
	uv run python scripts/compute_cap.py --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k \
		--pool data/pool/pool.jsonl

# pass@8 tiering of the pool (+ post-hoc ood_hard_200) -> data/splits/, data/samples/tiering_pass8.jsonl
tier:
	uv run rlordata tier --config configs/data/tiering.yaml

# Sample the tiering completions before the cap is locked (batch-invariant, seeded: the first `cap`
# tokens do not depend on max_tokens), then rebuild pass8 + splits offline once cap.yaml exists.
tier-provisional:
	uv run rlordata tier --config configs/data/tiering.yaml --provisional-cap 4096

tier-rescore:
	uv run rlordata tier --config configs/data/tiering.yaml --rescore-from data/samples/tiering_pass8.jsonl

# Re-verify the provisional cap run with the current verifier (after an extractor change), before `make cap`.
rescore-cap-run:
	uv run python scripts/rescore_samples.py --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k

eval-base:
	uv run rlordata eval --config configs/eval/base.yaml

transfer-pick:
	uv run rlordata eval --config configs/eval/transfer_pick.yaml

sanity:
	uv run python -m rlordata.analysis.sanity --splits-dir data/splits

# ---- tasks/03 (GPU box) — run in this order: rft-draw -> rft-sweep ARM=x (x3) -> rft-finals ARM=x (x3) ----
ARM ?= mixed
# The 192-sample draw for train_easy_100 and train_mixed_100 (once; refuses to overwrite).
rft-draw:
	uv run rlordata rft --config configs/rft/mixed.yaml --stage draw

rft-select:
	uv run rlordata rft --config configs/rft/$(ARM).yaml --stage select

# 9 short runs on seed 1, selection on val_mixed_100 only -> runs/rft/<arm>/{sweep,chosen}.json
rft-sweep:
	uv run python scripts/rft_sweep.py --config configs/rft/$(ARM).yaml

# Seeds 2 and 3 with the chosen config, then the final eval of all three seeds (test/ood once).
rft-finals:
	uv run python scripts/rft_finals.py --config configs/rft/$(ARM).yaml

# Local dry run of the whole tasks/03 pipeline with the stub sampler and a tiny model (never a result).
rft-dry:
	uv run python scripts/rft_dry_run.py

# ---- tasks/04 (GPU box, SEPARATE from the RFT sweep) ----
# Example: make grpo-train CONFIG=configs/grpo/mixed100.yaml SEED=1
CONFIG ?= configs/grpo/mixed100.yaml
SEED ?= 1
grpo-train:
	uv run rlordata grpo --config $(CONFIG) --seed $(SEED) --stage train

grpo-eval:
	uv run rlordata grpo --config $(CONFIG) --seed $(SEED) --stage eval --run-dir $(RUN_DIR)

sync:
	uv run python -m rlordata.artifacts sync-all

# ---- local dry runs with the stub sampler (no GPU, no weights; never a result) ----
tier-dry:
	uv run rlordata tier --config configs/data/tiering.yaml --stub \
		--output-dir data/splits_dry --samples-output data/samples/tiering_pass8_dry.jsonl --run-dir runs/tier_dry

# Acceptance test for the eval pipeline layout: every run-directory file, ~20 problems per split.
eval-dry: tier-dry
	uv run rlordata eval --config configs/eval/base.yaml --stub --force \
		--splits-dir data/splits_dry --output-dir runs/eval_dry --n-problems 20

clean:
	rm -rf outputs/tmp runs/eval_dry runs/tier_dry data/splits_dry data/samples/tiering_pass8_dry.jsonl
