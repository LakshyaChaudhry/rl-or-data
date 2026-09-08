.PHONY: setup-mac setup-gpu test test-all lint fmt gen gen-ood sample-pool cap-run cap tier tier-dry eval-base eval-dry sanity transfer-pick sync clean

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

# tasks/01b: 20 random pool problems with answers and per-step set sizes, for hand-review
sample-pool:
	uv run python scripts/sample_pool.py --pool data/pool/pool.jsonl --n 20 --seed 20260908 \
		--out notebook/samples/pool_v1.2_sample20_seed20260908.md

# ---- tasks/02 (GPU box) — run in this order: cap-run -> cap -> tier -> eval-base -> transfer-pick ----
# Provisional cap run: Qwen3-4B-Base, T=1.0, n=8, cap 4096, val_candidates. Refuses once cap.yaml exists.
cap-run:
	uv run rlordata eval --config configs/eval/provisional_cap.yaml

# Writes configs/locked/cap.yaml from the provisional run (refuses to overwrite). Commit the file afterwards.
cap:
	uv run python scripts/compute_cap.py --run-dir runs/cap_provisional/Qwen__Qwen3-4B-Base/val_candidates/mean_at_k

# pass@8 tiering of the pool (+ post-hoc ood_hard_200) -> data/splits/, data/samples/tiering_pass8.jsonl
tier:
	uv run rlordata tier --config configs/data/tiering.yaml

eval-base:
	uv run rlordata eval --config configs/eval/base.yaml

transfer-pick:
	uv run rlordata eval --config configs/eval/transfer_pick.yaml

sanity:
	uv run python -m rlordata.analysis.sanity --splits-dir data/splits

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
