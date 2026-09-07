.PHONY: setup-mac setup-gpu test lint fmt gen gen-ood sample-pool eval-base clean

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

# Implemented by tasks/02
eval-base:
	uv run rlordata eval --config configs/eval/base.yaml

clean:
	rm -rf outputs/tmp
