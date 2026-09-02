.PHONY: setup-mac setup-gpu test lint fmt gen eval-base clean

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

# Implemented by tasks/02
eval-base:
	uv run rlordata eval --config configs/eval/base.yaml

clean:
	rm -rf outputs/tmp
