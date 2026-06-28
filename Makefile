.PHONY: deps workspace-bootstrap workspace-validate lint type test run-sample clean

PYTHON ?= python3
DEPS_DIR ?= .deps
PYTHONPATH := $(DEPS_DIR):src

deps:
	$(PYTHON) -m pip install --target $(DEPS_DIR) -e ".[dev,fast-json]"

workspace-bootstrap:
	bash scripts/bootstrap_workspace.sh

workspace-validate:
	bash scripts/validate_workspace.sh quick

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check src tests

type:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mypy src

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

run-sample:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m crust_lite.cli run-all --config configs/kumamoto.yml --sample

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache data/interim/* data/processed/* outputs/maps/* outputs/tables/* outputs/dashboard/* outputs/reports/* outputs/3d/*
