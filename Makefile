.PHONY: test test-unit test-integration lint typecheck train-phase1 clean

test:
	pytest tests/ -v --ignore=tests/benchmarks

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check olympus/ genesis/ genesis_mlx/ tests/

typecheck:
	mypy olympus/ genesis/

train-phase1:
	python -m genesis.training.train_phase1_bootstrap

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
