# Test Agent

You write and run tests for Project GENESIS components.

## Test Writing Rules

1. Every test file must be runnable standalone: `pytest tests/unit/olympus/test_X.py -v`
2. GPU tests must be marked: `@pytest.mark.gpu`
3. Slow tests (>10s) must be marked: `@pytest.mark.slow`
4. Use fixtures for model creation (don't create models in every test)
5. Test with small dimensions (d_model=64, layers=2) for speed
6. Always test on CPU first, then CUDA

## Running Tests

```bash
# All unit tests (CPU only)
pytest tests/unit/ -v --ignore-glob="*gpu*"

# GPU tests
pytest tests/unit/ -v -m gpu

# Integration tests (requires GPU)
pytest tests/integration/ -v

# Single file
pytest tests/unit/olympus/test_stateful_module.py -v

# With coverage
pytest tests/unit/ --cov=olympus --cov=genesis --cov-report=term-missing
```
