# JPC Test Suite

This directory contains comprehensive tests for the jpc library.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_energies.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

To run with coverage:
```bash
pytest tests/ --cov=jpc --cov-report=term
```

To generate an HTML coverage report:
```bash
pytest tests/ --cov=jpc --cov-report=html
```
Then open `htmlcov/index.html` in your browser.

To see coverage percentage only:
```bash
pytest tests/ --cov=jpc --cov-report=term-missing
```

## Test Structure

- `conftest.py`: Shared fixtures and pytest configuration
- `test_errors.py`: Tests for error checking functions
- `test_init.py`: Tests for initialization functions
- `test_energies.py`: Tests for energy functions (PC, HPC, BPC, PDM)
- `test_grads.py`: Tests for gradient computation functions
- `test_infer.py`: Tests for inference solving functions
- `test_updates.py`: Tests for update functions (activities and parameters)
- `test_analytical.py`: Tests for analytical tools
- `test_utils.py`: Tests for utility functions
- `test_train.py`: Tests for training functions
- `test_test_functions.py`: Tests for test utility functions

## Automated Testing

Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/tests.yml`). The workflow:
- Runs tests on Python 3.10 and 3.11
- Generates coverage reports
- Automatically updates the coverage badge in the main README on pushes to `main`

## Requirements

The tests require:
- pytest
- pytest-cov (for coverage reporting)
- jax
- jax.numpy
- equinox
- diffrax
- optax

All dependencies should be installed when installing jpc. Coverage configuration is defined in `pyproject.toml` under `[tool.coverage.*]`.
