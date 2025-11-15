"""Pytest configuration and shared fixtures for jpc tests."""

import pytest
import jax
import jax.numpy as jnp
import equinox.nn as nn
from jpc import make_mlp


@pytest.fixture
def key():
    """Random key for testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 4


@pytest.fixture
def input_dim():
    """Input dimension for testing."""
    return 10


@pytest.fixture
def hidden_dim():
    """Hidden dimension for testing."""
    return 20


@pytest.fixture
def output_dim():
    """Output dimension for testing."""
    return 5


@pytest.fixture
def depth():
    """Network depth for testing."""
    return 3


@pytest.fixture
def simple_model(key, input_dim, hidden_dim, output_dim, depth):
    """Create a simple MLP model for testing."""
    return make_mlp(
        key=key,
        input_dim=input_dim,
        width=hidden_dim,
        depth=depth,
        output_dim=output_dim,
        act_fn="relu",
        use_bias=False,
        param_type="sp"
    )


@pytest.fixture
def x(key, batch_size, input_dim):
    """Sample input data."""
    return jax.random.normal(key, (batch_size, input_dim))


@pytest.fixture
def y(key, batch_size, output_dim):
    """Sample output/target data."""
    return jax.random.normal(key, (batch_size, output_dim))


@pytest.fixture
def y_onehot(key, batch_size, output_dim):
    """Sample one-hot encoded target data."""
    indices = jax.random.randint(key, (batch_size,), 0, output_dim)
    return jax.nn.one_hot(indices, output_dim)


@pytest.fixture
def layer_sizes(input_dim, hidden_dim, output_dim, depth):
    """Layer sizes for testing."""
    sizes = [input_dim]
    for _ in range(depth - 1):
        sizes.append(hidden_dim)
    sizes.append(output_dim)
    return sizes


def pytest_ignore_collect(collection_path, config):
    """Skip jpc/_test.py which contains library functions, not test functions."""
    # Skip _test.py files in the jpc directory (library functions, not test functions)
    if collection_path.name == "_test.py":
        file_str = str(collection_path.resolve())
        # Check if it's in jpc directory but not in tests
        if "/jpc/" in file_str and "/tests/" not in file_str:
            return True  # Ignore this file
    return None  # Use default behavior for other files


def pytest_collection_modifyitems(config, items):
    """Remove any test items that were incorrectly collected from jpc._test module."""
    filtered_items = []
    for item in items:
        # Check file path
        item_path = None
        if hasattr(item, 'fspath') and item.fspath:
            item_path = str(item.fspath)
        elif hasattr(item, 'path') and item.path:
            item_path = str(item.path)
        elif hasattr(item, 'nodeid'):
            # Extract path from nodeid (format: path/to/file.py::test_function)
            nodeid = item.nodeid
            if '::' in nodeid:
                item_path = nodeid.split('::')[0]
        
        # Check if it's from jpc/_test.py
        if item_path and ('jpc/_test.py' in item_path or 'jpc\\_test.py' in item_path):
            continue
        
        # Check module name
        if hasattr(item, 'module') and item.module:
            module_name = getattr(item.module, '__name__', None)
            if module_name == 'jpc._test':
                continue
        
        # Check function location
        if hasattr(item, 'function') and hasattr(item.function, '__module__'):
            if item.function.__module__ == 'jpc._test':
                continue
        
        filtered_items.append(item)
    
    items[:] = filtered_items
