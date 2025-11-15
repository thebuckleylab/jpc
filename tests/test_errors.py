"""Tests for error checking functions."""

import pytest
from jpc._core._errors import _check_param_type


def test_check_param_type_valid():
    """Test that valid parameter types pass."""
    _check_param_type("sp")
    _check_param_type("mupc")
    _check_param_type("ntk")


def test_check_param_type_invalid():
    """Test that invalid parameter types raise ValueError."""
    with pytest.raises(ValueError, match="Invalid parameterisation"):
        _check_param_type("invalid")
    
    with pytest.raises(ValueError, match="Invalid parameterisation"):
        _check_param_type("")
    
    with pytest.raises(ValueError, match="Invalid parameterisation"):
        _check_param_type("ntp")  # Note: library uses "ntk" not "ntp"

