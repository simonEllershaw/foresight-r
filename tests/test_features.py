"""
Example test file.
"""
import pytest
import pandas as pd
import numpy as np
from src.features import scale_features, encode_categorical


def test_scale_features():
    """Test feature scaling."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    result = scale_features(df, ["a", "b"])
    
    # Check that mean is close to 0 and std is close to 1
    assert np.allclose(result["a"].mean(), 0, atol=1e-10)
    assert np.allclose(result["a"].std(), 1, atol=1e-10)


def test_encode_categorical():
    """Test categorical encoding."""
    df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
    result = encode_categorical(df, ["cat"])
    
    # Check that values are numeric
    assert result["cat"].dtype in [np.int32, np.int64]
    # Check that unique values are preserved
    assert len(result["cat"].unique()) == 3
