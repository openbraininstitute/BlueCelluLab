"""Unit tests for analysis utils."""

import numpy as np
from bluecellulab.analysis.utils import exp_decay


def test_exp_decay_scalar():
    assert np.isclose(exp_decay(0, 2, 1, 3), 5.0)
    assert np.isclose(exp_decay(1, 1, 0, 0), 1.0)
    assert np.isclose(exp_decay(1, 1, 1, 0), 1 * np.exp(-1))

def test_exp_decay_vector():
    x = np.array([0, 1, 2])
    result = exp_decay(x, 2, 1, 3)
    expected = 2 * np.exp(-x) + 3
    assert np.allclose(result, expected)

def test_exp_decay_negative_and_large():
    assert np.isclose(exp_decay(-1, 1, 1, 0), np.exp(1))
    assert np.isclose(exp_decay(100, 1, 1, 2), np.exp(-100) + 2)
