"""Utility functions for analysis."""

import numpy as np


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c
