"""Unit tests for the analysis module."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from bluecellulab.analysis.analysis import compute_iv_curve
from pathlib import Path
import bluecellulab


parent_dir = Path(__file__).resolve().parent.parent


class MockRecording:
    def __init__(self):
        self.time = [1, 2, 3]
        self.voltage = [-70, -55, -40]


@pytest.fixture
def mock_run_stimulus():
    return MagicMock(return_value=MockRecording())


@pytest.fixture
def mock_search_threshold_current():
    return MagicMock(return_value=0.1)


@pytest.fixture
def mock_steady_state_voltage_stimend():
    return MagicMock(return_value=-65)


def test_compute_iv_curve(mock_run_stimulus, mock_search_threshold_current, mock_steady_state_voltage_stimend):
    """Test the compute_iv_curve function."""
    cell = bluecellulab.Cell(
        "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
        "%s/examples/cell_example1" % str(parent_dir))

    with patch('bluecellulab.analysis.analysis.run_stimulus', mock_run_stimulus), \
         patch('bluecellulab.analysis.analysis.search_threshold_current', mock_search_threshold_current), \
         patch('bluecellulab.analysis.analysis.steady_state_voltage_stimend', mock_steady_state_voltage_stimend):

        stim_start = 100.0
        duration = 500.0
        stim_delay = 100.0
        nb_bins = 11

        list_amp, steady_states = compute_iv_curve(cell, stim_start, duration, stim_delay, nb_bins)

        assert isinstance(list_amp, np.ndarray)
        assert isinstance(steady_states, np.ndarray)
        assert len(list_amp) == nb_bins
        assert len(steady_states) == nb_bins
