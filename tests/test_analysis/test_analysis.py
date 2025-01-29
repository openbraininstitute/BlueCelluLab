"""Unit tests for the analysis module."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from bluecellulab.analysis.analysis import compute_plot_iv_curve, compute_plot_fi_curve
from pathlib import Path
from bluecellulab.cell import Cell
from bluecellulab.circuit.circuit_access import EmodelProperties


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


@pytest.fixture
def mock_efel():
    efel_mock = MagicMock()
    efel_mock.getFeatureValues.return_value = [{'steady_state_voltage_stimend': [-65]}]
    return efel_mock


@pytest.fixture
def mock_cell():
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    cell = Cell(
        f"{parent_dir}/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
        f"{parent_dir}/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
        template_format="v6",
        emodel_properties=emodel_properties
    )
    return cell


def test_plot_iv_curve(mock_cell, mock_run_stimulus, mock_search_threshold_current, mock_efel):
    """Test the plot_iv_curve function."""
    with patch('bluecellulab.cell.core', mock_cell), \
         patch('bluecellulab.analysis.analysis.run_stimulus', mock_run_stimulus), \
         patch('bluecellulab.tools.search_threshold_current', mock_search_threshold_current), \
         patch('bluecellulab.analysis.analysis.efel', mock_efel), \
         patch('bluecellulab.analysis.analysis.calculate_rheobase') as mock_rheobase:

        injecting_section = "soma[0]"
        injecting_segment = 0.5
        recording_section = "soma[0]"
        recording_segment = 0.5
        stim_start = 100.0
        duration = 500.0
        post_delay = 100.0
        threshold_voltage = -30
        nb_bins = 11

        mock_rheobase.return_value = 0.2

        list_amp, steady_states = compute_plot_iv_curve(
            mock_cell,
            injecting_section=injecting_section,
            injecting_segment=injecting_segment,
            recording_section=recording_section,
            recording_segment=recording_segment,
            stim_start=stim_start,
            duration=duration,
            post_delay=post_delay,
            threshold_voltage=threshold_voltage,
            nb_bins=nb_bins
        )

        assert isinstance(list_amp, np.ndarray)
        assert isinstance(steady_states, np.ndarray)
        assert len(list_amp) == nb_bins
        assert len(steady_states) == nb_bins

        mock_rheobase.assert_called_once_with(
            cell=mock_cell,
            section=injecting_section,
            segx=injecting_segment
        )


def test_plot_fi_curve(mock_cell, mock_search_threshold_current):
    """Test the compute_plot_fi_curve function."""
    with patch('bluecellulab.cell.Cell', mock_cell), \
         patch('bluecellulab.tools.search_threshold_current', mock_search_threshold_current), \
         patch('bluecellulab.analysis.analysis.calculate_rheobase') as mock_rheobase:

        stim_start = 100.0
        duration = 500.0
        post_delay = 100.0
        max_current = 0.8
        nb_bins = 3

        mock_rheobase.return_value = 0.1

        list_amp, spike_count = compute_plot_fi_curve(
            cell=mock_cell,
            injecting_section="apic[0]",
            injecting_segment=0.2,
            recording_section="soma[0]",
            recording_segment=0.5,
            stim_start=stim_start,
            duration=duration,
            post_delay=post_delay,
            max_current=max_current,
            nb_bins=nb_bins
        )

        mock_rheobase.assert_called_once_with(
            cell=mock_cell,
            section="apic[0]",
            segx=0.2
        )

        assert isinstance(list_amp, np.ndarray)
        assert isinstance(spike_count, np.ndarray)
        assert len(list_amp) == nb_bins, f"list_amp length should be {nb_bins}."
        assert len(spike_count) == nb_bins, f"spike_count length should be {nb_bins}."

        # Test with invalid section name
        with pytest.raises(ValueError):
            compute_plot_fi_curve(
                cell=mock_cell,
                injecting_section="invalid_section",
                injecting_segment=0.5,
                recording_section="soma[0]",
                recording_segment=0.5,
                stim_start=stim_start,
                duration=duration,
                post_delay=post_delay,
                max_current=max_current,
                nb_bins=nb_bins
            )

        # Test with invalid segment position
        with pytest.raises(ValueError):
            compute_plot_fi_curve(
                cell=mock_cell,
                injecting_section="soma[0]",
                injecting_segment=1.5,  # Invalid segment
                recording_section="soma[0]",
                recording_segment=0.5,
                stim_start=stim_start,
                duration=duration,
                post_delay=post_delay,
                max_current=max_current,
                nb_bins=nb_bins
            )
