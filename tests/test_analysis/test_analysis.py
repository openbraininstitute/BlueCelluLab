"""Unit tests for the analysis module."""

import json
from unittest.mock import MagicMock, patch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pytest

from bluecellulab.analysis.analysis import BPAP, compute_plot_iv_curve, compute_plot_fi_curve
from bluecellulab.cell import Cell
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.exceptions import BluecellulabError


parent_dir = Path(__file__).resolve().parent.parent


class MockRecording:
    def __init__(self):
        self.time = [1, 2, 3]
        self.voltage = [-70, -55, -40]


def mock_run_stimulus(*args, **kwargs):
    return MockRecording()


@pytest.fixture
def mock_bpap_amplitude_distance():
    """Fixture to mock the bpap_amplitude_distance.json file."""
    amp_dist_json = parent_dir / "data" / "analysis" / "bpap_amplitude_distance.json"
    with open(amp_dist_json, 'r') as f:
        d = json.load(f)
    return d["soma_amp"], d["dend_amps"], d["dend_dist"], d["apic_amps"], d["apic_dist"]


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
        holding_current=-0.02,
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

@pytest.fixture
def run_bpap(mock_cell):
    """Fixture to create a BPAP instance with a mock cell."""
    bpap = BPAP(mock_cell)
    bpap.run(duration=1200, amplitude=0.2)
    return bpap


def test_plot_iv_curve(
    monkeypatch,
    mock_cell,
    mock_search_threshold_current,
    mock_efel
):
    """Test the plot_iv_curve function."""
    # for run_stimulus called in multiprocessing, use monkeypatch instead of MagicMock
    # because MagicMock is not pickleable and cannot be used in multiprocessing
    monkeypatch.setattr("bluecellulab.analysis.analysis.run_stimulus", mock_run_stimulus)
    with patch('bluecellulab.cell.core', mock_cell), \
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
            nb_bins=nb_bins,
            show_figure=False,
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
            nb_bins=nb_bins,
            show_figure=False,
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


# BPAP tests
dummy_recordings = {
    "soma[0]": [-65, -65, -65, -65, -20, 0, -20, -70, -65, -65, -65],
    "dend[0]": [-65, -65, -65, -65, -65, -20, -10, -20, -70, -65, -65],
    "dend[1]": [-65, -65, -65, -65, -65, -65, -20, -15, -20, -70, -65],
    "apic[0]": [-65, -65, -65, -65, -65, -20, -10, -20, -70, -65, -65],
    "apic[1]": [-65, -65, -65, -65, -65, -65, -20, -15, -20, -70, -65],
}


class DummyCell:
    """A dummy cell class to simulate the behavior of a Cell object for testing purposes."""
    def get_allsections_voltagerecordings(self):
        return dummy_recordings
    def get_time(self):
        return [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]


def test_bpap_init_and_properties():
    """Test the initialization and properties of the BPAP class."""
    cell = DummyCell()
    bpap = BPAP(cell)
    assert bpap.cell is cell
    assert bpap.dt == 0.025
    assert bpap.stim_start == 1000
    assert bpap.stim_duration == 5
    # Properties
    assert bpap.start_index == int(1000 / 0.025)
    assert bpap.end_index == int((1000 + 5) / 0.025)


def test_bpap_get_recordings():
    """Test the get_recordings method of the BPAP class."""
    cell = DummyCell()
    bpap = BPAP(cell)
    soma, dend, apic = bpap.get_recordings()
    assert soma == [-65, -65, -65, -65, -20, 0, -20, -70, -65, -65, -65]
    assert "dend[0]" in dend and dend["dend[0]"] == [-65, -65, -65, -65, -65, -20, -10, -20, -70, -65, -65]
    assert "apic[0]" in apic and apic["apic[0]"] == [-65, -65, -65, -65, -65, -20, -10, -20, -70, -65, -65]
    assert "dend[1]" in dend and dend["dend[1]"] == [-65, -65, -65, -65, -65, -65, -20, -15, -20, -70, -65]
    assert "apic[1]" in apic and apic["apic[1]"] == [-65, -65, -65, -65, -65, -65, -20, -15, -20, -70, -65]


def test_bpap_run(mock_cell):
    """Test the run method of the BPAP class."""
    bpap = BPAP(mock_cell)
    bpap.stim_start = 1  # Set start time for the test

    # check that we have no recordings before running the simulation
    pytest.raises(BluecellulabError, bpap.get_recordings)

    # Run the BPAP simulation
    bpap.run(duration=5, amplitude=0.02)

    # Check that we have recordings after running the simulation
    soma_rec, dend_rec, apic_rec = bpap.get_recordings()
    assert soma_rec is not None
    assert isinstance(dend_rec, dict)
    # our mock cell has apical dendrites so we can check for this
    assert isinstance(apic_rec, dict)


def test_amplitudes(run_bpap):
    """Test the amplitudes method of the BPAP class."""
    soma_rec, dend_rec, apic_rec = run_bpap.get_recordings()
    soma_amp = run_bpap.amplitudes({"soma": soma_rec})
    assert isinstance(soma_amp, list)
    assert len(soma_amp) == 1
    assert soma_amp[0] == 111.34521528672875
    dend_amp = run_bpap.amplitudes(dend_rec)
    assert isinstance(dend_amp, list)
    assert len(dend_amp) == 24
    assert dend_amp[0] == 98.23094942121509
    assert dend_amp[23] == 8.915576947165093
    apic_amp = run_bpap.amplitudes(apic_rec)
    assert isinstance(apic_amp, list)
    assert len(apic_amp) == 77
    assert apic_amp[0] == 110.80678737727112
    assert apic_amp[76] == 19.887804173247375


def test_distances_to_soma(mock_cell):
    """Test the distances_to_soma method of the BPAP class."""
    bpap = BPAP(mock_cell)
    dend_recs = {
        "dend[0]": dummy_recordings["dend[0]"],
        "dend[1]": dummy_recordings["dend[1]"],
    }
    dend_distances = bpap.distances_to_soma(dend_recs)
    assert isinstance(dend_distances, list)
    assert len(dend_distances) == 2
    assert dend_distances[0] == 4.132154495613162
    assert dend_distances[1] == 22.94928443789968
    apic_recs = {
        "apic[0]": dummy_recordings["apic[0]"],
        "apic[1]": dummy_recordings["apic[1]"],
    }
    apic_distances = bpap.distances_to_soma(apic_recs)
    assert isinstance(apic_distances, list)
    assert len(apic_distances) == 2
    assert apic_distances[0] == 4.765188446744128
    assert apic_distances[1] == 14.31071445662771


def test_get_amplitudes_and_distances(run_bpap):
    """Test the get_amplitudes_and_distances method of the BPAP class."""
    soma_amp, dend_amps, dend_dist, apic_amps, apic_dist = run_bpap.get_amplitudes_and_distances()
    assert isinstance(soma_amp, list)
    assert len(soma_amp) == 1
    assert soma_amp[0] == 111.34521528672875
    assert isinstance(dend_amps, list)
    assert len(dend_amps) == 24
    assert dend_amps[0] == 98.23094942121509
    assert isinstance(dend_dist, list)
    assert len(dend_dist) == 24
    assert dend_dist[0] == 4.132154495613162
    assert isinstance(apic_amps, list)
    assert len(apic_amps) == 77
    assert apic_amps[0] == 110.80678737727112
    assert isinstance(apic_dist, list)
    assert len(apic_dist) == 77
    assert apic_dist[0] == 4.765188446744128


def test_fit(mock_bpap_amplitude_distance):
    """Test the fit method of the BPAP class."""
    popt_dend, popt_apic = BPAP.fit(*mock_bpap_amplitude_distance)
    soma_amp_value = mock_bpap_amplitude_distance[0][0]
    assert popt_dend[0] > soma_amp_value / 2.
    assert popt_dend[0] < soma_amp_value * 2.
    assert popt_dend[1] > 0
    assert popt_dend[2] < soma_amp_value / 10.
    assert popt_apic[0] > soma_amp_value / 2.
    assert popt_apic[0] < soma_amp_value * 2.
    assert popt_apic[1] > 0
    assert popt_apic[2] < soma_amp_value / 10.


def test_validate(mock_bpap_amplitude_distance):
    """Test the validate method of the BPAP class."""
    cell = DummyCell()
    bpap = BPAP(cell)
    # good data
    validated, notes = bpap.validate(*mock_bpap_amplitude_distance)
    assert validated is True
    assert notes == ("Dendritic validation passed: dendritic amplitude is decaying with distance "
        "relative to soma.\nApical validation passed: apical amplitude is decaying with distance "
        "relative to soma.\n"
    )
    # empty data: validated is True, but we have some logging in notes
    validated, notes = bpap.validate([100], None, None, None, None)
    assert validated is True
    assert notes == "No dendritic recordings found.\nNo apical recordings found.\n"
    # bad data: validated is False, and we have some logging in notes
    validated, notes = bpap.validate([100], [110, 120], [10, 20], [110, 120], [10, 20])
    assert validated is False
    assert notes == "Dendritic fit is not decaying.\nApical fit is not decaying.\n"


def test_plot_amp_vs_dist(mock_bpap_amplitude_distance, tmp_path):
    """Test the plot_amp_vs_dist method of the BPAP class."""
    fname = "tmp.pdf"
    outpath = tmp_path / fname
    cell = DummyCell()
    bpap = BPAP(cell)
    # good data
    bpap.plot_amp_vs_dist(
        *mock_bpap_amplitude_distance,
        show_figure=False,
        save_figure=True,
        output_dir=tmp_path,
        output_fname=fname
    )
    assert outpath.exists()
    # empty data: should not raise an error
    bpap.plot_amp_vs_dist([100], None, None, None, None, show_figure=False, save_figure=False)


def test_plot_one_axis_recordings():
    """Test the plot_one_axis_recordings function."""
    cell = DummyCell()
    bpap = BPAP(cell)
    bpap.stim_start = 0
    _, dend_rec, _ = bpap.get_recordings()
    dummy_dend_dist = [10, 20]
    fig, ax = plt.subplots()
    # Test with valid section and segment
    bpap.plot_one_axis_recordings(
        fig,
        ax,
        rec_list=list(dend_rec.values()),
        dist=dummy_dend_dist,
        cmap=bpap.basal_cmap,
    )
    assert ax.get_xlim() == (-0.1, 30.)
    assert ax.get_ylim() == (-72, -8)
    assert len(ax.collections) > 0
    assert ax.collections[-1].colorbar is not None
    assert isinstance(ax.collections[-1].colorbar, matplotlib.colorbar.Colorbar)


@patch('matplotlib.pyplot.show')
def test_plot_recordings(mock_pyplot_show, run_bpap, tmp_path):
    """Test the plot_recordings method of the BPAP class."""
    fname = "tmp.pdf"
    outpath = tmp_path / fname
    # good data
    outpath = run_bpap.plot_recordings(
        show_figure=True,
        save_figure=True,
        output_dir=tmp_path,
        output_fname=fname,
    )
    assert outpath.exists()
    mock_pyplot_show.assert_called_once()
