import os
import tempfile
import h5py
import numpy as np
import pytest
from unittest import mock

from bluecellulab.reports.writers.spikes import SpikeReportWriter


@pytest.fixture
def tmp_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_spike_report_writer_creates_hdf5(tmp_output):
    output_path = os.path.join(tmp_output, "spikes.h5")

    writer = SpikeReportWriter({}, output_path, sim_dt=0.025)

    spikes_by_pop = {
        "popA": {
            1: [1.0, 2.0, 3.0],
            2: [0.5, 1.5],
        }
    }

    writer.write(spikes_by_pop)

    with h5py.File(output_path, "r") as f:
        assert "spikes" in f
        assert "popA" in f["spikes"]
        group = f["spikes/popA"]
        timestamps = group["timestamps"][:]
        node_ids = group["node_ids"][:]
        assert len(timestamps) == len(node_ids)
        assert np.all(np.diff(timestamps) >= 0)
        assert "units" in group["timestamps"].attrs
        assert group["timestamps"].attrs["units"] == "ms"


def test_spike_report_writer_warns_if_empty(tmp_output, caplog):
    output_path = os.path.join(tmp_output, "spikes_empty.h5")
    writer = SpikeReportWriter({}, output_path, sim_dt=0.025)

    spikes_by_pop = {"popA": {1: []}}
    with caplog.at_level("WARNING"):
        writer.write(spikes_by_pop)
    assert "No spikes to write for population" in caplog.text


def test_extract_spikes_from_cells_returns_correct_format():
    mock_cell = mock.Mock()
    mock_cell.get_recorded_spikes.return_value = [1.0, 2.0]

    cells = {("popA", 42): mock_cell}

    from bluecellulab.reports.utils import extract_spikes_from_cells

    result = extract_spikes_from_cells(cells)

    assert isinstance(result, dict)
    assert "popA" in result
    assert 42 in result["popA"]
    assert result["popA"][42] == [1.0, 2.0]


def test_extract_spikes_from_cells_raises_on_invalid_key():
    cells = {"invalid_key": mock.Mock()}
    from bluecellulab.reports.utils import extract_spikes_from_cells

    with pytest.raises(ValueError):
        extract_spikes_from_cells(cells)
