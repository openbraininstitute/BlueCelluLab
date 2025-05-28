# Copyright 2025 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for simulation compartment reports."""


from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from bluecellulab import CircuitSimulation
from bluecellulab.simulation.report import configure_all_reports, _configure_recording
from bluecellulab.cell import Cell
from bluecellulab.cell.cell_dict import CellDict
from bluecellulab.circuit.circuit_access import EmodelProperties


@pytest.fixture
def mock_cell():
    """Create a mock cell for testing."""
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    cell = Cell(
        f"{Path(__file__).parent.parent}/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
        f"{Path(__file__).parent.parent}/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
        template_format="v6",
        emodel_properties=emodel_properties
    )
    cell.cell_id = ("TestPop", 1)
    return cell


def test_compartment_report_uses_output_dir(mock_cell, tmp_path):
    """Test that report file is written under config['output']['output_dir'] using the report key as filename."""
    from pathlib import Path
    import h5py

    simulation_config = f"{Path(__file__).parent.parent}/examples/sim_quick_scx_sonata_multicircuit/simulation_config_hypamp.json"
    sim = CircuitSimulation(simulation_config)

    cell_ids = [("NodeA", 0)]
    sim.cells = CellDict()
    sim.cells[("NodeA", 0)] = mock_cell
    mock_cell.add_voltage_recording(mock_cell.soma, 0.5)

    # Inject report config manually
    report_cfg = {
        "type": "compartment",
        "cells": "test_nodes",
        "sections": "soma",
        "compartments": "center",
        "variable_name": "v",
        "dt": 0.1,
        "start_time": 0.0,
        "end_time": 10.0
    }

    sim.circuit_access.config.get_report_entries = lambda: {
        "soma": report_cfg
    }
    sim.circuit_access.config.get_node_sets = lambda: {
        "test_nodes": {"population": "NodeA", "node_id": [0]}
    }

    # Override report_file_path to resolve inside tmp_path
    sim.circuit_access.config.report_file_path = lambda report_cfg, report_key: tmp_path / f"{report_key}.h5"
    # Run simulation
    sim.instantiate_gids(cell_ids, add_stimuli=True, add_synapses=False)
    sim.run(t_stop=10.0)
    sim.write_reports()

    # Check expected file
    output_file = tmp_path / "soma.h5"
    assert output_file.exists()

    with h5py.File(output_file, "r") as f:
        assert "report/NodeA" in f
        report = f["report/NodeA"]
        assert "data" in report
        assert "mapping" in report

        data = report["data"][:]
        assert data.shape[1] == 1  # one compartment
        assert data.shape[0] > 0   # some time steps

        mapping = report["mapping"]
        assert "node_ids" in mapping
        assert "index_pointers" in mapping
        assert "element_ids" in mapping
        assert "time" in mapping


def test_invalid_report_type():
    """Test error handling for invalid report types."""
    sim = CircuitSimulation(f"{Path(__file__).parent.parent}/examples/sim_quick_scx_sonata_multicircuit/simulation_config_hypamp.json")

    # Mock invalid report config
    report_cfg = {
        "type": "invalid_type",
        "cells": "all_cells",
        "section": "soma"
    }

    with pytest.raises(NotImplementedError, match=r"Report type 'invalid_type' is not supported"):
        configure_all_reports(sim.cells, type("MockConfig", (), {"get_report_entries": lambda: {"test": report_cfg}}))


def test_invalid_compartments_value(mock_cell):
    """Test error handling for invalid compartments value in node-based recording."""
    from types import SimpleNamespace
    from pathlib import Path
    import pytest

    # Instantiate a dummy CircuitSimulation without calling __init__
    sim = CircuitSimulation.__new__(CircuitSimulation)

    # Set up cell dictionary with one mock cell
    sim.cells = CellDict()
    sim.cells[("TestPop", 1)] = mock_cell

    # Mock circuit access and config
    sim.circuit_access = SimpleNamespace()
    sim.circuit_access.config = SimpleNamespace()

    # Mock the report entries to include an invalid 'compartments' value
    sim.circuit_access.config.get_report_entries = lambda: {
        "invalid_report": {
            "type": "compartment",
            "cells": "all_cells",
            "section": "soma[0]",
            "compartments": "invalid",  # This should trigger the ValueError
            "variable_name": "v"
        }
    }

    # Provide a dummy node set that would pass if 'compartments' were valid
    sim.circuit_access.config.get_node_sets = lambda: {
        "all_cells": {"population": "TestPop", "node_id": [1]}
    }

    # Inject a dummy report_file_path method to satisfy the call in write_reports
    sim.circuit_access.config.report_file_path = lambda report_cfg, report_key: Path(f"{report_key}.h5")

    # Expect ValueError due to invalid compartments value
    with pytest.raises(ValueError, match=r"Unsupported 'compartments' value 'invalid'"):
        sim.write_reports()


def test_unsupported_variable(mock_cell):
    """Test handling of unsupported recording variables."""
    sim = CircuitSimulation(f"{Path(__file__).parent.parent}/examples/sim_quick_scx_sonata_multicircuit/simulation_config_hypamp.json")

    # Configure recording with unsupported variable
    report_cfg = {
        "type": "compartment",
        "cells": "all_cells",
        "section": "soma[0]",
        "compartments": "center",
        "variable_name": "unsupported"
    }

    # Should log warning but not raise error
    _configure_recording(mock_cell, report_cfg, {"population": "TestPop", "node_id": [1]}, "node_set", "test_report")


def test_configure_recording_adds_variable_recording():
    """Test that _configure_recording calls add_variable_recording correctly for a supported variable."""
    cell = Mock()
    cell.cell_id = 1
    report_cfg = {"variable_name": "v"}
    source = {"population": "TestPop", "node_id": [1]}
    source_type = "node_set"
    report_name = "test_report"
    targets = [("soma_section", "soma", 0.5)]

    with patch("bluecellulab.simulation.report.resolve_segments", return_value=targets):
        _configure_recording(cell, report_cfg, source, source_type, report_name)

    cell.add_variable_recording.assert_called_once_with(variable="v", section="soma_section", segx=0.5)


def test_configure_recording_unsupported_variable_logs_warning():
    """Test that _configure_recording logs a warning for unsupported variable (AttributeError)."""
    cell = Mock()
    cell.cell_id = 1
    cell.add_variable_recording.side_effect = AttributeError("Unsupported variable")
    report_cfg = {"variable_name": "unsupported"}
    source = {"population": "TestPop", "node_id": [1]}
    source_type = "node_set"
    report_name = "test_report"
    targets = [("soma_section", "soma", 0.5)]

    with patch("bluecellulab.simulation.report.resolve_segments", return_value=targets), \
         patch("bluecellulab.simulation.report.logger") as mock_logger:
        _configure_recording(cell, report_cfg, source, source_type, report_name)

    mock_logger.warning.assert_any_call("Recording for variable 'unsupported' is not implemented in Cell.")
    cell.add_variable_recording.assert_called_once_with(variable="unsupported", section="soma_section", segx=0.5)
