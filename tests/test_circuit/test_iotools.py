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

"""Unit tests for SONATA input/output operations."""

import pytest
import numpy as np
import h5py
from pathlib import Path

from bluecellulab.simulation.report import (
    write_sonata_report_file
)
from bluecellulab.cell import Cell
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
    return cell


def test_write_sonata_report_file(tmp_path):
    """Test writing SONATA report file directly."""
    data_matrix = [np.sin(np.linspace(0, 2 * np.pi, 100)) for _ in range(3)]
    node_ids = [1, 1, 1]
    index_pointers = [0, 1, 2, 3]
    element_ids = [0, 1, 2]
    report_cfg = {
        "start_time": 0.0,
        "end_time": 10.0,
        "dt": 0.1
    }

    output_file = tmp_path / "test_direct_report.h5"
    write_sonata_report_file(
        str(output_file),
        "TestPop",
        data_matrix,
        node_ids,
        index_pointers,
        element_ids,
        report_cfg
    )

    with h5py.File(output_file, 'r') as f:
        assert "report/TestPop" in f
        report = f["report/TestPop"]

        # Check data dimensions and type
        data = report["data"]
        assert data.shape == (100, 3)
        assert data.dtype == np.float32
        assert data.attrs["units"] == "mV"

        # Verify mapping data
        mapping = report["mapping"]
        assert np.array_equal(mapping["node_ids"][:], [1, 1, 1])
        assert np.array_equal(mapping["index_pointers"][:], [0, 1, 2, 3])
        assert np.array_equal(mapping["element_ids"][:], [0, 1, 2])
        assert np.array_equal(mapping["time"][:], [0.0, 10.0, 0.1])
        assert mapping["time"].attrs["units"] == "ms"
