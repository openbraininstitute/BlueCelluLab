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

"""Unit tests for section resolution tools."""

import pytest
from pathlib import Path
from bluecellulab.cell import Cell
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.tools import get_section, get_sections, resolve_segments


@pytest.fixture
def mock_cell():
    """Create a mock cell for testing."""
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    return Cell(
        f"{Path(__file__).parent.parent}/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
        f"{Path(__file__).parent.parent}/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
        template_format="v6",
        emodel_properties=emodel_properties
    )


def test_get_section_direct_match(mock_cell):
    """Test get_section with a direct section name match."""
    section = get_section(mock_cell, "soma[0]")
    assert hasattr(section, "nseg")


def test_get_section_soma_fallback(mock_cell):
    """Test get_section fallback from 'soma' to 'soma[0]'."""
    section = get_sections(mock_cell, "soma")
    assert hasattr(section[0], "nseg")


def test_get_section_invalid_name(mock_cell):
    """Test get_section with an invalid section name (not in cell.sections)."""
    with pytest.raises(ValueError, match=r"Section 'invalid' not found\. Available:"):
        get_sections(mock_cell, "invalid")


def test_resolve_segments_node_set(mock_cell):
    """Test resolve_segments for node_set recording."""
    report_cfg = {
        "section": "soma",
        "compartments": "center"
    }
    targets = resolve_segments(mock_cell, report_cfg, 1, None, "node_set")
    _, sec_name, seg = targets[0]
    assert sec_name == "soma[0]"
    assert seg == 0.5


def test_resolve_segments_node_set_all(mock_cell):
    """Test resolve_segments for node_set recording with all compartments."""
    report_cfg = {
        "section": "dend[0]",
        "compartments": "all"
    }
    targets = resolve_segments(mock_cell, report_cfg, 1, None, "node_set")
    assert len(targets) == 1


def test_resolve_segments_compartment_set(mock_cell):
    """Test resolve_segments for compartment_set recording."""
    compartment_nodes = [[1, "soma", 0.5], [1, "dend[0]", 0.25]]
    targets = resolve_segments(mock_cell, {}, 1, compartment_nodes, "compartment_set")
    assert len(targets) == 2
    _, sec_name, seg = targets[0]
    assert sec_name == "soma"
    assert seg == 0.5
    _, sec_name, seg = targets[1]
    assert sec_name == "dend[0]"
    assert seg == 0.25
