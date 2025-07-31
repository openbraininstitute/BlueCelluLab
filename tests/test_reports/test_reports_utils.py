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

import pytest
from unittest.mock import MagicMock

from bluecellulab.reports.utils import (
    build_recording_sites,
    extract_spikes_from_cells,
)


@pytest.fixture
def mock_cell():
    cell = MagicMock()
    cell.cell_id.id = 42
    cell.add_variable_recording = MagicMock()
    return cell


def test_extract_spikes_from_cells_valid_cell():
    cell = MagicMock()
    cell.get_recorded_spikes.return_value = [1.1, 2.2]
    cells = {("default", 1): cell}

    spikes = extract_spikes_from_cells(cells)
    assert spikes == {"default": {1: [1.1, 2.2]}}


def test_extract_spikes_invalid_key_format():
    cells = {"invalidkey": MagicMock()}
    with pytest.raises(ValueError, match="could not be parsed"):
        extract_spikes_from_cells(cells)


def test_extract_spikes_invalid_cell_type():
    cells = {("default", 1): "not_a_cell"}
    with pytest.raises(TypeError, match="not a Cell object"):
        extract_spikes_from_cells(cells)


def test_build_recording_sites_compartment(mock_cell):
    mock_cfg = {"sections": "soma", "compartments": "center"}
    mock_cell.resolve_segments_from_config.return_value = [("sec", "soma[0]", 0.5)]

    cells = {("pop", 1): mock_cell}
    result = build_recording_sites(cells, [1], "pop", "compartment", mock_cfg, None)

    assert 1 in result
    assert result[1][0][2] == 0.5


def test_build_recording_sites_compartment_set(mock_cell):
    mock_cell.resolve_segments_from_compartment_set.return_value = [("sec", "dend[0]", 0.3)]
    cells = {("pop", 2): mock_cell}
    result = build_recording_sites(cells, [2], "pop", "compartment_set", {}, [[2, "dend[0]", 0.3]])

    assert 2 in result
    assert result[2][0][1] == "dend[0]"
