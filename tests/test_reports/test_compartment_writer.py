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

import numpy as np
import h5py
from unittest.mock import MagicMock, patch

import pytest
from bluecellulab.reports.writers.compartment import CompartmentReportWriter


@pytest.fixture
def mock_cell():
    cell = MagicMock()
    cell.get_variable_recording = MagicMock(side_effect=lambda variable, section, segx: np.ones(10))
    return cell


@pytest.fixture
def mock_cells(mock_cell):
    return {
        ("default", 1): mock_cell,
        ("default", 2): mock_cell,
        ("default", 3): mock_cell
    }


@pytest.fixture
def mock_config_node_set():
    return {
        "name": "test_report",
        "type": "compartment",
        "cells": "soma_nodes",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {
            "soma_nodes": {
                "population": "default",
                "elements": [1, 2, 3]
            }
        }
    }


@pytest.fixture
def mock_config_compartment_set():
    return {
        "name": "test_report",
        "type": "compartment_set",
        "compartment_set": "custom_segments",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {
            "custom_segments": {
                "population": "default",
                "elements": {
                    "1": [["dend[0]", 0.3]],
                    "2": [["soma[0]", 0.5]]
                }
            }
        },
    }


@patch("bluecellulab.reports.writers.compartment.resolve_source_nodes")
@patch("bluecellulab.reports.writers.compartment.build_recording_sites")
def test_write_node_set(mock_build_sites, mock_resolve_nodes, tmp_path, mock_cells, mock_config_node_set):
    mock_resolve_nodes.return_value = ([1, 2, 3], None)
    mock_build_sites.return_value = {
        1: [(None, "soma[0]", 0.5)],
        2: [(None, "soma[0]", 0.5)],
        3: [(None, "soma[0]", 0.5)],
    }

    writer = CompartmentReportWriter(report_cfg=mock_config_node_set, output_path=tmp_path / "report.h5", sim_dt=0.1)
    writer.write(cells=mock_cells, tstart=0.0)

    assert (tmp_path / "report.h5").exists()
    with h5py.File(tmp_path / "report.h5", "r") as f:
        assert "/report/default/data" in f
        data = f["/report/default/data"][:]
        print(data)
        assert data.shape[0] == 10
        assert data.shape[1] == 3


@patch("bluecellulab.reports.writers.compartment.resolve_source_nodes")
@patch("bluecellulab.reports.writers.compartment.build_recording_sites")
def test_write_compartment_set(mock_build_sites, mock_resolve_nodes, tmp_path, mock_cells, mock_config_compartment_set):
    mock_resolve_nodes.return_value = ([1, 2], [["1", "dend[0]", 0.3], ["2", "soma[0]", 0.5]])
    mock_build_sites.return_value = {
        1: [(None, "dend[0]", 0.3)],
        2: [(None, "soma[0]", 0.5)]
    }

    writer = CompartmentReportWriter(report_cfg=mock_config_compartment_set, output_path=tmp_path / "report.h5", sim_dt=0.1)
    writer.write(cells=mock_cells, tstart=0.0)

    assert (tmp_path / "report.h5").exists()
    with h5py.File(tmp_path / "report.h5", "r") as f:
        assert "/report/default/data" in f
        assert f["/report/default/data"].shape[1] == 2
