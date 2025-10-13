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

from pathlib import Path
import numpy as np
import h5py
from unittest.mock import MagicMock, patch

import pytest
from bluecellulab.circuit_simulation import CircuitSimulation
from bluecellulab.reports.writers.compartment import CompartmentReportWriter
from bluecellulab.reports.manager import ReportManager

script_dir = Path(__file__).parent.parent


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


def make_trace(length, value):
    """Create a trace filled with a fixed value."""
    return (np.ones(length) * value).astype(np.float32)


def test_compartment_set_trace_mode_multinode_merge(tmp_path):
    """Ensure trace-mode data from multiple nodes is merged correctly by node ID order."""
    tlen = 10
    time = np.linspace(0, 1, tlen).tolist()

    traces = {
        "NodeA_2": {"time": time, "voltage": make_trace(tlen, 30.0)},
        "NodeA_0": {"time": time, "voltage": make_trace(tlen, 10.0)},
        "NodeA_1": {"time": time, "voltage": make_trace(tlen, 20.0)},
    }

    report_cfg = {
        "name": "trace_merge",
        "type": "compartment_set",
        "compartment_set": "NodeA",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {
            "NodeA": {
                "population": "NodeA",
                "compartment_set": [
                    [2, 0, 0.5],
                    [0, 0, 0.5],
                    [1, 0, 0.5]
                ]
            }
        }
    }

    with patch("bluecellulab.reports.utils.resolve_source_nodes") as mock_resolve, \
         patch("bluecellulab.reports.utils.build_recording_sites") as mock_build:

        mock_resolve.return_value = (
            [0, 1, 2],
            [[0, "soma[0]", 0.5], [1, "soma[0]", 0.5], [2, "soma[0]", 0.5]],
        )

        mock_build.return_value = {
            0: [(None, "soma[0]", 0.5)],
            1: [(None, "soma[0]", 0.5)],
            2: [(None, "soma[0]", 0.5)],
        }

        writer = CompartmentReportWriter(
            report_cfg=report_cfg,
            output_path=tmp_path / "trace_merge.h5",
            sim_dt=0.1
        )
        writer.write(cells=traces, tstart=0.0)

    with h5py.File(tmp_path / "trace_merge.h5", "r") as f:
        data = np.array(f["/report/NodeA/data"])
        node_ids = np.array(f["/report/NodeA/mapping/node_ids"])

        assert data.shape == (tlen, 3)
        assert node_ids.tolist() == [0, 1, 2]
        assert np.allclose(data[:, 0], 10.0)
        assert np.allclose(data[:, 1], 20.0)
        assert np.allclose(data[:, 2], 30.0)


def test_compartment_set_trace_mode_multisegment_node(tmp_path):
    """Test recording multiple segments from a single node in trace mode."""
    tlen = 10
    time = np.linspace(0, 1, tlen).tolist()

    traces = {
        "NodeA_0": {
            "time": time,
            "voltage": make_trace(tlen, 42.0)
        }
    }

    report_cfg = {
        "name": "trace_multisegment",
        "type": "compartment_set",
        "compartment_set": "NodeA",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {
            "NodeA": {
                "population": "NodeA",
                "compartment_set": [
                    [0, "soma[0]", 0.5],
                    [0, "dend[0]", 0.2],
                    [0, "dend[0]", 0.3],
                    [0, "axon[1]", 0.7]
                ]
            }
        }
    }

    writer = CompartmentReportWriter(
        report_cfg=report_cfg,
        output_path=tmp_path / "trace_multisegment.h5",
        sim_dt=0.1
    )
    writer.write(cells=traces, tstart=0.0)

    with h5py.File(tmp_path / "trace_multisegment.h5", "r") as f:
        data = np.array(f["/report/NodeA/data"])
        node_ids = np.array(f["/report/NodeA/mapping/node_ids"])
        elem_ids = np.array(f["/report/NodeA/mapping/element_ids"])
        ptrs = np.array(f["/report/NodeA/mapping/index_pointers"])

        assert data.shape == (tlen, 4)
        assert node_ids.tolist() == [0, 0, 0, 0]
        assert elem_ids.tolist() == [0, 1, 2, 3]
        assert ptrs.tolist() == [0, 1, 2, 3, 4]
        assert np.allclose(data, 42.0)


class TestSimCompartmentSet():
    """Test the graph.py module."""
    def setup_method(self):
        """Set up the test environment."""
        sim_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/simulation_config_compartment_set.json"
        )
        self.sim = CircuitSimulation(sim_path)
        dstut_cells = [('NodeA', 0), ('NodeA', 1)]

        self.sim.instantiate_gids(dstut_cells, add_stimuli=True, add_synapses=True)
        self.sim.run()

        report_mgr = ReportManager(self.sim.circuit_access.config, self.sim.dt)
        report_mgr.write_all(self.sim.cells)

        self.file1_path = f"{script_dir}/examples/sim_quick_scx_sonata_multicircuit/output_sonata_compartment_set/soma.h5"
        self.file2_path = f"{script_dir}/examples/sim_quick_scx_sonata_multicircuit/output_sonata_compartment_set/soma_compartment_set.h5"
        self.dataset_path = "/report/NodeA/data"

    def test_compartment_compartmentset_match(self):
        """Compare voltage reports from compartment and compartment_set output."""
        with h5py.File(self.file1_path, "r") as f1, h5py.File(self.file2_path, "r") as f2:
            assert self.dataset_path in f1, f"'{self.dataset_path}' not found in {self.file1_path}"
            assert self.dataset_path in f2, f"'{self.dataset_path}' not found in {self.file2_path}"

            data1 = np.array(f1[self.dataset_path])
            data2 = np.array(f2[self.dataset_path])

            assert data1.shape == data2.shape, f"Shape mismatch: {data1.shape} != {data2.shape}"
            assert np.allclose(data1, data2), "Data mismatch in dataset content"
