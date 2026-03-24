# Copyright 2025 Open Brain Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest

from bluecellulab.circuit_simulation import CircuitSimulation
from bluecellulab.reports.manager import ReportManager
from bluecellulab.reports.writers.compartment import CompartmentReportWriter

script_dir = Path(__file__).parent.parent


@pytest.fixture
def mock_cell():
    cell = MagicMock()
    cell.report_sites = {
        "test_report": [{"rec_name": "rec_0", "section": "soma[0]", "segx": 0.5}]
    }
    cell.get_recording = MagicMock(return_value=np.ones(10, dtype=np.float32))
    return cell


@pytest.fixture
def mock_cells(mock_cell):
    return {
        ("default", 1): mock_cell,
        ("default", 2): mock_cell,
        ("default", 3): mock_cell,
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
                "elements": [1, 2, 3],
            }
        },
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
                    "2": [["soma[0]", 0.5]],
                },
            }
        },
    }


def make_trace(length: int, value: float) -> np.ndarray:
    return (np.ones(length) * value).astype(np.float32)


def make_cell_for_report(
    *,
    report_name: str,
    rec_sites: list[dict],
    rec_to_trace: dict[str, np.ndarray],
) -> MagicMock:
    cell = MagicMock()
    cell.report_sites = {report_name: rec_sites}
    cell.get_recording = MagicMock(side_effect=lambda rec_name: rec_to_trace[rec_name])
    return cell


def test_write_node_set(tmp_path, mock_cells, mock_config_node_set):
    out = tmp_path / "report.h5"
    writer = CompartmentReportWriter(report_cfg=mock_config_node_set, output_path=out, sim_dt=0.1)

    writer.write(cells=mock_cells, tstart=0.0)

    assert out.exists()
    with h5py.File(out, "r") as f:
        assert "/report/default/data" in f
        data = f["/report/default/data"][:]

        # 10 time samples, 3 elements
        assert data.shape == (10, 3)
        assert np.allclose(data, 1.0)


def test_write_compartment_set(tmp_path, mock_config_compartment_set):
    out = tmp_path / "report.h5"

    c1 = make_cell_for_report(
        report_name="test_report",
        rec_sites=[{"rec_name": "rec_1", "section": "dend[0]", "segx": 0.3}],
        rec_to_trace={"rec_1": make_trace(10, 1.0)},
    )
    c2 = make_cell_for_report(
        report_name="test_report",
        rec_sites=[{"rec_name": "rec_2", "section": "soma[0]", "segx": 0.5}],
        rec_to_trace={"rec_2": make_trace(10, 2.0)},
    )

    cells = {("default", 1): c1, ("default", 2): c2}

    writer = CompartmentReportWriter(report_cfg=mock_config_compartment_set, output_path=out, sim_dt=0.1)
    writer.write(cells=cells, tstart=0.0)

    assert out.exists()
    with h5py.File(out, "r") as f:
        assert "/report/default/data" in f
        data = f["/report/default/data"][:]
        node_ids = f["/report/default/mapping/node_ids"][:]
        elem_ids = f["/report/default/mapping/element_ids"][:]
        ptrs = f["/report/default/mapping/index_pointers"][:]

        assert data.shape == (10, 2)
        assert node_ids.tolist() == [1, 2]
        assert elem_ids.tolist() == [0, 1]
        assert ptrs.tolist() == [0, 1, 2]

        assert np.allclose(data[:, 0], 1.0)
        assert np.allclose(data[:, 1], 2.0)


def test_compartment_set_multinode_order(tmp_path):
    out = tmp_path / "trace_merge.h5"
    tlen = 10

    cells = {
        ("NodeA", 2): make_cell_for_report(
            report_name="trace_merge",
            rec_sites=[{"rec_name": "r2", "section": "soma[0]", "segx": 0.5}],
            rec_to_trace={"r2": make_trace(tlen, 30.0)},
        ),
        ("NodeA", 0): make_cell_for_report(
            report_name="trace_merge",
            rec_sites=[{"rec_name": "r0", "section": "soma[0]", "segx": 0.5}],
            rec_to_trace={"r0": make_trace(tlen, 10.0)},
        ),
        ("NodeA", 1): make_cell_for_report(
            report_name="trace_merge",
            rec_sites=[{"rec_name": "r1", "section": "soma[0]", "segx": 0.5}],
            rec_to_trace={"r1": make_trace(tlen, 20.0)},
        ),
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
            "NodeA": {"population": "NodeA"},
        },
    }

    writer = CompartmentReportWriter(report_cfg=report_cfg, output_path=out, sim_dt=0.1)
    writer.write(cells=cells, tstart=0.0)

    assert out.exists()
    with h5py.File(out, "r") as f:
        data = np.array(f["/report/NodeA/data"])
        node_ids = np.array(f["/report/NodeA/mapping/node_ids"])

        assert data.shape == (tlen, 3)
        assert node_ids.tolist() == [0, 1, 2]
        assert np.allclose(data[:, 0], 10.0)
        assert np.allclose(data[:, 1], 20.0)
        assert np.allclose(data[:, 2], 30.0)


def test_compartment_set_multisegment_single_node(tmp_path):
    out = tmp_path / "trace_multisegment.h5"
    tlen = 10

    sites = [
        {"rec_name": "rsoma", "section": "soma[0]", "segx": 0.5},
        {"rec_name": "rdend2", "section": "dend[0]", "segx": 0.2},
        {"rec_name": "rdend3", "section": "dend[0]", "segx": 0.3},
        {"rec_name": "raxon7", "section": "axon[1]", "segx": 0.7},
    ]
    rec_to_trace = {s["rec_name"]: make_trace(tlen, 42.0) for s in sites}

    cells = {
        ("NodeA", 0): make_cell_for_report(
            report_name="trace_multisegment",
            rec_sites=sites,
            rec_to_trace=rec_to_trace,
        )
    }

    report_cfg = {
        "name": "trace_multisegment",
        "type": "compartment_set",
        "compartment_set": "NodeA",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {"NodeA": {"population": "NodeA"}},
    }

    writer = CompartmentReportWriter(report_cfg=report_cfg, output_path=out, sim_dt=0.1)
    writer.write(cells=cells, tstart=0.0)

    assert out.exists()
    with h5py.File(out, "r") as f:
        data = np.array(f["/report/NodeA/data"])
        node_ids = np.array(f["/report/NodeA/mapping/node_ids"])
        elem_ids = np.array(f["/report/NodeA/mapping/element_ids"])
        ptrs = np.array(f["/report/NodeA/mapping/index_pointers"])

        assert data.shape == (tlen, 4)
        assert node_ids.tolist() == [0, 0, 0, 0]
        assert elem_ids.tolist() == [0, 1, 2, 3]
        assert ptrs.tolist() == [0, 1, 2, 3, 4]
        assert np.allclose(data, 42.0)


class TestSimCompartmentSet:
    def setup_method(self):
        sim_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/simulation_config_compartment_set.json"
        )
        self.sim = CircuitSimulation(sim_path)
        dstut_cells = [("NodeA", 0), ("NodeA", 1)]

        self.sim.instantiate_gids(dstut_cells, add_stimuli=True, add_synapses=True)
        self.sim.run()

        sample_cell = next(iter(self.sim.cells.values()))
        if not hasattr(sample_cell, "get_recording") or not hasattr(sample_cell, "report_sites"):
            pytest.skip("Live cells do not expose report_sites/get_recording; update integration test to payload flow.")

        report_mgr = ReportManager(self.sim.circuit_access.config, self.sim.dt)
        report_mgr.write_all(self.sim.cells)

        self.file1_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/output_sonata_compartment_set/soma.h5"
        )
        self.file2_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/output_sonata_compartment_set/soma_compartment_set.h5"
        )
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
