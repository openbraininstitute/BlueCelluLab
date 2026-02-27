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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bluecellulab.circuit.node_id import CellId
from bluecellulab.reports.utils import (
    build_recording_sites,
    collect_local_payload,
    collect_local_spikes,
    extract_spikes_from_cells,
    gather_payload_to_rank0,
    gather_recording_sites,
    merge_dicts,
    merge_spikes,
    payload_to_cells,
    prepare_recordings_for_reports,
)


class DummyCell:
    def __init__(self, targets, rec_names):
        self.targets = targets
        self.rec_names = rec_names
        self.report_sites = None

    def resolve_segments_from_config(self, _cfg):
        return self.targets

    def resolve_segments_from_compartment_set(self, _node_id, _compartment_nodes):
        return self.targets

    def configure_recording(self, _sites, _variable, _report_name):
        return self.rec_names


class DummyConfig:
    def __init__(self, report_entries, node_sets=None, compartment_sets=None):
        self._report_entries = report_entries
        self._node_sets = node_sets or {}
        self._compartment_sets = compartment_sets or {}

    def get_report_entries(self):
        return self._report_entries

    def get_node_sets(self):
        return self._node_sets

    def get_compartment_sets(self):
        return self._compartment_sets


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


def test_build_recording_sites_compartment():
    mock_cfg = {"sections": "soma", "compartments": "center"}
    mock_cell = MagicMock()
    mock_cell.resolve_segments_from_config.return_value = [("sec", "soma[0]", 0.5)]

    cells = {CellId("pop", 1): mock_cell}
    result = build_recording_sites(cells, [1], "pop", "compartment", mock_cfg, None)

    assert 1 in result
    assert result[1][0][2] == 0.5


def test_build_recording_sites_compartment_set():
    mock_cell = MagicMock()
    mock_cell.resolve_segments_from_compartment_set.return_value = [("sec", "dend[0]", 0.3)]
    cells = {CellId("pop", 2): mock_cell}
    result = build_recording_sites(cells, [2], "pop", "compartment_set", {}, [[2, "dend[0]", 0.3]])

    assert 2 in result
    assert result[2][0][1] == "dend[0]"


def test_build_recording_sites_handles_missing_and_unsupported():
    cells = {}
    assert build_recording_sites(cells, [1], "pop", "compartment", {}, None) == {}

    cells_with_one = {CellId("pop", 1): DummyCell(targets=[], rec_names=[])}
    with pytest.raises(NotImplementedError):
        build_recording_sites(cells_with_one, [1], "pop", "unknown", {}, None)


def test_prepare_recordings_for_reports_compartment_populates_report_sites(caplog):
    cell_id = CellId("popA", 7)
    targets = [("sec", "soma[0]", 0.5), ("sec", "dend[0]", 0.3)]
    cell = DummyCell(targets=targets, rec_names=["rec_soma", "rec_dend"])
    cells = {cell_id: cell}

    cfg = DummyConfig(
        report_entries={"r1": {"type": "compartment", "cells": "targets", "variable_name": "v"}},
        node_sets={"targets": {"population": "popA"}},
    )

    with caplog.at_level("WARNING"):
        recording_index, sites_index = prepare_recordings_for_reports(cells, cfg)

    assert not caplog.records
    assert recording_index[cell_id] == ["rec_soma", "rec_dend"]
    assert len(sites_index[cell_id]) == 2
    assert "r1" in cell.report_sites
    assert [s["rec_name"] for s in cell.report_sites["r1"]] == ["rec_soma", "rec_dend"]


def test_prepare_recordings_for_reports_warns_on_rec_mismatch(caplog):
    cell_id = CellId("popA", 8)
    targets = [("sec", "soma[0]", 0.5), ("sec", "dend[0]", 0.3)]
    cell = DummyCell(targets=targets, rec_names=["only_one"])
    cells = {cell_id: cell}

    cfg = DummyConfig(
        report_entries={"r1": {"type": "compartment", "cells": "targets", "variable_name": "v"}},
        node_sets={"targets": {"population": "popA"}},
    )

    with caplog.at_level("WARNING"):
        recording_index, sites_index = prepare_recordings_for_reports(cells, cfg)

    assert "Configured 1/2 recording sites" in caplog.text
    assert recording_index[cell_id] == ["only_one"]
    assert len(sites_index[cell_id]) == 1


def test_prepare_recordings_for_reports_unsupported_type():
    cell_id = CellId("popA", 1)
    cells = {cell_id: DummyCell(targets=[], rec_names=[])}
    cfg = DummyConfig(report_entries={"r": {"type": "unsupported"}})

    with pytest.raises(NotImplementedError):
        prepare_recordings_for_reports(cells, cfg)


def test_payload_to_cells_and_recorded_cell_access():
    class Sec:
        def name(self):
            return "soma[0]"

    payload = {"popA_3": {"recordings": {"neuron.h.soma[0](0.5)._ref_v": [1.0, 2.0, 3.0]}}}
    sites_index = {
        CellId("popA", 3): [{
            "report": "r1",
            "rec_name": "neuron.h.soma[0](0.5)._ref_v",
            "section": "soma[0]",
            "segx": 0.5,
        }]
    }

    out = payload_to_cells(payload, sites_index)
    rc = out[CellId("popA", 3)]
    np.testing.assert_array_equal(rc.get_recording("neuron.h.soma[0](0.5)._ref_v"), np.array([1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        rc.get_variable_recording("v", Sec(), 0.5),
        np.array([1, 2, 3], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="No recording"):
        rc.get_recording("missing")


def test_merge_helpers():
    assert merge_dicts([{"a": 1}, {"b": 2}]) == {"a": 1, "b": 2}
    assert merge_spikes([{"p": {1: [0.1]}}, {"p": {2: [0.2]}}]) == {"p": {1: [0.1], 2: [0.2]}}


def test_gather_recording_sites_merges_and_skips_empty():
    gathered = [
        {},
        {CellId("p", 1): [{"rec_name": "a"}]},
        {CellId("p", 1): [{"rec_name": "b"}], CellId("p", 2): [{"rec_name": "c"}]},
    ]
    merged = gather_recording_sites(gathered)
    assert [s["rec_name"] for s in merged[CellId("p", 1)]] == ["a", "b"]
    assert [s["rec_name"] for s in merged[CellId("p", 2)]] == ["c"]


def test_collect_local_payload_and_spikes():
    c1 = MagicMock()
    c1.get_recording.return_value = np.array([1.0, 2.0], dtype=np.float32)
    c1.get_recorded_spikes.return_value = [0.2, 0.5]

    c2 = MagicMock()
    c2.get_recorded_spikes.side_effect = RuntimeError("no spikes")

    cells = {CellId("p", 1): c1}
    recording_index = {CellId("p", 1): ["r1"], CellId("p", 2): ["r2"]}
    cell_ids = [CellId("p", 1), CellId("p", 2)]

    payload = collect_local_payload(cells, cell_ids, recording_index)
    assert payload == {"p_1": {"recordings": {"r1": [1.0, 2.0]}}}

    sim = SimpleNamespace(
        cells={CellId("p", 1): c1, CellId("p", 2): c2},
        spike_location="soma",
        spike_threshold=-20.0,
    )
    spikes = collect_local_spikes(sim, cell_ids)
    assert spikes == {"p": {1: [0.2, 0.5], 2: []}}


def test_gather_payload_to_rank0_and_nonzero():
    class FakePC:
        def __init__(self, rank):
            self._rank = rank

        def py_gather(self, obj, _root):
            # Simulate 2 ranks already gathered
            return [obj, obj]

        def id(self):
            return self._rank

    local_payload = {"p_1": {"recordings": {"r": [1.0]}}}
    local_spikes = {"p": {1: [0.1]}}

    rank1 = FakePC(rank=1)
    assert gather_payload_to_rank0(rank1, local_payload, local_spikes) == (None, None)

    rank0 = FakePC(rank=0)
    all_payload, all_spikes = gather_payload_to_rank0(rank0, local_payload, local_spikes)
    assert all_payload == {"p_1": {"recordings": {"r": [1.0]}}}
    assert all_spikes == {"p": {1: [0.1]}}
