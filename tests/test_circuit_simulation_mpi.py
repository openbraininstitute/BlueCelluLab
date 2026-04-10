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

import pytest
import bluecellulab.circuit_simulation as circuit_simulation
from bluecellulab.circuit_simulation import CircuitSimulation
from bluecellulab.circuit import CellId, SynapseProperty
from bluecellulab.circuit.config.sections import ConnectionOverrides


class FakeSonataCircuitAccess:
    """Minimal SONATA CircuitAccess stub for GID namespace tests."""

    def __init__(self, sizes: dict[str, int], virtual_sizes: dict[str, int] | None = None):
        self._sizes = dict(sizes)
        self._virtual_sizes = dict(virtual_sizes or {})

    def node_population_sizes(self) -> dict[str, int]:
        return dict(self._sizes)

    def virtual_population_sizes(self) -> dict[str, int]:
        return dict(self._virtual_sizes)


class FakePC:
    def __init__(self, rank=0, gather_result=None, broadcast_result=None):
        self._rank = rank
        self.gather_result = gather_result
        self.broadcast_result = broadcast_result
        self.set_gid2node_calls = []
        self.cell_calls = []
        self.broadcasted = None
        self.gathered = None

    def id(self):
        return self._rank

    def py_gather(self, data, root):
        self.gathered = (data, root)
        return self.gather_result if self.gather_result is not None else [data]

    def py_broadcast(self, value, root):
        self.broadcasted = (value, root)
        return self.broadcast_result if self.broadcast_result is not None else value

    def set_gid2node(self, gid, node):
        self.set_gid2node_calls.append((gid, node))

    def cell(self, gid, nc):
        self.cell_calls.append((gid, nc))


class DummySynapse:
    def __init__(self, source_pop, pre_gid, delay_weights=None):
        self.syn_description = {
            "source_population_name": source_pop,
            SynapseProperty.PRE_GID: pre_gid,
        }
        self.delay_weights = delay_weights or []


class DummyCell:
    def __init__(self, synapses=None):
        self.synapses = synapses or {}
        self.connections = {}

    def add_replay_delayed_weight(self, *args, **kwargs):
        return None

    def create_netcon_spikedetector(self, *_, **__):
        return "netcon"


def make_sim(*, pc=None, circuit_access=None):
    sim = CircuitSimulation.__new__(CircuitSimulation)

    sim.pc = pc
    sim.dt = 0.1
    sim.spike_threshold = -20.0
    sim.spike_location = "soma"
    sim.gids = None
    sim.projections = False
    sim.circuit_format = circuit_simulation.CircuitFormat.SONATA
    sim.circuit_access = (
        circuit_access if circuit_access is not None else FakeSonataCircuitAccess({})
    )
    sim.cells = {}

    return sim


def test_add_connections_skips_zero_weight_override(monkeypatch):
    sim = make_sim(pc=None)

    post_id = CellId("PostPop", 5)
    syn = DummySynapse(source_pop="PrePop", pre_gid=3)
    post_cell = DummyCell(synapses={"syn1": syn})
    sim.cells = {post_id: post_cell}

    overrides = [
        ConnectionOverrides(
            source="src",
            target="dst",
            delay=None,
            weight=0.0,
            spont_minis=None,
            synapse_configure=None,
            mod_override=None,
        )
    ]

    class FakeCircuitAccess:
        def __init__(self, ov):
            self._ov = ov
            self.config = type("cfg", (), {"connection_entries": lambda _self: self._ov})()

        def target_contains_cell(self, *_):
            return True

    sim.circuit_access = FakeCircuitAccess(overrides)

    monkeypatch.setattr(
        circuit_simulation.bluecellulab,
        "Connection",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("Connection should not be created")
        ),
    )

    sim._add_connections(interconnect_cells=False)

    assert post_cell.connections == {}


def test_add_connections_applies_last_matching_override(monkeypatch):
    sim = make_sim(pc=None)

    post_id = CellId("PostPop", 1)
    syn = DummySynapse(source_pop="PrePop", pre_gid=2)
    post_cell = DummyCell(synapses={"synX": syn})
    sim.cells = {post_id: post_cell}

    overrides = [
        ConnectionOverrides(
            source="any",
            target="any",
            synapse_delay_override=1.5,
            delay=None,
            weight=2.0,
            spont_minis=None,
            synapse_configure=None,
            mod_override=None,
        ),
        ConnectionOverrides(
            source="any",
            target="any",
            synapse_delay_override=4.0,
            delay=None,
            weight=3.0,
            spont_minis=None,
            synapse_configure=None,
            mod_override=None,
        ),
    ]

    class FakeCircuitAccess:
        def __init__(self, ov):
            self._ov = ov
            self.config = type("cfg", (), {"connection_entries": lambda _self: self._ov})()

        def target_contains_cell(self, *_):
            return True

    sim.circuit_access = FakeCircuitAccess(overrides)

    class FakeConnection:
        def __init__(self, *_, **__):
            self.weight = 1.0
            self.weight_scalar = 1.0
            self.post_netcon_weight = self.weight
            self.post_netcon_delay = 0.0

        def set_weight_scalar(self, scalar: float):
            self.weight_scalar = scalar
            self.post_netcon_weight = self.weight * self.weight_scalar

        def set_netcon_delay(self, delay: float):
            self.post_netcon_delay = delay

    monkeypatch.setattr(circuit_simulation.bluecellulab, "Connection", FakeConnection)

    sim._add_connections(interconnect_cells=False)

    conn = post_cell.connections["synX"]
    assert conn.post_netcon_delay == pytest.approx(4.0)
    assert conn.post_netcon_weight == pytest.approx(3.0)


def test_gid_namespace_offsets_are_1000_blocked_and_0_based():
    sim = make_sim(pc=None, circuit_access=FakeSonataCircuitAccess({"PopA": 3, "PopB": 2}))
    cell_ids = [CellId("PopA", 0), CellId("PopA", 2), CellId("PopB", 0), CellId("PopB", 1)]
    sim.gids = sim._build_gid_namespace(cell_ids)

    assert sim.global_gid("PopA", 0) == 0
    assert sim.global_gid("PopA", 2) == 2

    assert sim.global_gid("PopB", 0) == 1000
    assert sim.global_gid("PopB", 1) == 1001


def test_gid_namespace_does_not_depend_on_projections():
    sim = make_sim(pc=None, circuit_access=FakeSonataCircuitAccess({"PopA": 3, "PopB": 2}))
    cell_ids = [CellId("PopA", 0), CellId("PopA", 2), CellId("PopB", 1)]

    sim.projections = False
    gids1 = sim._build_gid_namespace(cell_ids)

    sim.projections = True
    gids2 = sim._build_gid_namespace(cell_ids)

    assert gids1.pop_offset == gids2.pop_offset


def test_register_gids_for_mpi_uses_gid_namespace():
    cell_ids = [CellId("PopX", 0), CellId("PopX", 7)]

    pc = FakePC(
        rank=1,
        gather_result=[{"PopX": 7}, {"PopX": 7}],
        broadcast_result={"PopX": 0},
    )
    sim = make_sim(pc=pc, circuit_access=FakeSonataCircuitAccess({"PopX": 10}))
    sim.gids = sim._build_gid_namespace(cell_ids)

    cell_id = CellId("PopX", 7)
    sim.cells = {cell_id: DummyCell()}

    sim._register_gids_for_mpi()

    expected_gid = sim.global_gid("PopX", 7)
    assert pc.set_gid2node_calls == [(expected_gid, 1)]
    assert pc.cell_calls == [(expected_gid, "netcon")]


def test_init_instantiated_cells_mpi_root_collects_union():
    cell_a = CellId("PopA", 1)
    cell_b = CellId("PopB", 2)
    cell_c = CellId("PopA", 3)

    pc = FakePC(
        rank=0,
        gather_result=[
            [cell_a, cell_b],
            [cell_c],
        ],
    )
    sim = make_sim(pc=pc)
    sim.cells = {
        cell_a: DummyCell(),
        cell_b: DummyCell(),
    }

    sim._init_instantiated_cells_mpi()

    assert pc.gathered == ([cell_a, cell_b], 0)
    assert pc.broadcasted == ({cell_a, cell_b, cell_c}, 0)
    assert sim._instantiated_cells_mpi == {cell_a, cell_b, cell_c}


def test_init_instantiated_cells_mpi_non_root_receives_broadcast():
    local_cell = CellId("PopA", 1)
    expected = {local_cell, CellId("PopB", 2)}

    pc = FakePC(
        rank=1,
        gather_result=None,
        broadcast_result=expected,
    )
    sim = make_sim(pc=pc)
    sim.cells = {local_cell: DummyCell()}

    sim._init_instantiated_cells_mpi()

    assert pc.gathered == ([local_cell], 0)
    assert pc.broadcasted == (None, 0)
    assert sim._instantiated_cells_mpi == expected
