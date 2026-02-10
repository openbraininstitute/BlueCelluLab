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
import bluecellulab.circuit_simulation as circuit_simulation
from bluecellulab.circuit_simulation import CircuitSimulation
from bluecellulab.circuit import CellId, SynapseProperty
from bluecellulab.circuit.config.sections import ConnectionOverrides


class FakePC:
    def __init__(self, rank=0, gather_result=None):
        self._rank = rank
        self.gather_result = gather_result
        self.set_gid2node_calls = []
        self.cell_calls = []
        self.broadcasted = None

    def id(self):
        return self._rank

    def py_gather(self, data, root):
        self.gathered = (data, root)
        return self.gather_result or [data]

    def py_broadcast(self, value, root):
        self.broadcasted = (value, root)
        return value

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


def make_sim(pc=None):
    sim = CircuitSimulation.__new__(CircuitSimulation)
    sim.pc = pc
    sim._gid_stride = 1_000
    sim._pop_index = {"": 0}
    sim.dt = 0.1
    sim.spike_threshold = -20.0
    sim.spike_location = "soma"
    return sim


def test_init_pop_index_mpi_collects_all_populations():
    pc = FakePC(rank=0, gather_result=[["PostA", "SourceA"], ["PostB"]])
    sim = make_sim(pc=pc)

    syn_a = DummySynapse(source_pop="SourceA", pre_gid=1)
    cell_a = DummyCell(synapses={"s0": syn_a})
    cell_b = DummyCell(synapses={})

    sim.cells = {
        CellId("PostA", 10): cell_a,
        CellId("PostB", 11): cell_b,
    }

    sim._init_pop_index_mpi()

    assert sim._pop_index[""] == 0
    assert sim._pop_index["PostA"] == 1
    assert sim._pop_index["PostB"] == 2
    assert sim._pop_index["SourceA"] == 3


def test_register_gids_for_mpi_uses_global_mapping():
    pc = FakePC(rank=1)
    sim = make_sim(pc=pc)
    sim._pop_index = {"": 0, "PopX": 2}

    cell_id = CellId("PopX", 7)
    sim.cells = {cell_id: DummyCell()}

    sim._register_gids_for_mpi()

    expected_gid = sim.global_gid("PopX", 7)
    assert pc.set_gid2node_calls == [(expected_gid, 1)]
    assert pc.cell_calls == [(expected_gid, "netcon")]


def test_add_connections_mpi_uses_global_pre_gid(monkeypatch):
    pc = FakePC(rank=0)
    sim = make_sim(pc=pc)
    sim._pop_index = {"": 0, "PrePop": 1, "PostPop": 2}

    post_id = CellId("PostPop", 5)
    syn = DummySynapse(source_pop="PrePop", pre_gid=3)
    post_cell = DummyCell(synapses={"syn1": syn})
    sim.cells = {post_id: post_cell}

    created = []

    class FakeConnection:
        def __init__(self, synapse, pre_spiketrain, pre_gid=None, pre_cell=None,
                     stim_dt=None, parallel_context=None, spike_threshold=None,
                     spike_location=None):
            self.synapse = synapse
            self.pre_spiketrain = pre_spiketrain
            self.pre_gid = pre_gid
            self.pre_cell = pre_cell
            self.parallel_context = parallel_context
            self.weight = 1.0
            created.append(self)

    monkeypatch.setattr(circuit_simulation.bluecellulab, "Connection", FakeConnection)

    sim._add_connections(interconnect_cells=True)

    assert "syn1" in post_cell.connections
    conn = post_cell.connections["syn1"]
    assert conn is created[0]
    assert conn.pre_cell is None
    assert conn.parallel_context is pc
    assert conn.pre_gid == sim.global_gid("PrePop", 3)


def test_global_gid_uses_stride_and_pop_index():
    sim = make_sim(pc=None)
    sim._gid_stride = 1000
    sim._pop_index = {"": 0, "PopA": 2}

    assert sim.global_gid("PopA", 7) == 2 * 1000 + 7


def test_global_gid_raises_for_unknown_population():
    sim = make_sim(pc=None)
    sim._pop_index = {"": 0}
    with pytest.raises(KeyError):
        sim.global_gid("UnknownPop", 1)


def test_add_connections_skips_zero_weight_override(monkeypatch):
    sim = make_sim(pc=None)

    post_id = CellId("PostPop", 5)
    syn = DummySynapse(source_pop="PrePop", pre_gid=3)
    post_cell = DummyCell(synapses={"syn1": syn})
    sim.cells = {post_id: post_cell}

    overrides = [
        ConnectionOverrides(
            source="src", target="dst", delay=None, weight=0.0,
            spont_minis=None, synapse_configure=None, mod_override=None,
        )
    ]

    class FakeCircuitAccess:
        def __init__(self, ov):
            self._ov = ov
            self.config = type("cfg", (), {"connection_entries": lambda _self: self._ov})()

        def target_contains_cell(self, *_):
            return True

    sim.circuit_access = FakeCircuitAccess(overrides)

    # Monkeypatch Connection so it would raise if invoked
    monkeypatch.setattr(circuit_simulation.bluecellulab, "Connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Connection should not be created")))

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
            source="any", target="any", delay=1.5, weight=2.0,
            spont_minis=None, synapse_configure=None, mod_override=None,
        ),
        ConnectionOverrides(
            source="any", target="any", delay=4.0, weight=3.0,
            spont_minis=None, synapse_configure=None, mod_override=None,
        ),
    ]

    class FakeCircuitAccess:
        def __init__(self, ov):
            self._ov = ov
            self.config = type("cfg", (), {"connection_entries": lambda _self: self._ov})()

        def target_contains_cell(self, *_):
            return True  # everything matches for simplicity

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
