import types

import pytest

from bluecellulab.circuit_simulation import CircuitSimulation
from bluecellulab.stimulus.circuit_stimulus_definitions import Noise, Pulse


class FakeCell:
    def __init__(self, gid):
        self.gid = gid

        class DummySection:
            def name(self_inner):
                return f"soma[{gid}]"

        self.soma = DummySection()
        self.persistent = []
        self.calls = []

    def resolve_segments_from_compartment_set(self, node_id):
        return [(f"sec-{node_id}", "soma", 0.25)]

    def add_replay_noise(self, stimulus, section=None, segx=0.5):
        self.calls.append(("noise", section, segx, stimulus))

    def add_pulse(self, stimulus, section=None, segx=0.5):
        self.calls.append(("pulse", section, segx, stimulus))


class FakeConfig:
    def __init__(self, stimuli, compartment_sets=None):
        self._stimuli = stimuli
        self._compartment_sets = compartment_sets or {}

    def get_all_stimuli_entries(self):
        return self._stimuli

    def get_compartment_sets(self):
        return self._compartment_sets


class FakeCircuitAccess:
    def __init__(self, config, target_map=None):
        self.config = config
        self._target_map = target_map or {}

    def get_target_cell_ids(self, target):
        return set(self._target_map.get(target, []))

    def target_contains_cell(self, target, cell_id):
        return cell_id in self._target_map.get(target, [])


def make_dummy_sim(circuit_access, cells):
    dummy = types.SimpleNamespace()
    dummy.circuit_access = circuit_access
    dummy.cells = cells
    dummy.spike_threshold = -20.0
    dummy.spike_location = "soma"
    return dummy


def test_compartment_set_targets_resolved_and_dispatched():
    """Stimulus with compartment_set should resolve to custom section/segx."""
    noise = Noise(
        target="comp_set_1",
        delay=0.0,
        duration=1.0,
        mean_percent=10.0,
        variance=5.0,
        compartment_set="comp_set_1",
    )

    comp_sets = {
        "comp_set_1": {
            "population": "PopA",
            "compartment_set": [[0, "soma", 0.25]],
        }
    }

    cfg = FakeConfig([noise], compartment_sets=comp_sets)
    access = FakeCircuitAccess(cfg, target_map={"PopA": [1]})

    cell = FakeCell(1)
    cells = {1: cell}

    sim = make_dummy_sim(access, cells)

    CircuitSimulation._add_stimuli(sim, add_noise_stimuli=True)

    assert len(cell.calls) == 1
    call = cell.calls[0]
    assert call[0] == "noise"
    assert call[1] == "sec-1"
    assert call[2] == 0.25


def test_non_compartment_target_uses_soma_half():
    """Stimulus with node_set only should default to soma at segx=0.5."""
    pulse = Pulse(
        target="node_set_A",
        delay=0.0,
        duration=1.0,
        amp_start=0.1,
        width=0.5,
        frequency=1.0,
        node_set="node_set_A",
    )

    cfg = FakeConfig([pulse], compartment_sets={})
    access = FakeCircuitAccess(cfg, target_map={"node_set_A": [2]})

    cell = FakeCell(2)
    cells = {2: cell}

    sim = make_dummy_sim(access, cells)

    CircuitSimulation._add_stimuli(sim, add_pulse_stimuli=True)

    assert len(cell.calls) == 1
    call = cell.calls[0]
    assert call[0] == "pulse"
    assert call[1] == cell.soma
    assert call[2] == 0.5


def test_name_collision_node_set_precedence():
    """If a compartment_set with the same name exists, a stimulus that
    uses node_set should still be treated as a node_set (soma/0.5).
    """
    pulse = Pulse(
        target="Mosaic_A",
        delay=0.0,
        duration=1.0,
        amp_start=0.1,
        width=0.5,
        frequency=1.0,
        node_set="Mosaic_A",
    )

    comp_sets = {
        "Mosaic_A": {
            "population": "NodeA",
            "compartment_set": [[0, "soma[0]", 0.25]],
        }
    }

    cfg = FakeConfig([pulse], compartment_sets=comp_sets)
    access = FakeCircuitAccess(cfg, target_map={"Mosaic_A": [4]})

    cell = FakeCell(4)
    cells = {4: cell}

    sim = make_dummy_sim(access, cells)

    CircuitSimulation._add_stimuli(sim, add_pulse_stimuli=True)

    assert len(cell.calls) == 1
    call = cell.calls[0]
    assert call[0] == "pulse"
    assert call[1] == cell.soma
    assert call[2] == 0.5


def test_missing_target_fields_raises(monkeypatch):
    """A stimulus with neither node_set nor compartment_set should raise."""
    pulse = Pulse(
        target="BrokenTarget",
        delay=0.0,
        duration=1.0,
        amp_start=0.1,
        width=0.5,
        frequency=1.0,
    )

    cfg = FakeConfig([pulse], compartment_sets={})
    access = FakeCircuitAccess(cfg, target_map={})

    cell = FakeCell(5)
    cells = {5: cell}
    sim = make_dummy_sim(access, cells)

    with pytest.raises(ValueError, match="neither node_set nor compartment_set"):
        CircuitSimulation._add_stimuli(sim, add_pulse_stimuli=True)
