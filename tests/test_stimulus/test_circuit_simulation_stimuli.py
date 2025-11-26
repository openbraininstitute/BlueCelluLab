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

import json

from bluecellulab.circuit.config.sonata_simulation_config import SonataSimulationConfig
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

    def resolve_segments_from_compartment_set(self, node_id, comp_nodes):
        return [(f"sec-{node_id}", "soma", 0.25)]

    def add_replay_noise(self, stimulus, noise_seed=None, noisestim_count=0, section=None, segx=0.5):
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
    sim = CircuitSimulation.__new__(CircuitSimulation)
    sim.circuit_access = circuit_access
    sim.cells = cells
    sim.spike_threshold = -20.0
    sim.spike_location = "soma"
    return sim


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
    access = FakeCircuitAccess(cfg)

    cell = FakeCell(1)
    cell.population_name = "PopA"
    cell.id = 1

    cells = {cell: cell}

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


def test_stimulus_with_both_node_and_compartment_set_raises(tmp_path):
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type("impl", (), {
        "config": {
            "inputs": {
                "stim_1": {
                    "module": "linear",
                    "node_set": "Mosaic_A",
                    "compartment_set": "cs1",
                    "delay": 0.0,
                    "duration": 1.0,
                    "amp_start": 0.1,
                }
            }
        }
    })()
    sim._get_config_dir = lambda: str(tmp_path)

    with pytest.raises(ValueError, match="must not include both 'node_set' and 'compartment_set'"):
        sim.get_all_stimuli_entries()


def test_stimulus_compartment_set_without_file_raises(tmp_path):
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type("impl", (), {
        "config": {
            "inputs": {
                "stim_1": {
                    "module": "linear",
                    "compartment_set": "cs1",
                    "delay": 0.0,
                    "duration": 1.0,
                    "amp_start": 0.1,
                }
            }
        }
    })()
    sim._get_config_dir = lambda: str(tmp_path)

    msg = "SONATA simulation config references 'compartment_set' in inputs but no 'compartment_sets_file' is configured."
    with pytest.raises(ValueError, match="compartment_set.*no 'compartment_sets_file'"):
        sim.get_all_stimuli_entries()


def _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets):
    comp_file = tmp_path / "compartment_sets.json"
    comp_file.write_text(json.dumps(comp_sets))
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type("impl", (), {
        "config": {
            "inputs": inputs,
            "compartment_sets_file": str(comp_file),
        }
    })()
    sim._get_config_dir = lambda: str(tmp_path)
    return sim


def test_compartment_set_name_not_found_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs_missing",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    comp_sets = {}  # no "cs_missing"

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="Compartment set 'cs_missing' not found"):
        sim.get_all_stimuli_entries()


def test_compartment_set_without_compartment_list_key_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs1",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    comp_sets = {
        "cs1": {
            "population": "Mosaic",
            # no "compartment_set" key
        }
    }

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="does not contain 'compartment_set' key"):
        sim.get_all_stimuli_entries()


def test_compartment_set_invalid_entry_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs1",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    # entry is not a [node_id, section, seg] triple
    comp_sets = {
        "cs1": {
            "population": "Mosaic",
            "compartment_set": [["bad"]],
        }
    }

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="Invalid compartment_set entry"):
        sim.get_all_stimuli_entries()


def test_compartment_set_unsorted_entries_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs1",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    comp_sets = {
        "cs1": {
            "population": "Mosaic",
            "compartment_set": [
                [1, "soma", 0.5],
                [0, "soma", 0.5],  # out of order
            ],
        }
    }

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="must be sorted ascending"):
        sim.get_all_stimuli_entries()


def test_compartment_set_duplicate_entry_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs1",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    comp_sets = {
        "cs1": {
            "population": "Mosaic",
            "compartment_set": [
                [0, "soma", 0.5],
                [0, "soma", 0.5],  # duplicate
            ],
        }
    }

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="contains duplicate entry"):
        sim.get_all_stimuli_entries()


def test_compartment_set_non_comparable_entries_raises(tmp_path):
    inputs = {
        "stim_1": {
            "module": "linear",
            "compartment_set": "cs1",
            "delay": 0.0,
            "duration": 1.0,
            "amp_start": 0.1,
        }
    }
    # Mixed types in node_id so tuple comparison raises TypeError internally
    comp_sets = {
        "cs1": {
            "population": "Mosaic",
            "compartment_set": [
                [0, "soma", 0.5],
                ["bad", "soma", 0.5],
            ],
        }
    }

    sim = _make_sim_with_inputs_and_comp_sets(tmp_path, inputs, comp_sets)

    with pytest.raises(ValueError, match="contains non-comparable entries"):
        sim.get_all_stimuli_entries()


def test_compartment_set_applied_only_to_matching_population():
    """Compartment set should be restricted to its population, not all cells."""

    noise = Noise(
        target="Mosaic_A",
        delay=0.0,
        duration=1.0,
        mean_percent=10.0,
        variance=5.0,
        compartment_set="Mosaic_A",
    )

    comp_sets = {
        "Mosaic_A": {
            "population": "NodeA",
            "compartment_set": [[0, "soma[0]", 0.5]],
        }
    }

    cfg = FakeConfig([noise], compartment_sets=comp_sets)
    access = FakeCircuitAccess(cfg)

    cell_a = FakeCell(gid=0)
    cell_a.population_name = "NodeA"
    cell_a.id = 0

    cell_b = FakeCell(gid=1)
    cell_b.population_name = "NodeB"
    cell_b.id = 1

    cells = {cell_a: cell_a, cell_b: cell_b}

    sim = make_dummy_sim(access, cells)

    CircuitSimulation._add_stimuli(sim, add_noise_stimuli=True)

    # Stimulus should be applied only to NodeA, never to NodeB
    assert len(cell_a.calls) == 1
    assert cell_a.calls[0][0] == "noise"

    assert cell_b.calls == []
