# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for circuit/config/simulation_config.py."""

import json
from pathlib import Path

from bluepysnap import Simulation as SnapSimulation
import pytest

from bluecellulab.circuit.config import SonataSimulationConfig
from bluecellulab.circuit.config.sections import (
    ConditionEntry,
    Conditions,
    ConnectionOverrides,
    MechanismConditions,
)
from bluecellulab.stimulus.circuit_stimulus_definitions import (
    Noise,
    Hyperpolarizing,
    Linear,
    Pulse,
    RelativeLinear,
    ShotNoise,
    RelativeShotNoise,
    OrnsteinUhlenbeck,
    RelativeOrnsteinUhlenbeck,
)
from tests.helpers.os_utils import cwd


parent_dir = Path(__file__).resolve().parent.parent
cond_params_conf_path = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config.json"
)

multi_input_conf_path = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config_many_inputs.json"
)

hipp_sim_with_projection = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


def test_non_existing_config():
    with pytest.raises(FileNotFoundError):
        SonataSimulationConfig(parent_dir / "examples" / "non_existing_config")


def test_init_with_snap_simulation_config():
    snap_sim = SnapSimulation(cond_params_conf_path)
    SonataSimulationConfig(snap_sim)


def test_init_with_invalid_type():
    with pytest.raises(TypeError):
        SonataSimulationConfig({"invalid": "type", "dict": "config"})


def test_get_all_stimuli_entries():
    sim = SonataSimulationConfig(multi_input_conf_path)
    noise_stim = Noise(
        target="Mosaic_A",
        delay=10.0,
        duration=20.0,
        mean_percent=200.0,
        variance=0.001,
        node_set="Mosaic_A",
    )
    hyper_stim = Hyperpolarizing("Mosaic_A", 0.0, 50.0, node_set="Mosaic_A")
    pulse_stim = Pulse("Mosaic_A", 10.0, 20.0, 0.1, 25, 10, node_set="Mosaic_A")
    linear_stim = Linear("Mosaic_A", 10.0, 20.0, 0.1, 0.4, node_set="Mosaic_A")
    relative_linear_stim = RelativeLinear(
        "Mosaic_A", 10.0, 20.0, 50, 100, node_set="Mosaic_A"
    )
    shot_noise_stim = ShotNoise(
        "Mosaic_A", 10.0, 20, 2, 5, 10, 0.1, 0.02, 0.25, 42, node_set="Mosaic_A"
    )
    relative_shot_noise_stim = RelativeShotNoise(
        "Mosaic_A", 10.0, 20, 2, 5, 50, 10, 0.5, 0.25, 42, node_set="Mosaic_A"
    )
    ornstein_uhlenbeck_stim = OrnsteinUhlenbeck(
        "Mosaic_A", 10.0, 20.0, 5, 0.1, 0, 0.25, 42, node_set="Mosaic_A"
    )
    relative_ornstein_uhlenbeck_stim = RelativeOrnsteinUhlenbeck(
        "Mosaic_A", 10.0, 20.0, 5, 50, 10, 0.25, 42, node_set="Mosaic_A"
    )
    entries = sim.get_all_stimuli_entries()
    assert len(entries) == 9
    assert entries[0] == linear_stim
    assert entries[1] == ornstein_uhlenbeck_stim
    assert entries[2] == pulse_stim
    assert entries[3] == relative_linear_stim
    assert entries[4] == relative_ornstein_uhlenbeck_stim
    assert entries[5] == relative_shot_noise_stim
    assert entries[6] == shot_noise_stim
    assert entries[7] == noise_stim
    assert entries[8] == hyper_stim


def test_condition_parameters():
    sim = SonataSimulationConfig(cond_params_conf_path)
    conditions = sim.condition_parameters()
    assert conditions == Conditions(
        mech_conditions=MechanismConditions(
            ampanmda=ConditionEntry(minis_single_vesicle=0, init_depleted=1),
            gabaab=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
            glusynapse=ConditionEntry(minis_single_vesicle=None, init_depleted=None),
        ),
        mechanisms={
            "ProbAMPANMDA_EMS": {"init_depleted": True, "minis_single_vesicle": False},
            "ProbGABAAB_EMS": {"property_x": 1, "property_y": 0.25},
            "GluSynapse": {"property_z": "string"},
        },
        celsius=34.0,
        v_init=-80.0,
        extracellular_calcium=None,
        randomize_gaba_rise_time=False,
    )


def test_connection_entries():
    sim = SonataSimulationConfig(hipp_sim_with_projection)
    entries = sim.connection_entries()
    assert len(entries) == 4
    assert entries[0] == ConnectionOverrides(
        source="Mosaic",
        target="Mosaic",
        delay=None,
        weight=1.0,
        spont_minis=0.01,
        synapse_configure=None,
        mod_override=None,
    )
    assert (
        entries[2].synapse_configure
        == "%s.NMDA_ratio = 1.22 %s.tau_r_NMDA = 3.9 %s.tau_d_NMDA = 148.5"
    )
    assert entries[-1] == ConnectionOverrides(
        source="Excitatory",
        target="Mosaic",
        delay=None,
        weight=None,
        spont_minis=None,
        synapse_configure="%s.mg = 1.0",
        mod_override=None,
    )


def test_connection_override():
    sim = SonataSimulationConfig(hipp_sim_with_projection)

    entries = sim.connection_entries()
    assert len(entries) == 4

    connection_override = ConnectionOverrides(
        source="Excitatory",
        target="Mosaic",
        delay=2,
        weight=2.0,
        spont_minis=0.1,
        synapse_configure="%s.mg = 1.4",
        mod_override=None,
    )
    sim.add_connection_override(connection_override)

    entries = sim.connection_entries()
    assert len(entries) == 5
    assert entries[-1] == connection_override

    # overrides are not added multiple times
    entries = sim.connection_entries()
    entries = sim.connection_entries()
    assert len(entries) == 5


def test_connection_overrides_are_instance_scoped():
    sim_a = SonataSimulationConfig(cond_params_conf_path)
    sim_b = SonataSimulationConfig(cond_params_conf_path)

    override = ConnectionOverrides(
        source="A",
        target="B",
        delay=1.0,
        weight=0.5,
        spont_minis=None,
        synapse_configure=None,
        mod_override=None,
    )

    sim_a.add_connection_override(override)

    assert sim_a.connection_entries()[-1] == override
    assert override not in sim_b.connection_entries()


def test_get_all_projection_names():
    sim_dir = cond_params_conf_path.parent
    sim_config = cond_params_conf_path.name
    with cwd(sim_dir):
        sim = SonataSimulationConfig(sim_config)
        assert sim.get_all_projection_names() == []


def test_seeds():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.base_seed == 1
    assert sim.synapse_seed == 0
    assert sim.ionchannel_seed == 0
    assert sim.stimulus_seed == 0
    assert sim.minis_seed == 0


def test_rng_mode():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.rng_mode == "Random123"


def test_spike_threshold():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.spike_threshold == -30.0


def test_spike_location():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.spike_location == "soma"


def test_duration():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.duration == 50.0


def test_dt():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.dt == 0.025


def test_forward_skip():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.forward_skip is None


def test_celsius():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.celsius == 34.0


def test_v_init():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.v_init == -80.0


def test_output_root_path():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert Path(sim.output_root_path).name == "output_sonata"


def test_extracellular_calcium():
    sim = SonataSimulationConfig(cond_params_conf_path)
    assert sim.extracellular_calcium is None


def test_get_compartment_sets(tmp_path):
    file = tmp_path / "compartment_sets.json"
    file.write_text(
        json.dumps(
            {
                "soma_set": {
                    "population": "Mosaic",
                    "compartment_set": [[0, "soma", 0.5]],
                }
            }
        )
    )
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type("impl", (), {"config": {"compartment_sets_file": str(file)}})
    result = sim.get_compartment_sets()
    assert "soma_set" in result
    assert result["soma_set"]["population"] == "Mosaic"


def test_get_node_sets(tmp_path):
    # Circuit file content
    circuit_file = tmp_path / "circuit_node_sets.json"
    circuit_file.write_text(
        json.dumps(
            {
                "set_from_circuit": {"population": "PopA", "node_id": [1, 2]},
                "overwritten_set": {"population": "PopB", "node_id": [3]},
            }
        )
    )

    # Simulation file content
    sim_file = tmp_path / "sim_node_sets.json"
    sim_file.write_text(
        json.dumps(
            {
                "overwritten_set": {"population": "PopB", "node_id": [99]},
                "set_from_sim": {"population": "PopC", "node_id": [4]},
            }
        )
    )

    # Simulates the SonataSimulationConfig instance
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type(
        "impl",
        (),
        {
            "circuit": type(
                "circuit", (), {"config": {"node_sets_file": str(circuit_file)}}
            )(),
            "config": {"node_sets_file": str(sim_file)},
        },
    )

    # Call method
    result = sim.get_node_sets()

    # Validate merged result
    assert set(result) == {"set_from_circuit", "overwritten_set", "set_from_sim"}
    assert result["set_from_circuit"]["node_id"] == [1, 2]
    assert result["overwritten_set"]["node_id"] == [
        99
    ]  # Overwritten by simulation file
    assert result["set_from_sim"]["node_id"] == [4]


def test_get_node_sets_only_circuit(tmp_path):
    circuit_file = tmp_path / "node_sets.json"
    circuit_file.write_text(
        json.dumps({"only_circuit": {"population": "PopX", "node_id": [5]}})
    )
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type(
        "impl",
        (),
        {
            "circuit": type(
                "circuit", (), {"config": {"node_sets_file": str(circuit_file)}}
            )(),
            "config": {},
        },
    )
    result = sim.get_node_sets()
    assert "only_circuit" in result


def test_get_node_sets_only_sim(tmp_path):
    sim_file = tmp_path / "node_sets.json"
    sim_file.write_text(
        json.dumps({"only_sim": {"population": "PopY", "node_id": [6]}})
    )
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type(
        "impl",
        (),
        {
            "circuit": type("circuit", (), {"config": {}})(),
            "config": {"node_sets_file": str(sim_file)},
        },
    )
    result = sim.get_node_sets()
    assert "only_sim" in result


def test_get_node_sets_no_files():
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type(
        "impl", (), {"circuit": type("circuit", (), {"config": {}})(), "config": {}}
    )
    with pytest.raises(ValueError, match="No 'node_sets_file' found"):
        sim.get_node_sets()


def test_get_report_entries():
    sim = SonataSimulationConfig.__new__(SonataSimulationConfig)
    sim.impl = type(
        "impl",
        (),
        {
            "config": {
                "reports": {
                    "soma_v": {
                        "cells": "target_cells",
                        "section": "soma",
                        "variable_name": "v",
                        "compartments": "center",
                    }
                }
            }
        },
    )
    result = sim.get_report_entries()
    assert "soma_v" in result
    assert result["soma_v"]["variable_name"] == "v"
