"""Tests for resolving SONATA node sets into cell ids, following neurodamus."""

import json
import logging
from pathlib import Path

import pytest

from bluecellulab.circuit import CellId, SonataCircuitAccess
from bluecellulab.circuit.circuit_access import BluepyCircuitAccess

projections_dir = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
)

BIOPHYSICAL_POP = "hippocampus_neurons"
VIRTUAL_POP = "hippocampus_projections"
VIRTUAL_POP_SIZE = 12045
ALL_BIOPHYSICAL = [CellId(BIOPHYSICAL_POP, i) for i in range(10)]
EXPLICIT = [CellId(BIOPHYSICAL_POP, 2), CellId(BIOPHYSICAL_POP, 7)]

SIMULATION_NODE_SETS = {
    "Explicit": {"population": BIOPHYSICAL_POP, "node_id": [7, 2]},
    "WithProjections": ["Mosaic", "HipProjections"],
    "Regex": {"mtype": {"$regex": "SP_.*"}},
    "PopulationList": {"population": [BIOPHYSICAL_POP, VIRTUAL_POP]},
    "MissingAttribute": {"population": BIOPHYSICAL_POP, "no_such_attribute": "x"},
    # neurodamus skips the whole population when one member fails in it
    "CompoundMissingAttribute": ["MissingAttribute", "Explicit"],
    "WrongType": {"synapse_class": 1},
}


def _circuit_access(
    tmp_path: Path, node_set: str | None = None, node_sets: dict | None = None
) -> SonataCircuitAccess:
    node_sets_file = tmp_path / "node_sets.json"
    node_sets_file.write_text(json.dumps(node_sets or SIMULATION_NODE_SETS))
    config = {
        "network": str(projections_dir / "circuit_config.json"),
        "node_sets_file": str(node_sets_file),
        "run": {"tstop": 10.0, "dt": 0.025, "random_seed": 1},
    }
    if node_set is not None:
        config["node_set"] = node_set
    config_path = tmp_path / "simulation_config.json"
    config_path.write_text(json.dumps(config))
    return SonataCircuitAccess(config_path)


@pytest.mark.parametrize(
    "node_set, expected",
    [
        ("Explicit", set(EXPLICIT)),
        ("Mosaic", set(ALL_BIOPHYSICAL)),
        ("Regex", set(ALL_BIOPHYSICAL)),
        # the virtual population lacks synapse_class
        ("Excitatory", set(ALL_BIOPHYSICAL)),
    ],
)
def test_get_target_cell_ids(tmp_path, node_set, expected):
    assert _circuit_access(tmp_path).get_target_cell_ids(node_set) == expected


def test_get_target_cell_ids_keeps_virtual_populations(tmp_path):
    access = _circuit_access(tmp_path)

    for node_set in ("PopulationList", "WithProjections"):
        cell_ids = access.get_target_cell_ids(node_set)
        assert len(cell_ids) == len(ALL_BIOPHYSICAL) + VIRTUAL_POP_SIZE
        assert set(ALL_BIOPHYSICAL) <= cell_ids


@pytest.mark.parametrize(
    "node_set", ["MissingAttribute", "CompoundMissingAttribute", "WrongType"]
)
def test_get_target_cell_ids_skips_failing_population(tmp_path, caplog, node_set):
    access = _circuit_access(tmp_path)

    with caplog.at_level(logging.WARNING):
        assert access.get_target_cell_ids(node_set) == set()
    assert f"SonataError for node set {node_set} from population {BIOPHYSICAL_POP}" in (
        caplog.text
    )


def test_get_target_cell_ids_unknown_node_set(tmp_path):
    with pytest.raises(KeyError, match="Missing"):
        _circuit_access(tmp_path).get_target_cell_ids("Missing")


@pytest.mark.parametrize(
    "node_sets",
    [
        {"Malformed": {"node_id": "x"}},
        {"A": {"mtype": "SP_PC"}, "UnknownMember": ["A", "Missing"]},
        {"A": {"mtype": "SP_PC"}, "InlineMember": ["A", {"mtype": "SP_PC"}]},
        {"NotADict": "A"},
    ],
)
def test_get_target_cell_ids_invalid_node_sets(tmp_path, node_sets):
    access = _circuit_access(tmp_path, node_sets=node_sets)
    with pytest.raises(ValueError, match="Invalid node sets"):
        access.get_target_cell_ids(next(iter(node_sets)))


@pytest.mark.parametrize(
    "node_set, expected",
    [
        ("Explicit", EXPLICIT),
        ("Mosaic", ALL_BIOPHYSICAL),
        ("Excitatory", ALL_BIOPHYSICAL),
        ("Inhibitory", []),
        ("WithProjections", ALL_BIOPHYSICAL),
        ("PopulationList", ALL_BIOPHYSICAL),
        ("HipProjections", []),
        ("CompoundMissingAttribute", []),
        (None, ALL_BIOPHYSICAL),
    ],
)
def test_get_simulation_cell_ids(tmp_path, node_set, expected):
    access = _circuit_access(tmp_path, node_set)

    assert access.config.node_set == node_set
    assert access.get_simulation_cell_ids() == expected


def test_get_simulation_cell_ids_unknown_node_set(tmp_path):
    access = _circuit_access(tmp_path, "Missing")
    with pytest.raises(KeyError, match="Missing"):
        access.get_simulation_cell_ids()


def test_get_simulation_cell_ids_bluepy_not_supported():
    access = object.__new__(BluepyCircuitAccess)
    with pytest.raises(NotImplementedError):
        access.get_simulation_cell_ids()
