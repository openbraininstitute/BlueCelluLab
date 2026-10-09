"""Tests for resolving the cells simulated by a SONATA simulation config."""

import json
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
ALL_BIOPHYSICAL = [CellId(BIOPHYSICAL_POP, i) for i in range(10)]

SIMULATION_NODE_SETS = {
    "Explicit": {"population": BIOPHYSICAL_POP, "node_id": [7, 2]},
    "WithProjections": ["Mosaic", "HipProjections"],
}


def _circuit_access(tmp_path: Path, node_set: str | None) -> SonataCircuitAccess:
    node_sets_file = tmp_path / "node_sets.json"
    node_sets_file.write_text(json.dumps(SIMULATION_NODE_SETS))
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
        ("Explicit", [CellId(BIOPHYSICAL_POP, 2), CellId(BIOPHYSICAL_POP, 7)]),
        ("Mosaic", ALL_BIOPHYSICAL),
        # property query, the virtual population lacks synapse_class
        ("Excitatory", ALL_BIOPHYSICAL),
        ("Inhibitory", []),
        ("WithProjections", ALL_BIOPHYSICAL),
        ("HipProjections", []),
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
