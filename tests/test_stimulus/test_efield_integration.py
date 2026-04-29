"""Integration tests for spatially-uniform e-field stimulus (PR #82)."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bluecellulab import CircuitSimulation
from bluecellulab.stimulus import circuit_stimulus_definitions

parent_dir = Path(__file__).resolve().parent.parent.parent


def _fake_efield(node_set="Mosaic_A", compartment_set=None):
    """Return a minimal SpatiallyUniformEField stimulus."""
    target = compartment_set if compartment_set is not None else node_set
    return circuit_stimulus_definitions.SpatiallyUniformEField(
        target=target,
        node_set=None if compartment_set is not None else node_set,
        compartment_set=compartment_set,
        fields=[{"Ex": 100, "Ey": 0, "Ez": 0, "frequency": 0}],
        delay=0,
        duration=10,
        ramp_up_time=0,
        ramp_down_time=0,
    )


def _instantiate_with_efield(sim, gids, **patches):
    """Instantiate cells with the patched stimulus list and skip the apply step
    so that segment_displacements stay populated for inspection."""
    with patch.object(
        sim,
        "_apply_extracellular_stimuli",
        lambda: None,
    ):
        with patch.object(
            sim.circuit_access.config,
            "get_all_stimuli_entries",
            **patches,
        ):
            sim.instantiate_gids(
                cells=gids,
                add_stimuli=True,
                add_extracellular_stimuli=True,
            )


def test_efield_targets_all_sections():
    """EField stimulus on node_set should target all sections, not just soma."""
    base = parent_dir / "examples" / "2-sonata-network" / "sim_quick_scx_sonata_multicircuit" / "simulation_config_noinput.json"
    sim = CircuitSimulation(base)
    target = sim.circuit_access.get_target_cell_ids("Mosaic_A")
    gids = list(target)[:1]
    _instantiate_with_efield(sim, gids, return_value=[_fake_efield()])
    cell = sim.cells[gids[0]]
    assert gids[0] in sim._efield_sources
    es = sim._efield_sources[gids[0]]
    assert es.segment_displacements is not None
    assert len(es.segment_displacements) > 1


def test_efield_soma_displacement_is_zero():
    """Soma segments should have zero displacement vector."""
    base = parent_dir / "examples" / "2-sonata-network" / "sim_quick_scx_sonata_multicircuit" / "simulation_config_noinput.json"
    sim = CircuitSimulation(base)
    target = sim.circuit_access.get_target_cell_ids("Mosaic_A")
    gids = list(target)[:1]
    _instantiate_with_efield(sim, gids, return_value=[_fake_efield()])
    cell = sim.cells[gids[0]]
    es = sim._efield_sources[gids[0]]
    soma_seg = cell.soma(0.5)
    assert soma_seg in es.segment_displacements
    np.testing.assert_allclose(
        es.segment_displacements[soma_seg], [0.0, 0.0, 0.0], atol=1e-12
    )


def test_efield_compartment_set_targets():
    """EField stimulus should honour compartment_set targets."""
    base = parent_dir / "examples" / "2-sonata-network" / "sim_quick_scx_sonata_multicircuit" / "simulation_config_noinput.json"
    sim = CircuitSimulation(base)
    target = sim.circuit_access.get_target_cell_ids("Mosaic_A")
    gids = list(target)[:1]
    fake_compartment_sets = {
        "dend_only": {
            "population": gids[0].population_name,
            "compartment_set": [[gids[0].id, 1, 0.5]],
        }
    }
    with patch.object(
        sim.circuit_access.config,
        "get_compartment_sets",
        return_value=fake_compartment_sets,
    ):
        _instantiate_with_efield(
            sim, gids, return_value=[_fake_efield(compartment_set="dend_only")]
        )
    cell = sim.cells[gids[0]]
    es = sim._efield_sources[gids[0]]
    assert es.segment_displacements is not None
    # Should have at least one segment displacement, and soma(0.5) is not present.
    assert len(es.segment_displacements) >= 1
    assert cell.soma(0.5) not in es.segment_displacements


def test_get_cell_position_rotation_returns_quaternion():
    """SonataCircuitAccess returns position and quaternion when present."""
    base = parent_dir / "examples" / "2-sonata-network" / "sim_quick_scx_sonata_multicircuit" / "simulation_config_noinput.json"
    sim = CircuitSimulation(base)
    target = sim.circuit_access.get_target_cell_ids("Mosaic_A")
    gid = next(iter(target))
    pos, quat = sim.circuit_access.get_cell_position_rotation(gid)
    assert pos.shape == (3,)
    assert quat is not None
    assert quat.shape == (4,)


def test_cell_transform_is_populated():
    """Cell transform should be populated during instantiate_gids when SONATA coords exist."""
    base = parent_dir / "examples" / "2-sonata-network" / "sim_quick_scx_sonata_multicircuit" / "simulation_config_noinput.json"
    sim = CircuitSimulation(base)
    target = sim.circuit_access.get_target_cell_ids("Mosaic_A")
    gid = next(iter(target))
    sim.instantiate_gids(cells=[gid])
    cell = sim.cells[gid]
    assert cell._local_to_global_matrix is not None
    # Verify the translation part matches SONATA node position
    pos, _ = sim.circuit_access.get_cell_position_rotation(gid)
    np.testing.assert_allclose(cell._local_to_global_matrix[:, 3], pos)
