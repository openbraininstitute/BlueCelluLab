"""Wiring tests for ``GapJunctionManager`` / ``ContinuousConnectionManager``.

Runs ``CircuitSimulation.instantiate_gids`` against the hippocampus unit
circuit plus a synthetic ``"type": "electrical"`` edge population written
to a temporary SONATA edges.h5. Requires the ``gapdual`` test mechanism
(POINTER vpre + RANGE vgap dual-mode, like the celegans gapjunction.mod)
and the neurodamus-style ``Gap`` mechanism (RANGE vgap only).

As neurodamus, RANGE-capable mechs use ``source_var``/``target_var``
transfers in serial too — wiring happens under a one-rank
``ParallelContext`` (``sim.pc`` is not None after ``instantiate_gids``).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from bluecellulab import CircuitSimulation
from bluecellulab.circuit import CellId, SonataCircuitAccess
from bluecellulab.circuit.gap_junction_manager import GapJunctionManager


parent_dir = Path(__file__).resolve().parent.parent
HIP_CIRCUIT_DIR = (
    parent_dir / "examples" / "circuit_hipp_mooc_most_central_10_SP_PC"
).resolve()
PROJ_SIM_DIR = (
    parent_dir / "examples" / "sonata_unit_test_sims" / "projections"
).resolve()
ELECTRICAL_POP = "hippocampus_neurons__hippocampus_neurons__electrical"
CHEMICAL_POP = "hippocampus_neurons__hippocampus_neurons__chemical"

# Directed edges, file order sorted by source: reciprocal pairs 1<->2 and
# 1<->3. conductances differ per edge so the wiring can be checked
# one-to-one.
_GJ_EDGES = [  # (src, tgt, conductance)
    (1, 2, 0.5),
    (1, 3, 0.7),
    (2, 1, 0.9),
    (3, 1, 1.1),
]
_N_NODES = 10  # hippocampus_neurons has ids 0..9


def _write_electrical_edges(path: Path) -> None:
    """Write a minimal SONATA edges.h5 with one ``electrical`` population."""
    src = np.array([e[0] for e in _GJ_EDGES], dtype=np.uint64)
    tgt = np.array([e[1] for e in _GJ_EDGES], dtype=np.uint64)
    cond = np.array([e[2] for e in _GJ_EDGES], dtype=np.float32)
    n = len(src)

    # Junction ids per unordered pair: lower node id gets the even id.
    pair_base = {(1, 2): 0, (1, 3): 2}
    eff, aff = [], []
    for s, t, _ in _GJ_EDGES:
        base = pair_base[(min(s, t), max(s, t))]
        eff.append(base + (0 if s <= t else 1))
        aff.append(base + (1 if s <= t else 0))

    # File order is already source-sorted: contiguous source ranges.
    src_ranges = [(1, (0, 2)), (2, (2, 3)), (3, (3, 4))]
    # Target order: tgt=1 -> edges 2,3 ; tgt=2 -> edge 0 ; tgt=3 -> edge 1.
    tgt_ranges = [(1, (2, 4)), (2, (0, 1)), (3, (1, 2))]

    with h5py.File(path, "w") as f:
        f.create_group("edge_types").create_group("0")
        pop = f.create_group(f"edges/{ELECTRICAL_POP}")
        src_ds = pop.create_dataset("source_node_id", data=src)
        src_ds.attrs["node_population"] = "hippocampus_neurons"
        tgt_ds = pop.create_dataset("target_node_id", data=tgt)
        tgt_ds.attrs["node_population"] = "hippocampus_neurons"
        pop.create_dataset("edge_type_id", data=np.zeros(n, dtype=np.int32))
        pop.create_dataset("edge_group_id", data=np.zeros(n, dtype=np.uint32))
        pop.create_dataset("edge_group_index", data=np.arange(n, dtype=np.uint64))

        g0 = pop.create_group("0")
        g0.create_dataset("conductance", data=cond)
        for name in ("afferent_section_id", "efferent_section_id"):
            g0.create_dataset(name, data=np.zeros(n, dtype=np.int32))
        for name in ("afferent_section_pos", "efferent_section_pos"):
            g0.create_dataset(name, data=np.full(n, 0.5, dtype=np.float32))
        g0.create_dataset("efferent_junction_id",
                          data=np.asarray(eff, dtype=np.int64))
        g0.create_dataset("afferent_junction_id",
                          data=np.asarray(aff, dtype=np.int64))

        for idx_name, ranges in (("source_to_target", src_ranges),
                                 ("target_to_source", tgt_ranges)):
            n2r = np.zeros((_N_NODES, 2), dtype=np.uint64)
            r2e = np.zeros((len(ranges), 2), dtype=np.uint64)
            for i, (nid, span) in enumerate(ranges):
                n2r[nid] = (i, i + 1)
                r2e[i] = span
            idx = pop.create_group(f"indices/{idx_name}")
            idx.create_dataset("node_id_to_ranges", data=n2r)
            idx.create_dataset("range_to_edge_id", data=r2e)


def _write_circuit_config(tmp_path: Path) -> Path:
    """circuit_config grafting the synthetic electrical population onto the
    hippocampus unit circuit (via the projections sim's config, whose
    components paths resolve correctly)."""
    cfg = json.loads((PROJ_SIM_DIR / "circuit_config.json").read_text())
    cfg["manifest"] = {
        "$NETWORK_DIR": str(HIP_CIRCUIT_DIR),
        "$COMPONENT_DIR": str(HIP_CIRCUIT_DIR / "components"),
        "$GJ_DIR": str(tmp_path),
    }
    cfg["node_sets_file"] = "$NETWORK_DIR/node_sets.json"
    cfg["networks"]["edges"].append(
        {
            "edges_file": "$GJ_DIR/gj_edges.h5",
            "edge_types_file": None,
            "populations": {ELECTRICAL_POP: {"type": "electrical"}},
        }
    )
    out = tmp_path / "circuit_config.json"
    out.write_text(json.dumps(cfg))
    return out


def _write_sim_config(tmp_path: Path, extra: dict) -> Path:
    cfg = json.loads(
        (PROJ_SIM_DIR / "simulation_config.json").read_text()
    )
    cfg["network"] = str(tmp_path / "circuit_config.json")
    cfg["node_sets_file"] = str(PROJ_SIM_DIR / "node_sets.json")
    cfg["output"] = dict(cfg.get("output", {}), output_dir=str(tmp_path / "out"))
    cfg.update(extra)
    out = tmp_path / "simulation_config.json"
    out.write_text(json.dumps(cfg))
    return out


@pytest.fixture()
def gj_circuit(tmp_path):
    _write_electrical_edges(tmp_path / "gj_edges.h5")
    _write_circuit_config(tmp_path)
    return tmp_path


def test_electrical_population_auto_detected(gj_circuit):
    """``type: "electrical"`` populations are discovered without config."""
    access = SonataCircuitAccess(
        _write_sim_config(gj_circuit, {})
    )
    assert access.electrical_edge_populations() == [ELECTRICAL_POP]


def test_gap_junctions_wired_per_directed_edge(gj_circuit):
    """One mech on the *target* per directed edge; g = conductance."""
    sim_cfg = _write_sim_config(
        gj_circuit,
        {
            "gap_junctions": [
                {
                    "edge_population": ELECTRICAL_POP,
                    "mod": "gapdual",
                    "pointer_name": "vpre",
                }
            ]
        },
    )
    sim = CircuitSimulation(str(sim_cfg))
    sim.instantiate_gids(
        [CellId("hippocampus_neurons", i) for i in (1, 2, 3)],
        add_synapses=False,
        add_stimuli=False,
    )
    c1 = sim.cells[CellId("hippocampus_neurons", 1)]
    c2 = sim.cells[CellId("hippocampus_neurons", 2)]
    c3 = sim.cells[CellId("hippocampus_neurons", 3)]

    # Cell 1 is target of edges 2->1 (0.9) and 3->1 (1.1).
    assert sorted(g.g for g in c1.gap_junctions) == [
        pytest.approx(0.9), pytest.approx(1.1)]
    assert [g.g for g in c2.gap_junctions] == [pytest.approx(0.5)]
    assert [g.g for g in c3.gap_junctions] == [pytest.approx(0.7)]

    # Dual-mode mechs use pc transfers in serial too (neurodamus parity):
    # the RANGE transfer var is armed and seeded to v_init, and the
    # serial POINTER is bound to the source v only as a safety net.
    assert sim.pc is not None
    for gj in c2.gap_junctions:
        assert gj.use_vgap == 1
        assert gj.vgap == pytest.approx(
            sim.circuit_access.config.v_init)
        assert gj.vpre == pytest.approx(c1.soma(0.5).v)


def test_gap_junctions_pure_gap_mech_serial(gj_circuit):
    """Neurodamus-style RANGE-only ``Gap`` mechs wire under transfers."""
    sim_cfg = _write_sim_config(
        gj_circuit,
        {
            "gap_junctions": [
                {
                    "edge_population": ELECTRICAL_POP,
                    "mod": "Gap",
                    "pointer_name": "vgap",
                }
            ]
        },
    )
    sim = CircuitSimulation(str(sim_cfg))
    sim.instantiate_gids(
        [CellId("hippocampus_neurons", i) for i in (1, 2, 3)],
        add_synapses=False,
        add_stimuli=False,
    )
    c1 = sim.cells[CellId("hippocampus_neurons", 1)]
    c2 = sim.cells[CellId("hippocampus_neurons", 2)]

    assert sim.pc is not None
    assert sorted(g.g for g in c1.gap_junctions) == [
        pytest.approx(0.9), pytest.approx(1.1)]
    assert [g.g for g in c2.gap_junctions] == [pytest.approx(0.5)]
    # Pure RANGE mech: no POINTER, no use_vgap selector — coupling flows
    # entirely through source_var/target_var.
    for gj in c2.gap_junctions:
        assert not hasattr(gj, "use_vgap")
        assert not hasattr(gj, "_ref_vpre")


def test_electrical_population_excluded_from_synapse_path(gj_circuit):
    """Electrical edges are wired by the GJ manager, not as synapses."""
    access = SonataCircuitAccess(_write_sim_config(gj_circuit, {}))
    assert ELECTRICAL_POP in access._circuit.edges
    assert ELECTRICAL_POP not in access._select_edge_pop_names(None)
    assert ELECTRICAL_POP not in access._select_edge_pop_names(True)


def test_continuous_connections_wired_on_chemical_population(gj_circuit):
    """Graded POINTER connections ride the chemical edge population."""
    sim_cfg = _write_sim_config(
        gj_circuit,
        {
            "continuous_connections": [
                {
                    "name": "graded_chem",
                    "edge_population": CHEMICAL_POP,
                    "mod": "gapdual",
                    "pointer_name": "vpre",
                }
            ]
        },
    )
    sim = CircuitSimulation(str(sim_cfg))
    sim.instantiate_gids(
        [CellId("hippocampus_neurons", 9), CellId("hippocampus_neurons", 2)],
        add_synapses=False,
        add_stimuli=False,
    )
    c2 = sim.cells[CellId("hippocampus_neurons", 2)]
    # The chemical population carries three 9 -> 2 edges; dual-mode
    # mechs transfer via vgap (serial, neurodamus-style).
    assert sim.pc is not None
    assert len(c2.continuous_synapses) == 3
    for syn in c2.continuous_synapses:
        assert syn.use_vgap == 1
        assert syn.vgap == pytest.approx(
            sim.circuit_access.config.v_init)
        assert syn.vpre == pytest.approx(
            sim.cells[CellId("hippocampus_neurons", 9)].soma(0.5).v
        )


def test_junction_id_synthesis_without_columns():
    """Fallback: ids synthesized per unordered pair when columns absent."""
    synthesized: dict = {}
    eff, aff = GapJunctionManager._junction_ids(1, 2, {}, synthesized)
    assert (eff, aff) == (0, 1)
    eff, aff = GapJunctionManager._junction_ids(2, 1, {}, synthesized)
    assert (eff, aff) == (1, 0)  # reciprocal cross-matches
    eff, aff = GapJunctionManager._junction_ids(1, 3, {}, synthesized)
    assert (eff, aff) == (2, 3)

    # Columns present -> used verbatim.
    props = {"efferent_junction_id": 42, "afferent_junction_id": 7}
    assert GapJunctionManager._junction_ids(5, 9, props, {}) == (42, 7)
