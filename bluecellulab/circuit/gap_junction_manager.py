# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Gap junction & continuous (graded) connection management.

Implements the neurodamus-style gap-junction wiring using NEURON's
``ParallelContext.source_var`` / ``target_var`` for MPI-correct continuous
variable transfer. The same path is reused for graded chemical synapses
(``ContinuousConnectionConfig``); they differ only in symmetry.

The mod mechanism for each block must define:
- a ``POINTER`` named per ``GapJunctionConfig.pointer_name`` (default ``vgap``)
  which receives the presynaptic voltage,
- a ``RANGE g`` (or equivalent conductance variable) settable from the edge
  ``conductance`` column.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import neuron

from bluecellulab.circuit.config.gap_junctions import (
    ContinuousConnectionConfig,
    GapJunctionConfig,
)

if TYPE_CHECKING:
    from bluecellulab.cell import Cell

logger = logging.getLogger(__name__)


class _PointerConnectionManager:
    """Shared base for gap-junctions and continuous (graded) connections.

    Both drive `pc.source_var` (broadcast a continuous variable from the
    pre-cell) and `pc.target_var` (subscribe a POINTER on the post-cell).
    """

    def __init__(self, sim: Any):
        self._sim = sim
        self._pc = sim.pc  # may be None for serial; we'll handle locally
        # Pool of stable integer IDs used for source_var/target_var pairing.
        # ID = (population_offset << 20) | edge_index, kept unique per
        # population.
        self._next_var_id_per_pop: dict[str, int] = {}
        # Track (cell, var_id) pairs already registered as sources to
        # avoid double-registration with NEURON's ParallelContext.
        self._source_var_keys: set = set()

    # ---- helpers ---------------------------------------------------------

    def _alloc_var_id(self, edge_population: str) -> int:
        cur = self._next_var_id_per_pop.get(edge_population, 0)
        self._next_var_id_per_pop[edge_population] = cur + 1
        # Keep a generous offset per population to avoid id collisions.
        pop_idx = list(self._next_var_id_per_pop.keys()).index(edge_population)
        return (pop_idx << 24) | cur

    def _resolve_node_set(self, node_set_name: str) -> set[int]:
        """Resolve a node-set name to the set of node-ids contained in it.

        Returns an empty set for "All" — interpreted as no filter.
        """
        if node_set_name in (None, "All", "all"):
            return set()
        node_sets = self._sim.circuit_access.config.get_node_sets()
        entry = node_sets.get(node_set_name)
        if entry is None:
            raise ValueError(f"Unknown node_set '{node_set_name}'.")
        ids = entry.get("node_id") or entry.get("node_ids") or []
        return set(int(x) for x in ids)

    def _iter_edges(
        self,
        edge_population: str,
        src_filter: set[int],
        tgt_filter: set[int],
    ):
        """Yield (src_node_id, tgt_node_id, edge_index, props) for matching
        edges.

        Uses bluepysnap to access edges; reads conductance &
        afferent_section_id/pos when present. Filters by node-set
        membership.
        """
        access = self._sim.circuit_access
        edges = access._circuit.edges  # internal bluepysnap circuit
        if edge_population not in edges:
            raise ValueError(
                f"Edge population '{edge_population}' not found in circuit. "
                f"Available: {list(edges.keys())}"
            )
        epop = edges[edge_population]

        # Bluepysnap: epop.size, epop.get(properties=...)
        n = epop.size
        if n == 0:
            return

        wanted_props = []
        for p in ("conductance", "afferent_section_id", "afferent_section_pos",
                  "efferent_junction_id", "afferent_junction_id"):
            if p in epop.property_names:
                wanted_props.append(p)

        # bluepysnap requires explicit edge ids; pass the full id range.
        edge_ids = epop.ids()
        df = epop.get(edge_ids,
                      properties=["@source_node", "@target_node"] + wanted_props)
        # df columns may include the '@' prefix; rename for safe attribute access
        df = df.rename(columns={"@source_node": "source_node",
                                "@target_node": "target_node"})
        for idx, row in enumerate(df.itertuples(index=False)):
            src = int(row.source_node)
            tgt = int(row.target_node)
            if src_filter and src not in src_filter:
                continue
            if tgt_filter and tgt not in tgt_filter:
                continue
            props = {p: getattr(row, p) for p in wanted_props}
            yield src, tgt, idx, props

    def _resolve_target_section(self, cell: "Cell", props: dict):
        """Resolve target section + x location from edge properties."""
        sec_id = int(props.get("afferent_section_id", 0))
        pos = float(props.get("afferent_section_pos", 0.5))
        if pos <= 0.0:
            pos = 0.0001
        if pos >= 1.0:
            pos = 0.9999
        try:
            section = cell.get_psection(section_id=sec_id).hsection
        except Exception:
            section = cell.soma
        return section, pos


class GapJunctionManager(_PointerConnectionManager):
    """Adds gap junctions for one ``GapJunctionConfig`` block."""

    def add_block(self, cfg: GapJunctionConfig) -> int:
        node_set = self._resolve_node_set(cfg.node_set)
        n_added = 0
        seen_pairs: set[frozenset] = set()

        local_cells = self._sim.cells  # CellDict keyed by CellId

        # Build a fast (population, node_id) -> Cell map
        local_node_ids = {cid.id: cell for cid, cell in local_cells.items()}

        for src, tgt, idx, props in self._iter_edges(
            cfg.edge_population, node_set, node_set
        ):
            # Only wire pairs where both endpoints exist locally; otherwise
            # pc.setup_transfer() will see an unmatched target_var and segfault.
            if src not in local_node_ids or tgt not in local_node_ids:
                continue
            # symmetric -> dedupe unordered pair
            if cfg.symmetric:
                key = frozenset((src, tgt))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

            cond = float(props.get("conductance", 1.0)) * cfg.weight_factor

            # IDs for the two pointer-transfer channels
            aff_id = int(props.get(
                "afferent_junction_id",
                self._alloc_var_id(cfg.edge_population),
            ))
            eff_id = int(props.get(
                "efferent_junction_id",
                self._alloc_var_id(cfg.edge_population),
            ))

            n_added += self._wire_gap_pair(
                cfg, src, tgt, cond, aff_id, eff_id,
                props, local_node_ids,
            )

        logger.info(
            "GapJunctionManager: added %d gap junction(s) from edges='%s' "
            "(node_set=%s, mod=%s, pointer=%s, symmetric=%s)",
            n_added, cfg.edge_population, cfg.node_set, cfg.mod,
            cfg.pointer_name, cfg.symmetric,
        )
        return n_added

    def _wire_gap_pair(
        self, cfg: GapJunctionConfig, src: int, tgt: int, conductance: float,
        aff_id: int, eff_id: int, props: dict, local_node_ids: dict,
    ) -> int:
        """Wire a (potentially symmetric) gap-junction pair.

        Single-process direct pointer wiring via ``neuron.h.setpointer``.
        For MPI-correct cross-rank wiring, the pc.source_var/target_var
        path remains available via :py:meth:`_bind_pointer` but is not
        currently exercised here.
        """
        added = 0

        if tgt not in local_node_ids or (
            cfg.symmetric and src not in local_node_ids
        ):
            return 0

        # Target endpoint: receives src soma v
        tgt_cell = local_node_ids[tgt]
        tgt_section, tgt_x = self._resolve_target_section(tgt_cell, props)
        tgt_gj = self._build_pointer_mech(cfg, tgt_section, tgt_x, conductance)
        if src in local_node_ids:
            src_section = local_node_ids[src].soma
            neuron.h.setpointer(src_section(0.5)._ref_v, cfg.pointer_name, tgt_gj)
        tgt_cell.gap_junctions = getattr(tgt_cell, "gap_junctions", [])
        tgt_cell.gap_junctions.append(tgt_gj)
        added += 1

        # Source endpoint (symmetric only): receives tgt soma v
        if cfg.symmetric and src in local_node_ids:
            src_cell = local_node_ids[src]
            sec, x = self._resolve_target_section(src_cell, props)
            src_gj = self._build_pointer_mech(cfg, sec, x, conductance)
            tgt_section_v = tgt_cell.soma
            neuron.h.setpointer(tgt_section_v(0.5)._ref_v, cfg.pointer_name, src_gj)
            src_cell.gap_junctions = getattr(src_cell, "gap_junctions", [])
            src_cell.gap_junctions.append(src_gj)
            added += 1

        return added

    def _build_pointer_mech(self, cfg: GapJunctionConfig, section, x: float,
                            conductance: float):
        mech_cls = getattr(neuron.h, cfg.mod, None)
        if mech_cls is None:
            raise AttributeError(
                f"NEURON mechanism '{cfg.mod}' not found. Compile the .mod "
                "file (nrnivmodl) and ensure it is loaded."
            )
        gj = mech_cls(x, sec=section)
        # Conductance: try common names (g, weight, conductance) in order.
        for name in ("g", "weight", "conductance"):
            if hasattr(gj, name):
                setattr(gj, name, conductance)
                break
        return gj

    def _bind_pointer(self, gj, pointer_name: str, section, x: float,
                      receive_id: int, broadcast_id: int):
        """Wire gj's POINTER to a remote v via target_var/source_var.

        receive_id : id this cell subscribes to (matches broadcast_id from peer).
        broadcast_id : id this cell broadcasts its v under.
        """
        ptr_attr = f"_ref_{pointer_name}"
        if not hasattr(gj, ptr_attr):
            raise AttributeError(
                f"Mechanism '{type(gj).__name__}' has no POINTER named "
                f"'{pointer_name}'."
            )
        if self._pc is not None:
            with _section_in_stack(section):
                self._pc.target_var(gj, getattr(gj, ptr_attr), receive_id)
                self._pc.source_var(section(x)._ref_v, broadcast_id)
        else:
            # Serial / no ParallelContext: direct setpointer to local v.
            # This only works when source endpoint is on the same rank;
            # for cross-rank coupling pc.target_var/source_var is mandatory.
            neuron.h.setpointer(section(x)._ref_v, pointer_name, gj)


class ContinuousConnectionManager(_PointerConnectionManager):
    """Adds continuous (graded) chemical-synapse connections for one block."""

    def add_block(self, cfg: ContinuousConnectionConfig) -> int:
        src_filter = self._resolve_node_set(cfg.source)
        tgt_filter = self._resolve_node_set(cfg.target)
        n_added = 0

        local_cells = self._sim.cells
        local_node_ids = {cid.id: cell for cid, cell in local_cells.items()}

        for src, tgt, idx, props in self._iter_edges(
            cfg.edge_population, src_filter, tgt_filter
        ):
            # Only wire pairs where both endpoints exist locally.
            if src not in local_node_ids or tgt not in local_node_ids:
                continue
            cond = float(props.get("conductance", 1.0)) * cfg.weight_factor

            cell = local_node_ids[tgt]
            section, x = self._resolve_target_section(cell, props)
            syn = self._build_pointer_mech(cfg, section, x, cond)

            # Single-process direct pointer wiring (works for serial; for
            # MPI we'd switch to pc.source_var/target_var with a single
            # broadcast id per source cell).
            src_section = local_node_ids[src].soma
            ptr_attr = f"_ref_{cfg.pointer_name}"
            if not hasattr(syn, ptr_attr):
                raise AttributeError(
                    f"Mechanism '{cfg.mod}' has no POINTER named "
                    f"'{cfg.pointer_name}'."
                )
            neuron.h.setpointer(src_section(0.5)._ref_v, cfg.pointer_name, syn)

            cell.continuous_synapses = getattr(cell, "continuous_synapses", [])
            cell.continuous_synapses.append(syn)
            n_added += 1

        logger.info(
            "ContinuousConnectionManager '%s': added %d connection(s) from "
            "edges='%s' (mod=%s, pointer=%s)",
            cfg.name, n_added, cfg.edge_population, cfg.mod, cfg.pointer_name,
        )
        return n_added

    def _build_pointer_mech(self, cfg: ContinuousConnectionConfig, section,
                            x: float, conductance: float):
        mech_cls = getattr(neuron.h, cfg.mod, None)
        if mech_cls is None:
            raise AttributeError(
                f"NEURON mechanism '{cfg.mod}' not found. Compile the .mod "
                "file (nrnivmodl) and ensure it is loaded."
            )
        syn = mech_cls(x, sec=section)
        for name in ("conductance", "weight", "g"):
            if hasattr(syn, name):
                setattr(syn, name, conductance)
                break
        return syn

    def _bind_pointer(self, syn, pointer_name: str, section, x: float,
                      receive_id: int, broadcast_id: int,
                      broadcast_section):
        ptr_attr = f"_ref_{pointer_name}"
        if not hasattr(syn, ptr_attr):
            raise AttributeError(
                f"Mechanism '{type(syn).__name__}' has no POINTER named "
                f"'{pointer_name}'."
            )
        if self._pc is not None:
            with _section_in_stack(section):
                self._pc.target_var(syn, getattr(syn, ptr_attr), receive_id)


class _section_in_stack:
    """Context manager: push a NEURON section onto the section stack.

    Several NEURON ParallelContext APIs require the currently-accessed
    section to match the section the variable belongs to.
    """

    def __init__(self, section):
        self._section = section

    def __enter__(self):
        self._section.push()
        return self._section

    def __exit__(self, exc_type, exc, tb):
        neuron.h.pop_section()
        return False
