# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Gap junction & continuous (graded) connection management.

Gap junctions follow the neurodamus model (``neurodamus/gap_junction.py``
and ``Connection.finalize_gap_junctions``):

- every edge population with ``"type": "electrical"`` is wired; one point
  process per *directed* edge is placed on the target cell at the afferent
  location. Symmetric coupling is a property of the edge file (reciprocal
  rows), not a flag.
- conductance comes from the ``conductance`` edge property (neurodamus maps
  it to ``weight``) and is assigned to the mechanism's ``g``/``weight``/
  ``conductance`` RANGE variable, whichever it defines.
- under MPI, ``pc.target_var`` subscribes the junction's peer-voltage
  variable to the ``efferent_junction_id`` broadcast and ``pc.source_var``
  exports the target's voltage under ``afferent_junction_id`` — junction
  ids come from the edge properties when present, and are synthesized per
  unordered cell pair otherwise. Transfer targets must be ``RANGE``
  variables (NEURON cannot transfer into a ``POINTER``): mechanisms that
  expose the peer voltage only as a ``POINTER`` (serial ``setpointer``
  style) are wired via a ``RANGE vgap`` transfer variable + ``use_vgap``
  selector switch when those members exist (neurodamus ``Gap``
  convention). Single-rank wiring binds the pointer directly with
  ``neuron.h.setpointer`` (``pc.source_var`` under multiple ids on the
  same ``_ref_v`` is fragile in NEURON).
- ``continuous_connections`` reuse the same pointer-transfer path for
  graded chemical synapses (directional source -> target), a BlueCelluLab
  extension with no neurodamus counterpart. Under MPI each source cell
  exports its soma voltage once per population under a broadcast id of
  ``1e9 + global_gid`` (kept clear of the junction-id space).

The mod mechanism for each block must define:
- a peer-voltage variable named per ``GapJunctionConfig.pointer_name``
  (default ``vgap``) — a ``POINTER`` for serial ``setpointer`` wiring, or
  a ``RANGE`` variable for MPI ``target_var`` wiring (dual-mode mechs
  provide both: ``POINTER vpre`` + ``RANGE vgap``/``use_vgap``),
- a ``RANGE`` conductance variable (``g``, ``weight`` or ``conductance``)
  settable from the edge ``conductance`` column.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import neuron

from bluecellulab.circuit.config.gap_junctions import (
    ContinuousConnectionConfig,
    GapJunctionConfig,
)
from bluecellulab.exceptions import BluecellulabError

if TYPE_CHECKING:
    from bluecellulab.cell import Cell

logger = logging.getLogger(__name__)


class _PointerConnectionManager:
    """Shared base for gap-junctions and continuous (graded) connections.

    Both drive `pc.source_var` (broadcast a continuous variable from the
    pre-cell) and `pc.target_var` (subscribe a POINTER on the post-cell).
    """

    #: Base of the broadcast-id space used for continuous connections under
    #: MPI (per source cell). Kept far above typical junction-id ranges.
    _MPI_TRANSFER_BASE = 1_000_000_000

    def __init__(self, sim: Any):
        self._sim = sim
        self._pc = sim.pc  # may be None for serial; we'll handle locally

    # ---- helpers ---------------------------------------------------------

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
                  "efferent_section_id", "efferent_section_pos",
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
        return self._resolve_section(
            cell,
            props.get("afferent_section_id", 0),
            props.get("afferent_section_pos", 0.5),
        )

    def _resolve_source_section(self, cell: "Cell", props: dict):
        """Resolve source (efferent) section + x location, soma fallback."""
        return self._resolve_section(
            cell,
            props.get("efferent_section_id", 0),
            props.get("efferent_section_pos", 0.5),
        )

    @staticmethod
    def _resolve_section(cell: "Cell", sec_id, pos):
        sec_id = int(sec_id)
        pos = float(pos)
        if pos <= 0.0:
            pos = 0.0001
        if pos >= 1.0:
            pos = 0.9999
        try:
            section = cell.get_psection(section_id=sec_id).hsection
        except Exception:
            section = cell.soma
        return section, pos

    def _local_and_instantiated_node_ids(self) -> tuple[dict, set]:
        """(local node-id -> Cell, ids instantiated on any rank).

        Under MPI the global set is the gathered
        ``_instantiated_cells_mpi``; serially it equals the local set.
        """
        local_node_ids = {cid.id: cell for cid, cell in self._sim.cells.items()}
        instantiated = getattr(self._sim, "_instantiated_cells_mpi", None)
        all_node_ids = (
            {cid.id for cid in instantiated}
            if instantiated is not None
            else set(local_node_ids)
        )
        return local_node_ids, all_node_ids

    @staticmethod
    def _transfer_var_name(pp, pointer_name: str) -> str | None:
        """Return the RANGE variable usable as a transfer target, or None.

        ``pc.target_var`` can only write into ``RANGE`` variables — a
        ``POINTER`` member can never be a transfer target, and probing it
        under MPI is fatal (``hoc_execerror`` aborts MPI, not a catchable
        exception). The decision is convention-based:

        - ``vgap`` when the mech exposes it (neurodamus ``Gap``
          convention — also the dual-mode transfer var);
        - else ``pointer_name`` itself when it names a RANGE variable
          (reading an unbound POINTER raises ``ValueError``, so
          ``hasattr`` distinguishes the two);
        - else ``None`` — the mech's peer-voltage input is POINTER-only.
        """
        if hasattr(pp, "vgap"):
            return "vgap"
        if hasattr(pp, pointer_name):
            return pointer_name
        return None

    def _target_var(self, pp, var_name: str, src_id: int):
        """Subscribe ``pp``'s RANGE ``var_name`` to broadcast ``src_id``."""
        ref = getattr(pp, f"_ref_{var_name}", None)
        if ref is None:
            raise AttributeError(
                f"Mechanism '{type(pp).__name__}' has no variable named "
                f"'{var_name}'."
            )
        self._pc.target_var(pp, ref, src_id)
        if hasattr(pp, "use_vgap"):
            pp.use_vgap = 1


class GapJunctionManager(_PointerConnectionManager):
    """Wires one ``"type": "electrical"`` edge population as gap junctions.

    Follows neurodamus ``Connection.finalize_gap_junctions``: each directed
    edge creates one junction mechanism on the *target* cell; reciprocal
    edges in the file produce the symmetric coupling. ``cfg`` carries only
    the per-population overrides (mechanism / POINTER names, conductance
    scaling).
    """

    def add_block(self, cfg: GapJunctionConfig) -> int:
        n_added = 0
        # (min,max) node pair -> base junction id, used only when the edge
        # file does not carry afferent/efferent_junction_id columns.
        synthesized_pair_ids: dict[tuple[int, int], int] = {}

        local_node_ids, all_node_ids = self._local_and_instantiated_node_ids()

        for src, tgt, idx, props in self._iter_edges(
            cfg.edge_population, set(), set()
        ):
            # The junction lives on the rank owning the *target*; the
            # source's broadcast is registered by the reciprocal edge on
            # whichever rank owns the source. Only require that the source
            # is instantiated somewhere, so no target_var is left
            # unmatched for pc.setup_transfer().
            if tgt not in local_node_ids or src not in all_node_ids:
                continue

            conductance = float(props.get("conductance", 1.0)) * cfg.weight_factor
            eff_id, aff_id = self._junction_ids(src, tgt, props, synthesized_pair_ids)
            n_added += self._wire_gap_edge(
                cfg, src, tgt, conductance, eff_id, aff_id, props, local_node_ids,
            )

        logger.info(
            "GapJunctionManager: added %d gap junction(s) from edges='%s' "
            "(mod=%s, pointer=%s, weight_factor=%s)",
            n_added, cfg.edge_population, cfg.mod, cfg.pointer_name,
            cfg.weight_factor,
        )
        return n_added

    @staticmethod
    def _junction_ids(
        src: int, tgt: int, props: dict,
        synthesized_pair_ids: dict[tuple[int, int], int],
    ) -> tuple[int, int]:
        """Return (efferent_junction_id, afferent_junction_id) for an edge.

        With junction-id columns present (neurodamus/SONATA convention) they
        are used verbatim: the ``efferent_junction_id`` of edge s->t is the
        id under which s broadcasts its v (subscribed by the target's
        POINTER), and ``afferent_junction_id`` is the id under which the
        target broadcasts (subscribed by the reciprocal edge's mech).

        Without them, ids are synthesized per unordered cell pair: the
        lower node id broadcasts under ``2*pair_idx``, the higher under
        ``2*pair_idx + 1`` — reciprocal edges cross-match automatically.
        Only the MPI path consumes these; serial wiring uses setpointer.
        """
        if "efferent_junction_id" in props and "afferent_junction_id" in props:
            return int(props["efferent_junction_id"]), int(props["afferent_junction_id"])

        key = (min(src, tgt), max(src, tgt))
        base = synthesized_pair_ids.setdefault(key, 2 * len(synthesized_pair_ids))
        src_id = base + (0 if src == key[0] else 1)
        tgt_id = base + (1 if src == key[0] else 0)
        return src_id, tgt_id

    def _wire_gap_edge(
        self, cfg: GapJunctionConfig, src: int, tgt: int, conductance: float,
        eff_id: int, aff_id: int, props: dict, local_node_ids: dict,
    ) -> int:
        """Create one junction mech on the target cell for edge src -> tgt."""
        tgt_cell = local_node_ids[tgt]
        tgt_section, tgt_x = self._resolve_target_section(tgt_cell, props)
        gj = self._build_pointer_mech(cfg, tgt_section, tgt_x, conductance)

        # The source cell object is only needed for the serial setpointer
        # path; under MPI it may live on another rank (its broadcast is
        # matched by junction id).
        src_cell = local_node_ids.get(src)
        src_section, src_x = (
            self._resolve_source_section(src_cell, props)
            if src_cell is not None
            else (None, None)
        )
        self._bind_pointer(
            gj, cfg.pointer_name,
            tgt_section=tgt_section, tgt_x=tgt_x,
            eff_id=eff_id, aff_id=aff_id,
            src_section=src_section, src_x=src_x,
        )

        if (
            self._pc is not None
            and src == tgt
            and src_section is not None
            and self._transfer_var_name(gj, cfg.pointer_name) is not None
        ):
            # Self-junction: a single directed edge exists, so no
            # reciprocal edge ever exports under eff_id. The partner
            # side is this same cell — export its efferent-location
            # voltage ourselves.
            with _section_in_stack(src_section):
                self._pc.source_var(src_section(src_x)._ref_v, eff_id)

        tgt_cell.gap_junctions = getattr(tgt_cell, "gap_junctions", [])
        tgt_cell.gap_junctions.append(gj)
        return 1

    def _build_pointer_mech(self, cfg: GapJunctionConfig, section, x: float,
                            conductance: float):
        mech_cls = getattr(neuron.h, cfg.mod, None)
        if mech_cls is None:
            raise AttributeError(
                f"NEURON mechanism '{cfg.mod}' not found. Compile the .mod "
                "file (nrnivmodl) and ensure it is loaded."
            )
        gj = mech_cls(x, sec=section)
        # Conductance: neurodamus assigns `g` from the `conductance` column;
        # accept `weight`/`conductance` RANGE names for non-standard mods.
        for name in ("g", "weight", "conductance"):
            if hasattr(gj, name):
                setattr(gj, name, conductance)
                break
        return gj

    def _bind_pointer(self, gj, pointer_name: str, tgt_section, tgt_x: float,
                      eff_id: int, aff_id: int, src_section, src_x) -> None:
        """Bind gj's peer-voltage input to the source (neurodamus conventions).

        RANGE-capable mechs (``_transfer_var_name`` — neurodamus ``Gap``
        or dual-mode ``vgap``/``use_vgap`` mechs) always use variable
        transfers, serial or MPI alike: ``target_var`` subscribes the
        RANGE var to the source broadcast under ``eff_id`` and
        ``source_var`` exports the target's voltage under ``aff_id``
        (consumed by the reciprocal edge's junction). Dual-mode mechs
        additionally get their ``POINTER`` bound to a local voltage so
        it never dangles.

        POINTER-only mechs fall back to a direct serial ``setpointer``;
        under MPI they are rejected — NEURON cannot transfer into a
        POINTER.
        """
        has_pointer = hasattr(gj, f"_ref_{pointer_name}")
        transfer_var = self._transfer_var_name(gj, pointer_name)
        if self._pc is not None and transfer_var is not None:
            if hasattr(gj, "use_vgap"):
                # Dual-mode mech: bind the serial POINTER to a local
                # voltage so it is never dangling (the RANGE transfer
                # var drives the current anyway).
                bind_sec, bind_x = (
                    (src_section, src_x)
                    if src_section is not None
                    else (tgt_section, tgt_x)
                )
                neuron.h.setpointer(
                    bind_sec(bind_x)._ref_v, pointer_name, gj
                )
            with _section_in_stack(tgt_section):
                self._target_var(gj, transfer_var, eff_id)
                self._pc.source_var(tgt_section(tgt_x)._ref_v, aff_id)
            # Seed the transfer var with the resting voltage so a
            # mechanism INITIAL that reads it before the first transfer
            # sees a sane value.
            if getattr(gj, "use_vgap", 0):
                gj.vgap = self._sim.circuit_access.config.v_init
        elif has_pointer:
            if self._pc is not None and int(self._pc.nhost()) > 1:
                raise BluecellulabError(
                    f"Gap-junction mechanism '{type(gj).__name__}' only "
                    f"exposes POINTER '{pointer_name}': NEURON transfers "
                    "can only write into RANGE variables under MPI. "
                    "Provide a RANGE transfer variable (e.g. 'vgap') — "
                    "see the dual-mode convention."
                )
            assert src_section is not None
            neuron.h.setpointer(src_section(src_x)._ref_v, pointer_name, gj)
        else:
            raise AttributeError(
                f"Mechanism '{type(gj).__name__}' has neither a POINTER "
                f"named '{pointer_name}' nor a RANGE transfer variable "
                "('vgap' or a RANGE named by pointer_name)."
            )


class ContinuousConnectionManager(_PointerConnectionManager):
    """Adds continuous (graded) chemical-synapse connections for one block."""

    def add_block(self, cfg: ContinuousConnectionConfig) -> int:
        src_filter = self._resolve_node_set(cfg.source)
        tgt_filter = self._resolve_node_set(cfg.target)
        n_added = 0

        local_node_ids, all_node_ids = self._local_and_instantiated_node_ids()
        broadcasted_srcs: set[int] = set()

        for src, tgt, idx, props in self._iter_edges(
            cfg.edge_population, src_filter, tgt_filter
        ):
            # Under MPI each source cell exports its soma voltage once, on
            # the rank where it lives.
            if (
                self._pc is not None
                and src in local_node_ids
                and src not in broadcasted_srcs
            ):
                src_cell = local_node_ids[src]
                with _section_in_stack(src_cell.soma):
                    self._pc.source_var(
                        src_cell.soma(0.5)._ref_v, self._broadcast_id(src, cfg)
                    )
                broadcasted_srcs.add(src)

            if tgt not in local_node_ids or src not in all_node_ids:
                continue
            cond = float(props.get("conductance", 1.0)) * cfg.weight_factor

            cell = local_node_ids[tgt]
            section, x = self._resolve_target_section(cell, props)
            syn = self._build_pointer_mech(cfg, section, x, cond)

            has_pointer = hasattr(syn, f"_ref_{cfg.pointer_name}")
            transfer_var = self._transfer_var_name(syn, cfg.pointer_name)
            if self._pc is not None and transfer_var is not None:
                if hasattr(syn, "use_vgap"):
                    # Dual-mode mech: keep the serial POINTER bound (to
                    # the source v when local, else the target's own v)
                    # so it is never dangling.
                    src_sec = (
                        local_node_ids[src].soma
                        if src in local_node_ids
                        else section
                    )
                    neuron.h.setpointer(
                        src_sec(0.5)._ref_v, cfg.pointer_name, syn
                    )
                with _section_in_stack(section):
                    self._target_var(
                        syn, transfer_var, self._broadcast_id(src, cfg)
                    )
                if getattr(syn, "use_vgap", 0):
                    syn.vgap = self._sim.circuit_access.config.v_init
            elif has_pointer:
                if self._pc is not None and int(self._pc.nhost()) > 1:
                    raise BluecellulabError(
                        f"Mechanism '{cfg.mod}' only exposes POINTER "
                        f"'{cfg.pointer_name}': NEURON transfers can only "
                        "write into RANGE variables under MPI."
                    )
                src_section = local_node_ids[src].soma
                neuron.h.setpointer(src_section(0.5)._ref_v, cfg.pointer_name, syn)
            else:
                raise AttributeError(
                    f"Mechanism '{cfg.mod}' has neither a POINTER named "
                    f"'{cfg.pointer_name}' nor a RANGE transfer variable."
                )

            cell.continuous_synapses = getattr(cell, "continuous_synapses", [])
            cell.continuous_synapses.append(syn)
            n_added += 1

        logger.info(
            "ContinuousConnectionManager '%s': added %d connection(s) from "
            "edges='%s' (mod=%s, pointer=%s)",
            cfg.name, n_added, cfg.edge_population, cfg.mod, cfg.pointer_name,
        )
        return n_added

    def _broadcast_id(self, src: int, cfg: ContinuousConnectionConfig) -> int:
        """Broadcast id under which cell ``src`` exports its voltage.

        One id per source cell, offset far above the junction-id space.
        Uses the global gid namespace so multi-population circuits stay
        collision-free.
        """
        pop = self._sim.circuit_access._circuit.edges[
            cfg.edge_population
        ].source.name
        return self._MPI_TRANSFER_BASE + self._sim.global_gid(pop, src)

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
