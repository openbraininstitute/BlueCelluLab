# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""User corrections on gap junction connectivity.

Port of ``neurodamus/gap_junction_user_corrections.py``. The modification
parameters are read from the ``beta_features`` section of the SONATA
simulation config:

- ``deterministic_stoch``: set ``deterministic_StochKv* = 1`` on all sections.
- ``procedure_type`` in ``{"validation_sim", "find_holding_current"}``:
  overwrite every gap-junction conductance with ``gjc``.
- ``remove_channels``: ``"all"`` / ``"only_stoch"`` / ``"only_non_stoch"``
  uninserts channel mechanisms.
- ``load_g_pas_file``: per-cell ``g_pas`` from an HDF5 file
  (``g_pas/<gjc>/a<gid>/<segment>``, indexed by ``correction_iteration_load``).
- ``manual_MEComboInfo_file``: per-cell IClamp holding currents
  (``holding_per_gid/<gjc>/a<gid>``); incompatible with
  ``procedure_type == "find_holding_current"``.
- ``procedure_type == "find_holding_current"`` + ``vc_amp``: per-cell
  SEClamp voltage clamps (``v_per_gid/a<gid>``); the clamp currents are
  recorded and can be saved with ``CircuitSimulation``'s seclamp recorder.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import neuron

from bluecellulab.exceptions import BluecellulabError

if TYPE_CHECKING:
    from bluecellulab.circuit_simulation import CircuitSimulation

logger = logging.getLogger(__name__)

non_stochastic_mechs = [
    "NaTs2_t",
    "SKv3_1",
    "Nap_Et2",
    "Ih",
    "Im",
    "KdShu2007",
    "K_Pst",
    "K_Tst",
    "Ca",
    "SK_E2",
    "Ca_LVAst",
    "CaDynamics_E2",
    "NaTa_t",
    "CaDynamics_DC0",
    "Ca_HVA2",
    "NaTg",
    "TC_cad",
    "TC_ih_Bud97",
    "TC_Nap_Et2",
    "TC_iA",
    "TC_iL",
    "TC_HH",
    "TC_iT_Des98",
    "kdrb",
    "na3",
    "kap",
    "hd",
    "can",
    "cal",
    "cat",
    "cagk",
    "kca",
    "cacum",
    "kdb",
    "kmb",
    "kad",
    "nax",
    "cacumb",
]

stochastic_mechs = ["StochKv", "StochKv2", "StochKv3"]


def load_user_modifications(sim: "CircuitSimulation") -> tuple[dict[int, Any], dict[int, Any]]:
    """Apply ``beta_features`` user modifications on gap junction connections,
    following neurodamus.

    Returns ``(holding_ic_per_gid, seclamp_per_gid)`` — clamps are keyed
    by SONATA node id (identical to neurodamus' final gid for single-
    node- population circuits).
    """
    settings = sim.circuit_access.config.beta_features()
    gjc = settings.get("gjc")

    if settings.get("deterministic_stoch"):
        logger.info("Set deterministic = 1 for StochKv")
        _deterministic_stoch(sim)

    if settings.get("procedure_type") in {"validation_sim", "find_holding_current"}:
        process_gap_conns = _update_conductance(gjc, sim)
        all_ranks_total = int(_mpi_sum(sim, process_gap_conns))
        logger.info("Set GJc = %s for %s gap synapses", gjc, all_ranks_total)

    remove_channels = settings.get("remove_channels")
    if remove_channels:
        if remove_channels == "all":
            rm_mechanisms = non_stochastic_mechs + stochastic_mechs
        elif remove_channels == "only_stoch":
            rm_mechanisms = stochastic_mechs
        elif remove_channels == "only_non_stoch":
            rm_mechanisms = non_stochastic_mechs
        else:
            logger.warning("Unknown GJ remove_channels setting: %s", remove_channels)
            rm_mechanisms = []
        if rm_mechanisms:
            logger.info("Remove channels type = %s", remove_channels)
            _perform_remove_channels(sim, rm_mechanisms)

    if filename := settings.get("load_g_pas_file"):
        processed_cells = _update_gpas(
            sim, filename, gjc, settings.get("correction_iteration_load", -1)
        )
        all_ranks_total = int(_mpi_sum(sim, processed_cells))
        logger.info(
            "Update g_pas to fit %s - file %s for %s cells", gjc, filename, all_ranks_total
        )

    holding_ic_per_gid: dict[int, Any] = {}
    if filename := settings.get("manual_MEComboInfo_file"):
        if settings.get("procedure_type") == "find_holding_current":
            raise BluecellulabError(
                "find_holding_current should not read manual_MEComboInfo_file"
            )
        holding_ic_per_gid = _load_holding_ic(sim, filename, gjc=gjc)
        all_ranks_total = int(_mpi_sum(sim, len(holding_ic_per_gid)))
        logger.info(
            "Load holding_ic from manual_MEComboInfoFile %s for %s cells",
            filename,
            all_ranks_total,
        )

    seclamp_per_gid: dict[int, Any] = {}
    if settings.get("procedure_type") == "find_holding_current":
        seclamp_per_gid = _find_holding_current(sim, settings.get("vc_amp"))
        all_ranks_total = int(_mpi_sum(sim, len(seclamp_per_gid)))
        logger.info(
            "Inject holding voltages from file %s for %s cells",
            settings["vc_amp"],
            all_ranks_total,
        )

    return holding_ic_per_gid, seclamp_per_gid


def _mpi_sum(sim: "CircuitSimulation", value):
    """All-rank sum under MPI, identity in serial."""
    pc = getattr(sim, "pc", None)
    if pc is not None:
        return pc.py_sum(value, 0)
    return value


def _local_node_ids(sim: "CircuitSimulation") -> dict[int, Any]:
    """SONATA node ids of locally instantiated cells -> Cell."""
    return {cid.id: cell for cid, cell in sim.cells.items()}


def _update_conductance(gjc, sim: "CircuitSimulation") -> int:
    """Update gap junction connection conductance on local cells."""
    n_conn = 0
    for cell in sim.cells.values():
        for gj in getattr(cell, "gap_junctions", []):
            for name in ("g", "weight", "conductance"):
                if hasattr(gj, name):
                    setattr(gj, name, gjc)
                    break
            n_conn += 1
    return n_conn


def _deterministic_stoch(sim: "CircuitSimulation") -> None:
    """Enable deterministic_StochKv in cells."""
    for cell in sim.cells.values():
        for sec in cell.cell.all:
            if "StochKv3" in dir(sec(0.5)):
                sec.deterministic_StochKv3 = 1
            if "StochKv2" in dir(sec(0.5)):
                sec.deterministic_StochKv2 = 1
            if "StochKv1" in dir(sec(0.5)):
                sec.deterministic_StochKv1 = 1


def _perform_remove_channels(sim: "CircuitSimulation", mechanisms: list) -> None:
    """Remove certain mechanisms from the cells."""
    for cell in sim.cells.values():
        for sec in cell.cell.all:
            for mec in mechanisms:
                if mec in dir(sec(0.5)):
                    sec.uninsert(mec)


def _update_gpas(sim: "CircuitSimulation", filename, gjc, correction_iteration_load) -> int:
    """Update the g_pas attribute of certain cells.

    Cell ids, segment names and new g_pas values are provided in the input
    file (``g_pas/<gjc>/a<gid>/<segment_name>`` datasets).
    """
    import h5py

    processed_cells = 0
    try:
        g_pas_file = h5py.File(filename, "r")
    except OSError as e:
        raise BluecellulabError(f"Error opening g_pas file {filename}") from e
    local_cells = _local_node_ids(sim)
    if f"g_pas/{gjc}" not in g_pas_file:
        logger.warning("Data for g_pas/%s not found in %s", gjc, filename)
        return 0
    for agid in g_pas_file[f"g_pas/{gjc}/"]:
        gid = int(agid[1:])
        if gid in local_cells:  # if the node has a part of the cell
            cell = local_cells[gid]
            processed_cells += 1
            for sec in cell.cell.all:
                for seg in sec:
                    try:
                        attr_name = str(seg)[str(seg).index(".") + 1:]
                        value = g_pas_file[f"g_pas/{gjc}/{agid}"][attr_name][
                            correction_iteration_load
                        ]
                    except Exception as e:
                        raise BluecellulabError(
                            f"Failed to load data in g_pas file {filename}: {e}"
                        ) from e
                    seg.g_pas = value
    return processed_cells


def _load_holding_ic(sim: "CircuitSimulation", filename, gjc) -> dict[int, Any]:
    """Add IClamps to certain cells for holding the membrane voltage.

    Cell ids and current amplitudes are provided in the input file
    (``holding_per_gid/<gjc>/a<gid>`` datasets).
    """
    import h5py

    holding_ic_per_gid: dict[int, Any] = {}
    try:
        holding_per_gid = h5py.File(filename, "r")
    except OSError as e:
        raise BluecellulabError(f"Error opening MEComboInfo file {filename}") from e

    if f"holding_per_gid/{gjc}" not in holding_per_gid:
        logger.warning("Data for holding_per_gid/%s not found in %s", gjc, holding_per_gid)
        return holding_ic_per_gid
    local_cells = _local_node_ids(sim)
    for agid in holding_per_gid["holding_per_gid"][str(gjc)]:
        gid = int(agid[1:])
        if gid in local_cells:
            holding_ic_per_gid[gid] = neuron.h.IClamp(0.5, sec=local_cells[gid].soma)
            holding_ic_per_gid[gid].dur = 9e9
            try:
                holding_ic_per_gid[gid].amp = holding_per_gid["holding_per_gid"][str(gjc)][agid][()]
            except Exception as e:
                raise BluecellulabError(
                    f"Failed to load data in g_pas file {filename}: {e}"
                ) from e
    return holding_ic_per_gid


def _find_holding_current(sim: "CircuitSimulation", filename) -> dict[int, Any]:
    """Add SEClamps to certain cells for holding the membrane voltage.

    Cell ids and voltage amplitudes are provided in the input file
    (``v_per_gid/a<gid>`` datasets).
    """
    import h5py

    try:
        v_per_gid = h5py.File(filename, "r")
    except OSError as e:
        raise BluecellulabError(f"Error opening voltage file {filename}") from e

    logger.info("Inject voltage clamps without disabling holding current!")

    seclamp_per_gid: dict[int, Any] = {}
    local_cells = _local_node_ids(sim)
    for agid in v_per_gid["v_per_gid"]:
        gid = int(agid[1:])
        if gid in local_cells:
            seclamp_per_gid[gid] = neuron.h.SEClamp(0.5, sec=local_cells[gid].soma)
            seclamp_per_gid[gid].dur1 = 9e9
            seclamp_per_gid[gid].amp1 = float(v_per_gid["v_per_gid"][agid][()])
            seclamp_per_gid[gid].rs = 0.0000001
    return seclamp_per_gid
