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
"""Report class of bluecellulab."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Dict, Any, List, Mapping, Optional, Tuple

from bluecellulab.circuit.node_id import CellId
import numpy as np

from bluecellulab.cell.section_tools import section_to_variable_recording_str
from bluecellulab.type_aliases import NeuronSection, SiteEntry
from bluecellulab.tools import (
    resolve_source_nodes,
)

logger = logging.getLogger(__name__)

SUPPORTED_REPORT_TYPES = {"compartment", "compartment_set"}


def _get_source_for_report(simulation_config: Any, report_name: str, report_cfg: dict) -> tuple[str, dict]:
    report_type = report_cfg.get("type", "compartment")

    if report_type == "compartment_set":
        source_sets = simulation_config.get_compartment_sets()
        source_name = report_cfg.get("compartment_set")
        key = "compartment_set"
    elif report_type == "compartment":
        source_sets = simulation_config.get_node_sets()
        source_name = report_cfg.get("cells")
        key = "cells"
    else:
        raise NotImplementedError(
            f"Report type '{report_type}' is not supported. Supported types: {SUPPORTED_REPORT_TYPES}"
        )

    if not source_name:
        logger.warning("Report '%s' missing '%s' for type '%s'.", report_name, key, report_type)
        raise KeyError("missing_source_name")

    source = source_sets.get(source_name)
    if not source:
        logger.warning("%s '%s' not found for report '%s', skipping.", report_type, source_name, report_name)
        raise KeyError("missing_source")

    return report_type, source


def _ensure_report_sites(cell: Any) -> dict[str, list[SiteEntry]]:
    report_sites = getattr(cell, "report_sites", None)
    if not isinstance(report_sites, dict):
        report_sites = {}
        setattr(cell, "report_sites", report_sites)
    return report_sites


def prepare_recordings_for_reports(
    cells: Dict[CellId, Any],
    simulation_config: Any,
) -> tuple[dict[CellId, list[str]], dict[CellId, list[SiteEntry]]]:
    """Configure report recordings on instantiated cells and build recording
    indices.

    Parameters
    ----------
    cells
        Mapping of CellId -> live Cell objects.
    simulation_config
        Simulation config providing report entries and node/compartment sets.

    Returns
    -------
    (recording_index, sites_index)
        recording_index maps CellId -> ordered list of recording names (rec_name).
        sites_index maps CellId -> list of site entries (report, rec_name, section, segx).

    Notes
    -----
    Populates `cell.report_sites[report_name]` with the configured site entries.
    """
    recording_index: dict[CellId, list[str]] = defaultdict(list)
    sites_index: dict[CellId, list[SiteEntry]] = defaultdict(list)

    for report_name, report_cfg in simulation_config.get_report_entries().items():
        try:
            report_type, source = _get_source_for_report(simulation_config, report_name, report_cfg)
        except KeyError:
            continue

        population = source["population"]
        node_ids, compartment_nodes = resolve_source_nodes(source, report_type, cells, population)

        sites_per_cell = build_recording_sites(
            cells, node_ids, population, report_type, report_cfg, compartment_nodes
        )
        variable = report_cfg.get("variable_name", "v")

        for node_id, sites in sites_per_cell.items():
            cell_id = CellId(population, node_id)
            cell = cells.get(cell_id)
            if cell is None or not sites:
                continue

            report_sites = _ensure_report_sites(cell)
            report_sites.setdefault(report_name, [])

            rec_names = cell.configure_recording(sites, variable, report_name)
            if len(rec_names) != len(sites):
                logger.warning(
                    "Configured %d/%d recording sites for report '%s' on %s.",
                    len(rec_names), len(sites), report_name, cell_id,
                )

            for (sec, sec_name, segx), rec_name in zip(sites, rec_names):
                recording_index[cell_id].append(rec_name)

                entry: SiteEntry = {
                    "report": report_name,
                    "rec_name": rec_name,
                    "section": sec_name,
                    "segx": float(segx),
                }
                sites_index[cell_id].append(entry)
                report_sites[report_name].append(entry)

    return dict(recording_index), dict(sites_index)


def build_recording_sites(
    cells: Dict[CellId, Any],
    node_ids: list[int],
    population: str,
    report_type: str,
    report_cfg: dict,
    compartment_nodes: list | None,
) -> Dict[int, List[Tuple[Any, str, float]]]:
    """Resolve recording sites for instantiated cells in one population.

    Parameters
    ----------
    cells : dict[CellId, Any]
        Mapping from CellId to cell-like objects.
    node_ids : list[int]
        Node IDs to resolve within `population`.
    population : str
        Population name used to build CellId(population, node_id).
    report_type : str
        "compartment" or "compartment_set".
    report_cfg : dict
        Report configuration.
    compartment_nodes : list | None
        Compartment-set entries used when `report_type == "compartment_set"`.

    Returns
    -------
    dict[int, list[tuple[Any, str, float]]]
        Mapping `{node_id: [(section_obj, section_name, segx), ...]}`.
    """
    targets_per_cell: Dict[int, List[Tuple[Any, str, float]]] = {}

    for node_id in node_ids:
        cell = cells.get(CellId(population, node_id))
        if cell is None:
            continue

        if report_type == "compartment_set":
            if compartment_nodes is None:
                continue
            targets = cell.resolve_segments_from_compartment_set(node_id, compartment_nodes)
        elif report_type == "compartment":
            targets = cell.resolve_segments_from_config(report_cfg)
        else:
            raise NotImplementedError(
                f"Report type '{report_type}' is not supported. Supported: {SUPPORTED_REPORT_TYPES}"
            )

        if targets:
            targets_per_cell[node_id] = targets

    return targets_per_cell


def extract_spikes_from_cells(
    cells: Dict[Any, Any],
    location: str = "soma",
    threshold: float = -20.0,
) -> Dict[str, Dict[int, list]]:
    """Extract spike times from recorded cells, grouped by population.

    Parameters
    ----------
    cells : dict
        Mapping from (population, gid) → Cell object, or similar.

    location : str
        Recording location passed to Cell.get_recorded_spikes().

    threshold : float
        Voltage threshold (mV) used for spike detection.

    Returns
    -------
    spikes_by_population : dict
        {population → {gid_int → [spike_times_ms]}}
    """
    spikes_by_pop: defaultdict[str, Dict[int, List[float]]] = defaultdict(dict)

    for key, cell in cells.items():
        # Resolve the key to (population, gid)
        if isinstance(key, tuple):
            pop, gid = key
        elif isinstance(key, str):
            try:
                pop, gid_str = key.rsplit("_", 1)
                gid = int(gid_str)
            except Exception:
                raise ValueError(
                    f"Cell key '{key}' could not be parsed as 'population_gid'"
                )
        else:
            raise ValueError(f"Cell key '{key}' is not a recognized format.")

        if not hasattr(cell, "get_recorded_spikes"):
            raise TypeError(
                f"Cannot extract spikes: cell entry {key} is not a Cell object (got {type(cell)}). "
                "If you have precomputed traces, pass them as `spikes_by_pop`."
            )

        times = cell.get_recorded_spikes(location=location, threshold=threshold)
        # Always assign, even if empty
        spikes_by_pop[pop][gid] = list(times) if times is not None else []

    return dict(spikes_by_pop)


@dataclass(frozen=True)
class RecordedCell:
    """Read-only cell-like object backed by stored recordings."""
    recordings: Dict[str, np.ndarray]
    report_sites: Dict[str, list[dict]]
    soma: NeuronSection | None = None

    def get_recording(self, var_name: str) -> np.ndarray:
        try:
            return self.recordings[var_name]
        except KeyError as e:
            raise ValueError(f"No recording for '{var_name}' was found.") from e

    def get_variable_recording(self, variable: str, section: Any, segx: float) -> np.ndarray:
        if section is None:
            section = self.soma
        rec_name = section_to_variable_recording_str(section, float(segx), variable)
        return self.get_recording(rec_name)


def payload_to_cells(
    payload: Mapping[str, Any],
    sites_index: Mapping[CellId, list[SiteEntry]],
) -> Dict[CellId, RecordedCell]:
    """
    payload: {"pop_gid": {"recordings": {rec_name: [floats...]}}}
    sites_index: {(pop,gid): [{"report":..., "rec_name":..., "section":..., "segx":...}, ...]}
    """
    out: Dict[CellId, RecordedCell] = {}

    for key, blob in payload.items():
        pop, gid_s = key.rsplit("_", 1)
        gid = int(gid_s)

        recs = blob.get("recordings", {}) or {}
        recs_np = {name: np.asarray(vals, dtype=np.float32) for name, vals in recs.items()}

        by_report: dict[str, list[dict]] = defaultdict(list)
        cell_id = CellId(pop, gid)
        for site in sites_index.get(cell_id, []):
            by_report[site["report"]].append(site)

        out[cell_id] = RecordedCell(
            recordings=recs_np,
            report_sites=dict(by_report),
        )

    return out


def merge_dicts(dicts: list[dict]) -> dict:
    out: dict = {}
    for d in dicts:
        out.update(d)
    return out


def merge_spikes(list_of_pop_dicts: list[dict[str, dict[int, list]]]) -> dict[str, dict[int, list]]:
    out: dict[str, dict[int, list]] = defaultdict(dict)
    for pop_dict in list_of_pop_dicts:
        for pop, gid_map in pop_dict.items():
            out[pop].update(gid_map)
    return out


def gather_recording_sites(
    gathered_per_rank: list[Dict[CellId, List[SiteEntry]]]
) -> Dict[CellId, List[SiteEntry]]:
    """Combine per-rank recording site registries into a global one.

    Each rank contributes recording locations for the cells it
    instantiated. This reconstructs the full recording topology across
    MPI ranks.
    """
    merged: dict[CellId, list[SiteEntry]] = defaultdict(list)

    for rank_dict in gathered_per_rank:
        if not rank_dict:
            continue
        for cell_key, sites in rank_dict.items():
            merged[cell_key].extend(sites)

    return dict(merged)


def collect_local_payload(
    cells: Dict[CellId, Any],
    cell_ids_for_this_rank: list[CellId],
    recording_index: Dict[CellId, list[str]],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Build rank-local payload: {'pop_gid': {'recordings': {rec_name: trace_list}}}
    """
    payload: dict[str, dict[str, dict[str, list[float]]]] = {}

    for pop, gid in cell_ids_for_this_rank:
        cell_id = CellId(pop, gid)
        cell = cells.get(cell_id)
        if cell is None:
            continue

        recs: dict[str, list[float]] = {}
        for rec_name in recording_index.get(cell_id, []):
            recs[rec_name] = cell.get_recording(rec_name).tolist()

        payload[f"{pop}_{gid}"] = {"recordings": recs}

    return payload


def gather_payload_to_rank0(
    pc: Any,
    local_payload: dict,
    local_spikes: dict,
) -> tuple[Optional[dict], Optional[dict]]:
    """Gather payload + spikes.

    Returns (all_payload, all_spikes) on rank 0, else (None, None).
    """
    gathered_payload = pc.py_gather(local_payload, 0)
    gathered_spikes = pc.py_gather(local_spikes, 0)

    if int(pc.id()) != 0:
        return None, None

    all_payload = merge_dicts(gathered_payload)
    all_spikes = merge_spikes(gathered_spikes)
    return all_payload, all_spikes


def collect_local_spikes(
    sim: Any,
    cell_ids_for_this_rank: list[CellId],
) -> dict[str, dict[int, list[float]]]:
    """
    Collect recorded spike times for local cells in {pop: {gid: [times...]}} form.
    """
    spikes: dict[str, dict[int, list[float]]] = defaultdict(dict)

    for pop, gid in cell_ids_for_this_rank:
        try:
            cell = sim.cells[CellId(pop, gid)]
            times = cell.get_recorded_spikes(
                location=sim.spike_location,
                threshold=sim.spike_threshold,
            )
            spikes[pop][gid] = list(times) if times is not None else []
        except Exception:
            spikes[pop][gid] = []

    return spikes
