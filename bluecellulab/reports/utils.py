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

from collections import defaultdict
import logging
from typing import Dict, Any, List

from bluecellulab.tools import (
    resolve_segments_from_compartment_set,
    resolve_segments_from_config,
    resolve_source_nodes,
)

logger = logging.getLogger(__name__)

SUPPORTED_REPORT_TYPES = {"compartment", "compartment_set"}


def _configure_recording(cell, recording_sites, variable_name, report_name):
    """Configure recording of a variable on a single cell.

    This function sets up the recording of the specified variable (e.g., membrane voltage)
    in the target cell, for each resolved segment.

    Parameters
    ----------
    cell : Any
        The cell object on which to configure recordings.

    recording_sites : list of tuples
        List of tuples (section, section_name, segment) where:
        - section is the section object in the cell.
        - section_name is the name of the section.
        - segment is the Neuron segment index (0-1).

    variable_name : str
        The name of the variable to record (e.g., "v" for membrane voltage).

    report_name : str
        The name of the report (used in logging).
    """
    node_id = cell.cell_id.id

    for sec, sec_name, seg in recording_sites:
        try:
            cell.add_variable_recording(variable=variable_name, section=sec, segx=seg)
            logger.info(
                f"Recording '{variable_name}' at {sec_name}({seg}) on GID {node_id} for report '{report_name}'"
            )
        except AttributeError:
            logger.warning(
                f"Recording for variable '{variable_name}' is not implemented in Cell."
            )
            return
        except Exception as e:
            logger.warning(
                f"Failed to record '{variable_name}' at {sec_name}({seg}) on GID {node_id} for report '{report_name}': {e}"
            )


def configure_all_reports(cells, simulation_config):
    """Configure recordings for all reports defined in the simulation
    configuration.

    This iterates through all report entries, resolves source nodes or compartments,
    and configures the corresponding recordings on each cell.

    Parameters
    ----------
    cells : dict
        Mapping from (population, gid) → Cell object.

    simulation_config : Any
        Simulation configuration object providing report entries,
        node sets, and compartment sets.
    """
    report_entries = simulation_config.get_report_entries()

    for report_name, report_cfg in report_entries.items():
        report_type = report_cfg.get("type", "compartment")
        if report_type == "compartment_set":
            source_sets = simulation_config.get_compartment_sets()
            source_name = report_cfg.get("compartment_set")
            if not source_name:
                logger.warning(
                    f"Report '{report_name}' does not specify a node set in 'compartment_set' for {source_type}."
                )
                continue
        elif report_type == "compartment":
            source_sets = simulation_config.get_node_sets()
            source_name = report_cfg.get("cells")
            if not source_name:
                logger.warning(
                    f"Report '{report_name}' does not specify a node set in 'cells' for {source_type}."
                )
                continue
        else:
            raise NotImplementedError(
                f"Report type '{report_type}' is not supported. "
                f"Supported types: {SUPPORTED_REPORT_TYPES}"
            )

        source = source_sets.get(source_name)
        if not source:
            logger.warning(
                f"{report_type} '{source_name}' not found for report '{report_name}', skipping recording."
            )
            continue

        population = source["population"]
        node_ids, compartment_nodes = resolve_source_nodes(
            source, report_type, cells, population
        )
        recording_sites_per_cell = build_recording_sites(
            cells, node_ids, population, report_type, report_cfg, compartment_nodes
        )
        variable_name = report_cfg.get("variable_name", "v")

        for node_id, recording_sites in recording_sites_per_cell.items():
            cell = cells.get((population, node_id))
            if not cell or recording_sites is None:
                continue

            _configure_recording(cell, recording_sites, variable_name, report_name)


def build_recording_sites(
    cells, node_ids, population, report_type, report_cfg, compartment_nodes
):
    """Build per-cell recording sites based on source type and report
    configuration.

    This function resolves the segments (section, name, seg.x) where variables
    should be recorded for each cell, based on either a node set (standard
    compartment reports) or a compartment set (predefined segment list).

    Parameters
    ----------
    cells : dict
        Mapping from (population, node_id) to Cell object.

    node_ids : list of int
        List of node IDs for which recordings should be configured.

    population : str
        Name of the population to which the cells belong.

    report_type : str
        The report type, either 'compartment_set' or 'compartment'.

    report_cfg : dict
        Configuration dictionary specifying report parameters

    compartment_nodes : list or None
        Optional list of [node_id, section_name, seg_x] defining segment locations
        for each cell (used if report_type == 'compartment_set').

    Returns
    -------
    dict
        Mapping from node ID to list of recording site tuples:
        (section_object, section_name, seg_x).
    """
    targets_per_cell = {}

    for node_id in node_ids:
        cell = cells.get((population, node_id)) or cells.get(f"{population}_{node_id}")
        if not cell:
            continue

        if report_type == "compartment_set":
            targets = resolve_segments_from_compartment_set(
                cell, node_id, compartment_nodes
            )
        elif report_type == "compartment":
            targets = resolve_segments_from_config(cell, report_cfg)
        else:
            raise NotImplementedError(
                f"Report type '{report_type}' is not supported. "
                f"Supported types: {SUPPORTED_REPORT_TYPES}"
            )

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
