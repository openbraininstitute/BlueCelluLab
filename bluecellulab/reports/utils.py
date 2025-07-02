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

from bluecellulab.tools import resolve_segments, resolve_source_nodes

logger = logging.getLogger(__name__)


def _configure_recording(cell, report_cfg, source, source_type, report_name):
    """Configure recording of a variable on a single cell.

    This function sets up the recording of the specified variable (e.g., membrane voltage)
    in the target cell, for each resolved segment.

    Parameters
    ----------
    cell : Any
        The cell object on which to configure recordings.

    report_cfg : dict
        The configuration dictionary for this report.

    source : dict
        The source definition specifying nodes or compartments.

    source_type : str
        Either "node_set" or "compartment_set".

    report_name : str
        The name of the report (used in logging).
    """
    variable = report_cfg.get("variable_name", "v")

    node_id = cell.cell_id
    compartment_nodes = source.get("compartment_set") if source_type == "compartment_set" else None

    targets = resolve_segments(cell, report_cfg, node_id, compartment_nodes, source_type)
    for sec, sec_name, seg in targets:
        try:
            cell.add_variable_recording(variable=variable, section=sec, segx=seg)
        except AttributeError:
            logger.warning(f"Recording for variable '{variable}' is not implemented in Cell.")
            return
        except Exception as e:
            logger.warning(
                f"Failed to record '{variable}' at {sec_name}({seg}) on GID {node_id} for report '{report_name}': {e}"
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
        section = report_cfg.get("sections", "soma")

        if report_type != "compartment":
            raise NotImplementedError(f"Report type '{report_type}' is not supported.")

        if section == "compartment_set":
            source_type = "compartment_set"
            source_sets = simulation_config.get_compartment_sets()
            source_name = report_cfg.get("compartments")
            if not source_name:
                logger.warning(f"Report '{report_name}' does not specify a node set in 'compartments' for {source_type}.")
                continue
        else:
            source_type = "node_set"
            source_sets = simulation_config.get_node_sets()
            source_name = report_cfg.get("cells")
            if not source_name:
                logger.warning(f"Report '{report_name}' does not specify a node set in 'cells' for {source_type}.")
                continue

        source = source_sets.get(source_name)
        if not source:
            logger.warning(f"{source_type.title()} '{source_name}' not found for report '{report_name}', skipping recording.")
            continue

        population = source["population"]
        node_ids, _ = resolve_source_nodes(source, source_type, cells, population)

        for node_id in node_ids:
            cell = cells.get((population, node_id))
            if not cell:
                continue
            _configure_recording(cell, report_cfg, source, source_type, report_name)


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
        if isinstance(key, tuple):
            pop, gid = key
        else:
            raise ValueError(f"Cell key {key} is not a (population, gid) tuple.")

        times = cell.get_recorded_spikes(location=location, threshold=threshold)
        if times is not None and len(times) > 0:
            spikes_by_pop[pop][gid] = list(times)

    return dict(spikes_by_pop)
