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

import logging

from bluecellulab.tools import resolve_segments

logger = logging.getLogger(__name__)


def _configure_recording(cell, report_cfg, source, source_type, report_name):
    variable = report_cfg.get("variable_name", "v")
    if variable != "v":
        logger.warning(f"Unsupported variable '{variable}' for report '{report_name}'")
        return

    node_id = cell.cell_id
    compartment_nodes = source.get("compartment_set") if source_type == "compartment_set" else None

    targets = resolve_segments(cell, report_cfg, node_id, compartment_nodes, source_type)
    for sec, sec_name, seg in targets:
        try:
            cell.add_voltage_recording(section=sec, segx=seg)
        except Exception as e:
            logger.warning(
                f"Failed to record voltage at {sec_name}({seg}) on GID {node_id} for report '{report_name}': {e}"
            )


def configure_all_reports(cells, simulation_config):
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
        else:
            source_type = "node_set"
            source_sets = simulation_config.get_node_sets()
            source_name = report_cfg.get("cells")

        source = source_sets.get(source_name)
        if not source:
            logger.warning(f"{source_type.title()} '{source_name}' not found for report '{report_name}'")
            continue

        population = source["population"]

        if source_type == "compartment_set":
            node_ids = [entry[0] for entry in source.get("compartment_set", [])]
        else:  # node_set
            if "node_id" in source:
                node_ids = source["node_id"]
            else:
                # Fallback: use all available node IDs from this population
                node_ids = [node_id for (pop, node_id) in cells.keys() if pop == population]

        for node_id in node_ids:
            cell = cells.get((population, node_id))
            if not cell:
                continue
            _configure_recording(cell, report_cfg, source, source_type, report_name)
