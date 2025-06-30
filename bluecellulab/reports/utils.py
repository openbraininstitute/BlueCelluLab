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
from pathlib import Path
import h5py
import numpy as np
from typing import Dict, Any

from bluecellulab.tools import resolve_segments, resolve_source_nodes

logger = logging.getLogger(__name__)


def _configure_recording(cell, report_cfg, source, source_type, report_name):
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


def write_sonata_report_file(
    output_path,
    population,
    data_matrix,
    recorded_node_ids,
    index_pointers,
    element_ids,
    report_cfg,
    sim_dt
):
    start_time = float(report_cfg.get("start_time", 0.0))
    end_time = float(report_cfg.get("end_time", 0.0))
    dt_report = float(report_cfg.get("dt", sim_dt))

    # Clamp dt_report if finer than simuldation dt
    if dt_report < sim_dt:
        logger.warning(
            f"Requested report dt={dt_report} ms is finer than simulation dt={sim_dt} ms. "
            f"Clamping report dt to {sim_dt} ms."
        )
        dt_report = sim_dt

    step = int(round(dt_report / sim_dt))
    if not np.isclose(step * sim_dt, dt_report, atol=1e-9):
        raise ValueError(
            f"dt_report={dt_report} is not an integer multiple of dt_data={sim_dt}"
        )

    # Downsample the data if needed
    # Compute start and end indices in the original data
    start_index = int(round(start_time / sim_dt))
    end_index = int(round(end_time / sim_dt)) + 1  # inclusive

    # Now slice and downsample
    data_matrix_downsampled = [
        trace[start_index:end_index:step] for trace in data_matrix
    ]
    data_array = np.stack(data_matrix_downsampled, axis=1).astype(np.float32)

    # Prepare metadata arrays
    node_ids_arr = np.array(recorded_node_ids, dtype=np.uint64)
    index_ptr_arr = np.array(index_pointers, dtype=np.uint64)
    element_ids_arr = np.array(element_ids, dtype=np.uint32)
    time_array = np.array([start_time, end_time, dt_report], dtype=np.float64)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to HDF5
    with h5py.File(output_path, "w") as f:
        grp = f.require_group(f"/report/{population}")
        data_ds = grp.create_dataset("data", data=data_array.astype(np.float32))

        variable = report_cfg.get("variable_name", "v")
        if variable == "v":
            data_ds.attrs["units"] = "mV"

        mapping = grp.require_group("mapping")
        mapping.create_dataset("node_ids", data=node_ids_arr)
        mapping.create_dataset("index_pointers", data=index_ptr_arr)
        mapping.create_dataset("element_ids", data=element_ids_arr)
        time_ds = mapping.create_dataset("time", data=time_array)
        time_ds.attrs["units"] = "ms"


def extract_spikes_from_cells(
    cells: Dict[Any, Any],
    location: str = "soma",
    threshold: float = -20.0,
) -> Dict[str, Dict[int, list]]:
    """
    Extract spike times from recorded cells, grouped by population.

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
    spikes_by_pop: Dict[str, Dict[int, list[float]]] = {}

    for key, cell in cells.items():
        if isinstance(key, tuple):
            pop, gid = key

        times = cell.get_recorded_spikes(location=location, threshold=threshold)
        if times is not None:
            spikes_by_pop[pop][gid] = list(times)

    return dict(spikes_by_pop)
