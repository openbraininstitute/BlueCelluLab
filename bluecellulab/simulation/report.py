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
from typing import List
import numpy as np
import os

from bluecellulab.tools import resolve_segments, resolve_source_nodes
from bluecellulab.cell.cell_dict import CellDict

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


def write_compartment_report(
    report_name: str,
    output_path: str,
    cells: CellDict,
    report_cfg: dict,
    source_sets: dict,
    source_type: str,
    sim_dt: float
):
    """Write a SONATA-compatible compartment report to an HDF5 file.

    This function collects time series data (e.g., membrane voltage, ion currents)
    from a group of cells defined by either a node set or a compartment set, and
    writes the data to a SONATA-style report file.

    Args:
        output_path (str): Path to the output HDF5 file.
        cells (CellDict): Mapping of (population, node_id) to cell objects that
            provide access to pre-recorded variable traces.
        report_cfg (dict): Configuration for the report. Must include:
            - "variable_name": Name of the variable to report (e.g., "v", "ica", "ina").
            - "start_time", "end_time", "dt": Timing parameters.
            - "cells" or "compartments": Name of the node or compartment set.
        source_sets (dict): Dictionary of either node sets or compartment sets.
        source_type (str): Either "node_set" or "compartment_set".
        sim_dt (float): Simulation time step used for the recorded data.

    Raises:
        ValueError: If the specified source set is not found.

    Notes:
        - Currently supports only variables explicitly handled in Cell.get_variable_recording().
        - Cells without recordings for the requested variable will be skipped.
    """
    source_name = report_cfg.get("cells") if source_type == "node_set" else report_cfg.get("compartments")
    source = source_sets.get(source_name)
    if not source:
        logger.warning(f"{source_type.title()} '{source_name}' not found for report '{report_name}', skipping write.")
        return

    population = source["population"]

    node_ids, compartment_nodes = resolve_source_nodes(source, source_type, cells, population)

    data_matrix: List[np.ndarray] = []
    recorded_node_ids: List[int] = []
    index_pointers: List[int] = [0]
    element_ids: List[int] = []

    for node_id in node_ids:
        try:
            cell = cells[(population, node_id)]
        except KeyError:
            continue
        if not cell:
            continue

        targets = resolve_segments(cell, report_cfg, node_id, compartment_nodes, source_type)
        for sec, sec_name, seg in targets:
            try:
                variable = report_cfg.get("variable_name", "v")
                trace = cell.get_variable_recording(variable=variable, section=sec, segx=seg)
                data_matrix.append(trace)
                recorded_node_ids.append(node_id)
                element_ids.append(len(element_ids))
                index_pointers.append(index_pointers[-1] + 1)
            except Exception as e:
                logger.warning(f"Failed recording: GID {node_id} sec {sec_name} seg {seg}: {e}")

    if not data_matrix:
        logger.warning(f"No data recorded for report '{source_name}'. Skipping write.")
        return

    write_sonata_report_file(
        output_path, population, data_matrix, recorded_node_ids, index_pointers, element_ids, report_cfg, sim_dt
    )


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


def write_sonata_spikes(f_name: str, spikes_dict: dict[int, np.ndarray], population: str):
    """Write a SONATA spike group to a spike file from {node_id: [t1, t2,
    ...]}."""
    all_node_ids: List[int] = []
    all_timestamps: List[float] = []

    for node_id, times in spikes_dict.items():
        all_node_ids.extend([node_id] * len(times))
        all_timestamps.extend(times)

    if not all_timestamps:
        logger.warning(f"No spikes to write for population '{population}'.")

    # Sort by time for consistency
    sorted_indices = np.argsort(all_timestamps)
    node_ids_sorted = np.array(all_node_ids, dtype=np.uint64)[sorted_indices]
    timestamps_sorted = np.array(all_timestamps, dtype=np.float64)[sorted_indices]

    os.makedirs(os.path.dirname(f_name), exist_ok=True)
    with h5py.File(f_name, 'a') as f:  # 'a' to allow multiple writes
        spikes_group = f.require_group("spikes")
        if population in spikes_group:
            logger.warning(f"Overwriting existing group for population '{population}' in {f_name}.")
            del spikes_group[population]

        group = spikes_group.create_group(population)
        sorting_enum = h5py.enum_dtype({'none': 0, 'by_id': 1, 'by_time': 2}, basetype='u1')
        group.attrs.create("sorting", 2, dtype=sorting_enum)  # 2 = by_time

        timestamps_ds = group.create_dataset("timestamps", data=timestamps_sorted)
        group.create_dataset("node_ids", data=node_ids_sorted)

        timestamps_ds.attrs["units"] = "ms"  # SONATA-required
