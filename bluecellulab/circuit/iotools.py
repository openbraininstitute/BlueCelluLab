# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Input/output operations from circuits and simulations."""

from __future__ import annotations
from pathlib import Path
import logging
from typing import List

import numpy as np
import h5py

from bluecellulab.tools import get_sections, resolve_segments
from bluecellulab.cell.cell_dict import CellDict
from bluecellulab.circuit.node_id import CellId

logger = logging.getLogger(__name__)


def parse_outdat(path: str | Path) -> dict[CellId, np.ndarray]:
    """Parse the replay spiketrains in a out.dat formatted file pointed to by
    path."""
    import bluepy
    spikes = bluepy.impl.spike_report.SpikeReport.load(path).get()
    # convert Series to DataFrame with 2 columns for `groupby` operation
    spike_df = spikes.to_frame().reset_index()
    if (spike_df["t"] < 0).any():
        logger.warning('Found negative spike times in out.dat ! '
                       'Clipping them to 0')
        spike_df["t"].clip(lower=0., inplace=True)

    outdat = spike_df.groupby("gid")["t"].apply(np.array)
    # convert outdat's index from int to CellId
    outdat.index = [CellId("", gid) for gid in outdat.index]
    return outdat.to_dict()


def write_compartment_report(
    output_path: str,
    cells: CellDict,
    report_cfg: dict,
    source_sets: dict,
    source_type: str,
):
    """Write a SONATA-compatible compartment report to an HDF5 file."""
    source_name = report_cfg.get("cells") if source_type == "node_set" else report_cfg.get("compartments")
    source = source_sets.get(source_name)
    if not source:
        raise ValueError(f"{source_type} '{source_name}' not found.")

    population = source["population"]

    if source_type == "compartment_set":
        compartment_nodes = source.get("compartment_set", [])
        node_ids = [n[0] for n in compartment_nodes]
    else:
        if "node_id" in source:
            node_ids = source["node_id"]
        else:
            # Fallback: get all node IDs for the population from cells
            node_ids = [node_id for (pop, node_id) in cells.keys() if pop == population]
        compartment_nodes = None  # Not used for node_set

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
                trace = cell.get_voltage_recording(section=sec, segx=seg)
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
        output_path, population, data_matrix, recorded_node_ids, index_pointers, element_ids, report_cfg
    )


def write_sonata_report_file(
    output_path, population, data_matrix, recorded_node_ids, index_pointers, element_ids, report_cfg
):
    data_array = np.stack(data_matrix, axis=1)
    node_ids_arr = np.array(recorded_node_ids, dtype=np.uint64)
    index_ptr_arr = np.array(index_pointers, dtype=np.uint64)
    element_ids_arr = np.array(element_ids, dtype=np.uint32)
    time_array = np.array([
        report_cfg.get("start_time", 0.0),
        report_cfg.get("end_time", 0.0),
        report_cfg.get("dt", 0.1)
    ], dtype=np.float64)

    with h5py.File(output_path, "w") as f:
        grp = f.require_group(f"/report/{population}")
        data_ds = grp.create_dataset("data", data=data_array.astype(np.float32))
        data_ds.attrs["units"] = "mV"

        mapping = grp.require_group("mapping")
        mapping.create_dataset("node_ids", data=node_ids_arr)
        mapping.create_dataset("index_pointers", data=index_ptr_arr)
        mapping.create_dataset("element_ids", data=element_ids_arr)
        time_ds = mapping.create_dataset("time", data=time_array)
        time_ds.attrs["units"] = "ms"
