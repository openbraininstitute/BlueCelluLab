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

from pathlib import Path
import numpy as np
import h5py
from typing import Dict, List
from .base_writer import BaseReportWriter
from bluecellulab.reports.utils import (
    resolve_source_nodes,
    resolve_segments,
)
import logging

logger = logging.getLogger(__name__)


class CompartmentReportWriter(BaseReportWriter):
    """Writes SONATA compartment (voltage) reports."""

    def write(self, cells: Dict, tstart=0):
        report_name = self.cfg.get("name", "unnamed")
        # section     = self.cfg.get("sections")
        variable = self.cfg.get("variable_name", "v")

        source_sets = self.cfg["_source_sets"]
        source_type = self.cfg["_source_type"]
        src_name = self.cfg.get("cells") if source_type == "node_set" else self.cfg.get("compartments")
        src = source_sets.get(src_name)
        if not src:
            logger.warning(f"{source_type.title()} '{src_name}' not found – skipping '{report_name}'.")
            return

        population = src["population"]
        node_ids, comp_nodes = resolve_source_nodes(src, source_type, cells, population)

        data_matrix: List[np.ndarray] = []
        node_id_list: List[int] = []
        idx_ptr: List[int] = [0]
        elem_ids: List[int] = []

        for nid in node_ids:
            cell = cells.get((population, nid)) or cells.get(f"{population}_{nid}")
            if cell is None:
                continue

            if isinstance(cell, dict):
                # No section/segment structure to resolve for traces
                trace = np.asarray(cell["voltage"], dtype=np.float32)
                data_matrix.append(trace)
                node_id_list.append(nid)
                elem_ids.append(len(elem_ids))
                idx_ptr.append(idx_ptr[-1] + 1)
                continue

            targets = resolve_segments(cell, self.cfg, nid, comp_nodes, source_type)
            for sec, sec_name, seg in targets:
                try:
                    if hasattr(cell, "get_variable_recording"):
                        trace = cell.get_variable_recording(variable=variable, section=sec, segx=seg)
                    else:
                        trace = np.asarray(cell["voltage"], dtype=np.float32)
                    data_matrix.append(trace)
                    node_id_list.append(nid)
                    elem_ids.append(len(elem_ids))
                    idx_ptr.append(idx_ptr[-1] + 1)
                except Exception as e:
                    logger.warning(f"Failed recording {nid}:{sec_name}@{seg}: {e}")

        if not data_matrix:
            logger.warning(f"No data for report '{report_name}'.")
            return

        self._write_sonata_report_file(
            self.output_path,
            population,
            data_matrix,
            node_id_list,
            idx_ptr,
            elem_ids,
            self.cfg,
            self.sim_dt,
            tstart
        )

    def _write_sonata_report_file(
        self,
        output_path,
        population,
        data_matrix,
        recorded_node_ids,
        index_pointers,
        element_ids,
        report_cfg,
        sim_dt,
        tstart
    ):
        """Write a SONATA HDF5 report file containing time series data.

        This function downsamples the data if needed, prepares metadata arrays,
        and writes the report in SONATA format to the specified HDF5 file.

        Parameters
        ----------
        output_path : str or Path
            Destination path of the report file.

        population : str
            Name of the population being recorded.

        data_matrix : list of ndarray
            List of arrays containing recorded time series per element.

        recorded_node_ids : list of int
            Node IDs corresponding to the recorded traces.

        index_pointers : list of int
            Index pointers mapping node IDs to data.

        element_ids : list of int
            Element IDs (e.g., segment IDs) corresponding to each trace.

        report_cfg : dict
            Report configuration specifying time window and variable name.

        sim_dt : float
            Simulation timestep (ms).

        tstart : float
            Simulation start time (ms).
        """
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
        start_index = int(round((start_time - tstart) / sim_dt))
        end_index = int(round((end_time - tstart) / sim_dt)) + 1  # inclusive

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
