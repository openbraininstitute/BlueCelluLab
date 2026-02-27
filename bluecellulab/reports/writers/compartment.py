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
    build_recording_sites,
    resolve_source_nodes,
)
import logging

logger = logging.getLogger(__name__)


class CompartmentReportWriter(BaseReportWriter):
    """Writes SONATA compartment reports."""

    def write(self, cells: Dict, tstart=0, tstop=None):
        report_name = self.cfg.get("name", "unnamed")

        # Resolve which population this report targets (for H5 group path)
        report_type = self.cfg.get("type", "compartment")
        source_sets = self.cfg["_source_sets"]
        if report_type == "compartment":
            src_name = self.cfg.get("cells")
        elif report_type == "compartment_set":
            src_name = self.cfg.get("compartment_set")
        else:
            raise NotImplementedError(f"Unsupported report type '{report_type}' for '{report_name}'")

        src = source_sets.get(src_name)
        if not src:
            logger.warning("%s '%s' not found – skipping '%s'.", report_type, src_name, report_name)
            return

        population = src["population"]

        data_matrix: List[np.ndarray] = []
        node_id_list: List[int] = []
        idx_ptr: List[int] = [0]
        elem_ids: List[int] = []

        # Iterate cells belonging to this population only
        pop_cells = [(gid, cell) for (pop, gid), cell in cells.items() if pop == population]
        if not pop_cells:
            logger.warning("No cells found for population '%s' – skipping '%s'.", population, report_name)
            return

        for gid, cell in sorted(pop_cells, key=lambda x: x[0]):
            sites = getattr(cell, "report_sites", {}).get(report_name, [])
            if not sites:
                continue

            for site in sites:
                rec_name = site["rec_name"]
                try:
                    trace = cell.get_recording(rec_name)
                except Exception as e:
                    logger.warning("Missing recording '%s' for (%s,%d) in '%s': %s",
                                   rec_name, population, gid, report_name, e)
                    continue

                data_matrix.append(np.asarray(trace, dtype=np.float32))
                node_id_list.append(gid)
                elem_ids.append(len(elem_ids))
                idx_ptr.append(idx_ptr[-1] + 1)

        if not data_matrix:
            logger.warning("No data for report '%s'.", report_name)
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
            tstart,
            tstop,
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
        tstart,
        tstop
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

        tstop : float
            Simulation end time (ms).
        """
        start_time = float(report_cfg.get("start_time", 0.0))
        end_time = float(report_cfg.get("end_time", 0.0))
        dt_report = float(report_cfg.get("dt", sim_dt))

        if tstop is not None and end_time > tstop:
            end_time = tstop

        # Clamp dt_report if finer than simuldation dt
        if dt_report < sim_dt:
            logger.warning(
                f"Requested report dt={dt_report} ms is finer than simulation dt={sim_dt} ms. "
                f"Clamping report dt to {sim_dt} ms. "
                f"To achieve finer temporal resolution, reduce the simulation dt in your config."
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
        end_index = int(round((end_time - tstart) / sim_dt))

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

            units = report_cfg.get("unit")
            if units is None:
                units = "mV" if variable == "v" else "unknown"

            data_ds.attrs["units"] = str(units)

            mapping = grp.require_group("mapping")
            mapping.create_dataset("node_ids", data=node_ids_arr)
            mapping.create_dataset("index_pointers", data=index_ptr_arr)
            mapping.create_dataset("element_ids", data=element_ids_arr)
            time_ds = mapping.create_dataset("time", data=time_array)
            time_ds.attrs["units"] = "ms"
