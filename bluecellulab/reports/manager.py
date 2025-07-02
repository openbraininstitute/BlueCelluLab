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

from typing import Optional, Dict
from bluecellulab.reports.writers import get_writer
from bluecellulab.reports.utils import extract_spikes_from_cells  # helper you already have / write


class ReportManager:
    """Orchestrates writing all requested SONATA reports."""

    def __init__(self, config, sim_dt: float):
        self.cfg = config
        self.dt = sim_dt

    def write_all(
        self,
        cells_or_traces: Dict,
        spikes_by_pop: Optional[Dict[str, Dict[int, list]]] = None,
    ):
        """Write all configured reports (compartment and spike) in SONATA
        format.

        Parameters
        ----------
        cells_or_traces : dict
            A dictionary mapping (population, gid) to either:
            - Cell objects with recorded data (used in single-process simulations), or
            - Precomputed trace dictionaries, e.g., {"voltage": ndarray}, typically gathered across ranks in parallel runs.

        spikes_by_pop : dict, optional
            A precomputed dictionary of spike times by population.
            If not provided, spike times are extracted from `cells_or_traces`.

        Notes
        -----
        In parallel simulations, you must gather all traces and spikes to rank 0 and pass them here.
        """
        self._write_voltage_reports(cells_or_traces)
        self._write_spike_report(spikes_by_pop or extract_spikes_from_cells(cells_or_traces, location=self.cfg.spike_location, threshold=self.cfg.spike_threshold))

    def _write_voltage_reports(self, cells_or_traces):
        for name, rcfg in self.cfg.get_report_entries().items():
            if rcfg.get("type") != "compartment":
                continue

            section = rcfg.get("sections")
            if section == "compartment_set":
                if rcfg.get("cells") is not None:
                    raise ValueError("'cells' may not be set with 'compartment_set'")
                src_sets, src_type = self.cfg.get_compartment_sets(), "compartment_set"
            else:
                if rcfg.get("compartments") not in ("center", "all"):
                    raise ValueError("invalid 'compartments' value")
                src_sets, src_type = self.cfg.get_node_sets(), "node_set"

            rcfg["_source_sets"] = src_sets
            rcfg["_source_type"] = src_type

            out_path = self.cfg.report_file_path(rcfg, name)
            writer = get_writer("compartment")(rcfg, out_path, self.dt)
            writer.write(cells_or_traces, self.cfg.tstart)

    def _write_spike_report(self, spikes_by_pop):
        out_path = self.cfg.spikes_file_path
        writer = get_writer("spikes")({}, out_path, self.dt)
        writer.write(spikes_by_pop)
