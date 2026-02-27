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

from typing import Any, Optional, Dict

from bluecellulab.circuit.node_id import CellId
from bluecellulab.reports.writers import get_writer
from bluecellulab.reports.utils import SUPPORTED_REPORT_TYPES, extract_spikes_from_cells  # helper you already have / write

import logging

logger = logging.getLogger(__name__)


class ReportManager:
    """Orchestrates writing all requested SONATA reports."""

    def __init__(self, config, sim_dt: float):
        self.cfg = config
        self.dt = sim_dt

    def write_all(
        self,
        cells: Dict[CellId, Any],
        spikes_by_pop: Optional[Dict[str, Dict[int, list[float]]]] = None,
    ):
        """
        Write all configured reports (compartment and spike) in SONATA format.
        `cells` entries must expose `report_sites` and `get_recording(rec_name)`
        for compartment reports. If `spikes_by_pop` is None, entries must also
        provide `get_recorded_spikes(location=..., threshold=...)`.
        """
        self._write_compartment_reports(cells)
        self._write_spike_report(
            spikes_by_pop or extract_spikes_from_cells(
                cells, location=self.cfg.spike_location, threshold=self.cfg.spike_threshold
            )
        )

    def _write_compartment_reports(self, cells):
        for name, rcfg in self.cfg.get_report_entries().items():
            if rcfg.get("type") not in SUPPORTED_REPORT_TYPES:
                continue

            report_type = rcfg.get("type")
            if report_type == "compartment_set":
                if rcfg.get("cells") is not None:
                    raise ValueError("'cells' may not be set when using 'compartment_set' report type.")
                if rcfg.get("sections") is not None:
                    raise ValueError("'sections' may not be set when using 'compartment_set' report type.")
                if rcfg.get("compartments") is not None:
                    raise ValueError("'compartments' may not be set when using 'compartment_set' report type.")
                src_sets = self.cfg.get_compartment_sets()
            elif report_type == "compartment":
                compartments = rcfg.get("compartments") or "center"
                if compartments not in ("center", "all"):
                    raise ValueError("invalid 'compartments' value")
                if rcfg.get("cells") is None:
                    raise ValueError("'cells' must be specified when using compartment reports")
                src_sets = self.cfg.get_node_sets()
            else:
                logger.error(f"Unsupported report type '{report_type}' in configuration for report '{name}'")

            rcfg["_source_sets"] = src_sets
            rcfg["name"] = name

            out_path = self.cfg.report_file_path(rcfg, name)
            writer = get_writer("compartment")(rcfg, out_path, self.dt)
            writer.write(cells, self.cfg.tstart, self.cfg.tstop)

    def _write_spike_report(self, spikes_by_pop: Dict[str, Dict[int, list[float]]]):
        out_path = self.cfg.spikes_file_path
        writer = get_writer("spikes")({}, out_path, self.dt)
        writer.write(spikes_by_pop)
