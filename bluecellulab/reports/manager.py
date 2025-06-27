from pathlib import Path
from typing import Dict, Any
from bluecellulab.reports.writers import get_writer
from bluecellulab.reports.utils import extract_spikes_from_cells  # helper you already have / write

class ReportManager:
    """Orchestrates writing all requested SONATA reports."""

    def __init__(self, config, sim_dt: float):
        self.cfg  = config
        self.dt   = sim_dt

    def write_all(
        self,
        cells_or_traces: Dict,
        spikes_by_pop:   Dict[str, Dict[int, list]] | None = None,
    ):
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

            rcfg["_source_sets"]  = src_sets
            rcfg["_source_type"]  = src_type

            out_path = self.cfg.report_file_path(rcfg, name)
            writer   = get_writer("compartment")(rcfg, out_path, self.dt)
            writer.write(cells_or_traces)

    def _write_spike_report(self, spikes_by_pop):
        out_path = self.cfg.spikes_file_path
        writer   = get_writer("spikes")({}, out_path, self.dt)
        writer.write(spikes_by_pop)