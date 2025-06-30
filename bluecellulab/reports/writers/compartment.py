import numpy as np
from typing import Dict, List
from .base_writer import BaseReportWriter
from bluecellulab.reports.utils import (
    resolve_source_nodes,
    resolve_segments,
    write_sonata_report_file,
)
import logging

logger = logging.getLogger(__name__)


class CompartmentReportWriter(BaseReportWriter):
    """Writes SONATA compartment (voltage) reports."""

    def write(self, cells: Dict):
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

        write_sonata_report_file(
            self.output_path,
            population,
            data_matrix,
            node_id_list,
            idx_ptr,
            elem_ids,
            self.cfg,
            self.sim_dt,
        )
