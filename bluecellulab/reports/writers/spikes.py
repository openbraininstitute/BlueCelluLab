from pathlib import Path
from typing import Dict
from bluecellulab.reports.writers.base_writer import BaseReportWriter
from bluecellulab.reports.utils import write_sonata_spikes
import logging, os

logger = logging.getLogger(__name__)

class SpikeReportWriter(BaseReportWriter):
    """Writes SONATA spike report from pop→gid→times mapping."""

    def write(self, spikes_by_pop: Dict[str, Dict[int, list]]):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

        for pop, gid_map in spikes_by_pop.items():
            write_sonata_spikes(self.output_path, gid_map, pop)