from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseReportWriter(ABC):
    """Abstract interface for every report writer."""

    def __init__(self, report_cfg: Dict[str, Any], output_path: Path, sim_dt: float):
        self.cfg = report_cfg
        self.output_path = Path(output_path)
        self.sim_dt = sim_dt

    @abstractmethod
    def write(self, data: Dict):
        """Write one report to disk."""
