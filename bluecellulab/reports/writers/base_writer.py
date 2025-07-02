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
