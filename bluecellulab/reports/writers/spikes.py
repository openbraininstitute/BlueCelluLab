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

from typing import Dict, List
from bluecellulab.reports.writers.base_writer import BaseReportWriter
import logging
import numpy as np
import h5py

logger = logging.getLogger(__name__)


class SpikeReportWriter(BaseReportWriter):
    """Writes SONATA spike report from pop→gid→times mapping."""

    def write(self, spikes_by_pop: Dict[str, Dict[int, list]]):
        if self.output_path.exists():
            self.output_path.unlink()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        for pop, gid_map in spikes_by_pop.items():
            all_node_ids: List[int] = []
            all_timestamps: List[float] = []
            for node_id, times in gid_map.items():
                all_node_ids.extend([node_id] * len(times))
                all_timestamps.extend(times)

            if not all_timestamps:
                logger.warning(f"No spikes to write for population '{pop}'.")

            # Sort by time for consistency
            sorted_indices = np.argsort(all_timestamps)
            node_ids_sorted = np.array(all_node_ids, dtype=np.uint64)[sorted_indices]
            timestamps_sorted = np.array(all_timestamps, dtype=np.float64)[sorted_indices]

            with h5py.File(self.output_path, 'a') as f:
                spikes_group = f.require_group("spikes")
                if pop in spikes_group:
                    logger.warning(f"Overwriting existing group for population '{pop}' in {self.output_path}.")
                    del spikes_group[pop]

                group = spikes_group.create_group(pop)
                sorting_enum = h5py.enum_dtype({'none': 0, 'by_id': 1, 'by_time': 2}, basetype='u1')
                group.attrs.create("sorting", 2, dtype=sorting_enum)  # 2 = by_time

                timestamps_ds = group.create_dataset("timestamps", data=timestamps_sorted)
                group.create_dataset("node_ids", data=node_ids_sorted)

                timestamps_ds.attrs["units"] = "ms"  # SONATA-required
