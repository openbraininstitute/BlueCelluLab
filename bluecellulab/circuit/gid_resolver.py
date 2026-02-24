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
"""GID resolver for NEURON simulations in BlueCelluLab."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GidNamespace:
    pop_offset: dict[str, int]

    def global_gid(self, pop: str, local_id: int) -> int:
        return int(self.pop_offset[pop]) + int(local_id) + 1  # 1-based indexing to mirror Neurodamus synapse seeding implementation
