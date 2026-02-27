# Copyright 2026 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, TypeAlias

# Keep section as Any here: NeuronSection is currently a runtime alias to
# NEURON's hoc object, and using `NeuronSection | None` in a TypeAlias is
# evaluated at import time.
ReportSite: TypeAlias = tuple[Any, str, float]


class ReportSiteResolvable(Protocol):
    """Object able to resolve recording locations for a SONATA report.

    Implemented by instantiated Cell objects during simulation setup.
    """

    def resolve_segments_from_config(
        self,
        report_cfg: dict
    ) -> list[ReportSite]:
        ...

    def resolve_segments_from_compartment_set(
        self,
        node_id: int,
        compartment_nodes: list
    ) -> list[ReportSite]:
        ...


class ReportConfigurableCell(ReportSiteResolvable, Protocol):
    """Cell-like object that can configure recordings from resolved sites."""

    def configure_recording(
        self,
        recording_sites: Iterable[ReportSite],
        variable_name: str,
        report_name: str,
    ) -> list[str]:
        ...


class SpikeExtractableCell(Protocol):
    """Cell-like object that can return recorded spike times."""

    def get_recorded_spikes(
        self,
        location: str = "soma",
        threshold: float = -20.0,
    ) -> Any:
        ...
