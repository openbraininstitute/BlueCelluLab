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

"""Protocols define how to stimulate a cell and obtain a recording."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bluecellulab.analysis.inject_sequence import Recording, run_stimulus
from bluecellulab.stimulus.factory import IDRestTimings, StimulusFactory


class Protocol(ABC):
    """Abstract base for stimulation protocols.

    A protocol injects a stimulus into a cell and returns a recording.
    It also provides timing information for feature extraction.
    """

    @abstractmethod
    def execute(self, template_params, rheobase: float) -> Recording:
        """Run the protocol on a cell.

        Args:
            template_params: BlueCelluLab TemplateParams for creating the cell.
            rheobase: The rheobase (threshold) current in nA.

        Returns:
            A Recording with time, voltage, current, and optionally spikes.
        """

    @property
    @abstractmethod
    def stim_start(self) -> float:
        """Stimulus onset time in ms."""

    @property
    @abstractmethod
    def stim_end(self) -> float:
        """Stimulus offset time in ms."""


@dataclass
class StepProtocol(Protocol):
    """IDRest-style step current injection at a percentage of rheobase.

    Attributes:
        threshold_percentage: Percentage of rheobase to inject (e.g. 130 = 130%).
        dt: Time step for the stimulus waveform in ms.
        section: Section to inject into.
        segment: Segment position along the section (0.0 to 1.0).
        add_hypamp: Whether to add the holding current.
    """

    threshold_percentage: float = 130.0
    dt: float = 1.0
    section: str = "soma[0]"
    segment: float = 0.5
    add_hypamp: bool = True

    @property
    def stim_start(self) -> float:
        return IDRestTimings.PRE_DELAY.value

    @property
    def stim_end(self) -> float:
        return IDRestTimings.PRE_DELAY.value + IDRestTimings.DURATION.value

    def execute(self, template_params, rheobase: float) -> Recording:
        """Inject a step current and record the response."""
        stim_factory = StimulusFactory(dt=self.dt)
        stimulus = stim_factory.idrest(
            threshold_current=rheobase,
            threshold_percentage=self.threshold_percentage,
        )
        return run_stimulus(
            template_params,
            stimulus,
            self.section,
            self.segment,
            add_hypamp=self.add_hypamp,
        )
