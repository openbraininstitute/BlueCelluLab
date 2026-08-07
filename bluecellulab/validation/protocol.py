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
from dataclasses import dataclass, field

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


@dataclass
class SequenceProtocol(Protocol):
    """Multi-phase stimulus protocol composed of sequential steps.

    Each phase is defined by a duration and amplitude (as percentage of rheobase).
    Phases are concatenated in order with a configurable pre-delay and post-delay.

    This enables protocols like rebound bursting (hold -> hyperpolarize -> release)
    or any custom multi-step stimulation sequence.

    Attributes:
        phases: List of (duration_ms, threshold_percentage) tuples. Each phase is
            a step at the given percentage of rheobase for the given duration.
        pre_delay: Delay before the first phase in ms.
        post_delay: Delay after the last phase in ms.
        dt: Time step for the stimulus waveform in ms.
        section: Section to inject into.
        segment: Segment position along the section (0.0 to 1.0).
        add_hypamp: Whether to add the holding current.

    Example:
        Rebound burst protocol (hold 250ms, hyperpolarize 500ms at -200%, release 500ms):

        >>> protocol = SequenceProtocol(
        ...     phases=[(500.0, -200.0), (500.0, 0.0)],
        ...     pre_delay=250.0,
        ...     post_delay=250.0,
        ... )
    """

    phases: list[tuple[float, float]] = field(default_factory=lambda: [(1350.0, 130.0)])
    pre_delay: float = 250.0
    post_delay: float = 250.0
    dt: float = 1.0
    section: str = "soma[0]"
    segment: float = 0.5
    add_hypamp: bool = True

    @property
    def stim_start(self) -> float:
        return self.pre_delay

    @property
    def stim_end(self) -> float:
        total_duration = sum(duration for duration, _ in self.phases)
        return self.pre_delay + total_duration

    def execute(self, template_params, rheobase: float) -> Recording:
        """Build a multi-phase stimulus and record the response."""
        stim_factory = StimulusFactory(dt=self.dt)

        # Build the first phase with the pre_delay
        first_duration, first_pct = self.phases[0]
        first_amplitude = rheobase * first_pct / 100.0
        combined = stim_factory.step(
            pre_delay=self.pre_delay,
            duration=first_duration,
            post_delay=0.0,
            amplitude=first_amplitude,
        )

        # Add subsequent phases
        for i, (duration, pct) in enumerate(self.phases[1:], start=1):
            amplitude = rheobase * pct / 100.0
            is_last = i == len(self.phases) - 1
            post = self.post_delay if is_last else 0.0
            phase_stimulus = stim_factory.step(
                pre_delay=0.0,
                duration=duration,
                post_delay=post,
                amplitude=amplitude,
            )
            combined = combined + phase_stimulus

        # If only one phase, append the post_delay as a zero-amplitude step
        if len(self.phases) == 1:
            post_stimulus = stim_factory.step(
                pre_delay=0.0,
                duration=self.post_delay,
                post_delay=0.0,
                amplitude=0.0,
            )
            combined = combined + post_stimulus

        return run_stimulus(
            template_params,
            combined,
            self.section,
            self.segment,
            add_hypamp=self.add_hypamp,
        )
