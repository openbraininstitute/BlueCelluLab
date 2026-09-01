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
from math import ceil, isfinite
from numbers import Integral, Real

from bluecellulab.analysis.inject_sequence import Recording, run_stimulus
from bluecellulab.stimulus.factory import IDRestTimings, StimulusFactory


def _validate_real(name: str, value: object) -> float:
    """Return ``value`` as a float, rejecting non-real and non-finite input."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number.")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number.")
    return numeric


def _validate_time_value(
    name: str, value: object, strictly_positive: bool = False
) -> float:
    """Return ``value`` as a float time in ms, rejecting invalid durations."""
    numeric = _validate_real(name, value)
    if strictly_positive and numeric <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if not strictly_positive and numeric < 0:
        raise ValueError(f"{name} must not be negative.")
    return numeric


def _sample_count(duration: float, dt: float) -> int:
    """Return the sample count the stimulus factory uses for a duration.

    The stimulus waveforms are built with ``numpy.arange(0.0, duration, dt)``,
    so a duration that is not an exact multiple of ``dt`` is rounded up to the
    next whole sample.
    """
    return ceil(float(duration) / float(dt))


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

    Timing windows are aligned to the stimulus sample grid, so they match the
    generated waveform even when the fixed IDRest timings are not exact
    multiples of ``dt``.

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

    def __post_init__(self) -> None:
        """Validate the waveform time step before simulating."""
        _validate_time_value("dt", self.dt, strictly_positive=True)
        _validate_real("threshold_percentage", self.threshold_percentage)

    @property
    def stim_start(self) -> float:
        return _sample_count(IDRestTimings.PRE_DELAY.value, self.dt) * float(self.dt)

    @property
    def stim_end(self) -> float:
        samples = _sample_count(IDRestTimings.PRE_DELAY.value, self.dt) + _sample_count(
            IDRestTimings.DURATION.value, self.dt
        )
        return samples * float(self.dt)

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

    Each phase is defined by a duration and amplitude. By default, amplitudes are
    interpreted as percentages of rheobase. Set `absolute_amplitudes=True` to use
    raw current values in nA instead.

    Phases are concatenated in order with a configurable pre-delay and post-delay.
    Timing windows are aligned to the stimulus sample grid, so they match the
    waveform even when a requested duration is not an exact multiple of ``dt``.

    This enables protocols like rebound bursting (hold -> hyperpolarize -> release)
    or any custom multi-step stimulation sequence.

    Attributes:
        phases: List of (duration_ms, amplitude) tuples. Amplitude meaning depends
            on `absolute_amplitudes`.
        pre_delay: Delay before the first phase in ms.
        post_delay: Delay after the last phase in ms.
        dt: Time step for the stimulus waveform in ms.
        section: Section to inject into.
        segment: Segment position along the section (0.0 to 1.0).
        add_hypamp: Whether to add the holding current.
        absolute_amplitudes: If True, phase amplitudes are in nA. If False (default),
            they are percentages of rheobase.
        measurement_phase: If set, stim_start/stim_end cover only this phase index
            (0-based). Useful when feature extraction should target a specific phase
            (e.g. the release window in a rebound burst protocol).

    Example:
        Rebound burst protocol with absolute currents, measuring spikes in phase 2:

        >>> protocol = SequenceProtocol(
        ...     phases=[(250.0, 0.087), (500.0, -0.145), (1000.0, 0.087)],
        ...     pre_delay=0.0,
        ...     post_delay=250.0,
        ...     absolute_amplitudes=True,
        ...     measurement_phase=2,
        ... )
    """

    phases: list[tuple[float, float]] = field(default_factory=lambda: [(1350.0, 130.0)])
    pre_delay: float = 250.0
    post_delay: float = 250.0
    dt: float = 1.0
    section: str = "soma[0]"
    segment: float = 0.5
    add_hypamp: bool = True
    absolute_amplitudes: bool = False
    measurement_phase: int | None = None

    def __post_init__(self) -> None:
        """Validate phase definitions before any simulation is attempted."""
        _validate_time_value("dt", self.dt, strictly_positive=True)
        _validate_time_value("pre_delay", self.pre_delay)
        _validate_time_value("post_delay", self.post_delay)

        if not isinstance(self.phases, (list, tuple)) or not self.phases:
            raise ValueError(
                "phases must contain at least one (duration, amplitude) pair."
            )

        for index, phase in enumerate(self.phases):
            if not isinstance(phase, (list, tuple)) or len(phase) != 2:
                raise ValueError(
                    f"phases[{index}] must be a (duration, amplitude) pair."
                )
            duration, amplitude = phase
            _validate_time_value(
                f"phases[{index}].duration", duration, strictly_positive=True
            )
            _validate_real(f"phases[{index}].amplitude", amplitude)

        if self.measurement_phase is not None:
            if isinstance(self.measurement_phase, bool) or not isinstance(
                self.measurement_phase, Integral
            ):
                raise ValueError("measurement_phase must be an integer or None.")
            if not 0 <= self.measurement_phase < len(self.phases):
                raise IndexError(
                    f"measurement_phase={self.measurement_phase} is out of range "
                    f"for {len(self.phases)} phase(s)."
                )

    def _sample_count(self, duration: float) -> int:
        """Return the sample count the stimulus uses for a duration."""
        return _sample_count(duration, self.dt)

    def _phase_window(self) -> tuple[float, float]:
        """Return the sample-grid-aligned feature extraction window in ms.

        If measurement_phase is set, the window covers only that phase.
        Otherwise it covers the full stimulus (first phase start to last
        phase end). The post-delay is intentionally excluded.
        """
        start_samples = self._sample_count(self.pre_delay)
        if self.measurement_phase is not None:
            start_samples += sum(
                self._sample_count(duration)
                for duration, _ in self.phases[: self.measurement_phase]
            )
            end_samples = start_samples + self._sample_count(
                self.phases[self.measurement_phase][0]
            )
        else:
            end_samples = start_samples + sum(
                self._sample_count(duration) for duration, _ in self.phases
            )

        dt = float(self.dt)
        return start_samples * dt, end_samples * dt

    @property
    def stim_start(self) -> float:
        return self._phase_window()[0]

    @property
    def stim_end(self) -> float:
        return self._phase_window()[1]

    def _compute_amplitude(self, amplitude_value: float, rheobase: float) -> float:
        """Convert phase amplitude to nA based on amplitude mode."""
        if self.absolute_amplitudes:
            return amplitude_value
        return rheobase * amplitude_value / 100.0

    def execute(self, template_params, rheobase: float) -> Recording:
        """Build a multi-phase stimulus and record the response."""
        stim_factory = StimulusFactory(dt=self.dt)

        # Build the first phase with the pre_delay
        first_duration, first_amp = self.phases[0]
        first_amplitude = self._compute_amplitude(first_amp, rheobase)
        combined = stim_factory.step(
            pre_delay=self.pre_delay,
            duration=first_duration,
            post_delay=0.0,
            amplitude=first_amplitude,
        )

        # Add subsequent phases
        for i, (duration, amp) in enumerate(self.phases[1:], start=1):
            amplitude = self._compute_amplitude(amp, rheobase)
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
