# Copyright 2023-2024 Blue Brain Project / EPFL
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

from __future__ import annotations
from enum import Enum
from typing import Optional
import logging
from bluecellulab.stimulus.stimulus import DelayedZap, Empty, Ramp, Slope, Step, StepNoise, Stimulus, OrnsteinUhlenbeck, ShotNoise, Sinusoidal, Pulse

logger = logging.getLogger(__name__)


class APWaveformTimings(Enum):
    """APWaveform timings."""
    PRE_DELAY = 250.0
    DURATION = 50.0
    POST_DELAY = 250.0


class IDRestTimings(Enum):
    """IDRest timings."""
    PRE_DELAY = 250.0
    DURATION = 1350.0
    POST_DELAY = 250.0


class IVTimings(Enum):
    """IV timings."""
    PRE_DELAY = 250.0
    DURATION = 3000.0
    POST_DELAY = 250.0


class FirePatternTimings(Enum):
    """FirePattern timings."""
    PRE_DELAY = 250.0
    DURATION = 3600.0
    POST_DELAY = 250.0


class PosCheopsTimings(Enum):
    """PosCheops timings."""
    PRE_DELAY = 250.0
    RAMP1_DURATION = 4000.0
    RAMP2_DURATION = 2000.0
    RAMP3_DURATION = 1333.0
    INTER_DELAY = 2000.0
    POST_DELAY = 250.0


class NegCheopsTimings(Enum):
    """NegCheops timings."""
    PRE_DELAY = 1750.0
    RAMP1_DURATION = 3333.0
    RAMP2_DURATION = 1666.0
    RAMP3_DURATION = 1111.0
    INTER_DELAY = 2000.0
    POST_DELAY = 250.0


class SineSpecTimings(Enum):
    """SineSpec timings."""
    PRE_DELAY = 0
    DURATION = 5000.0
    POST_DELAY = 0


class StimulusFactory:
    def __init__(self, dt: float):
        self.dt = dt

    def step(
        self, pre_delay: float, duration: float, post_delay: float, amplitude: float
    ) -> Stimulus:
        return Step.amplitude_based(
            self.dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )

    def ramp(
        self,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: Optional[float] = None,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 220.0,
    ) -> Stimulus:
        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in ramp."
                    " Will only keep amplitude value."
                )
            return Ramp.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Ramp.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def ap_waveform(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 220.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """Returns the APWaveform Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in ap_waveform."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=APWaveformTimings.PRE_DELAY.value,
                duration=APWaveformTimings.DURATION.value,
                post_delay=APWaveformTimings.POST_DELAY.value,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=APWaveformTimings.PRE_DELAY.value,
                duration=APWaveformTimings.DURATION.value,
                post_delay=APWaveformTimings.POST_DELAY.value,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def idrest(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 200.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """Returns the IDRest Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in idrest."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=IDRestTimings.PRE_DELAY.value,
                duration=IDRestTimings.DURATION.value,
                post_delay=IDRestTimings.POST_DELAY.value,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=IDRestTimings.PRE_DELAY.value,
                duration=IDRestTimings.DURATION.value,
                post_delay=IDRestTimings.POST_DELAY.value,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def iv(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = -40.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """Returns the IV Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in iv."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=IVTimings.PRE_DELAY.value,
                duration=IVTimings.DURATION.value,
                post_delay=IVTimings.POST_DELAY.value,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=IVTimings.PRE_DELAY.value,
                duration=IVTimings.DURATION.value,
                post_delay=IVTimings.POST_DELAY.value,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def fire_pattern(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 200.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """Returns the FirePattern Stimulus object, a type of Step stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in fire_pattern."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=FirePatternTimings.PRE_DELAY.value,
                duration=FirePatternTimings.DURATION.value,
                post_delay=FirePatternTimings.POST_DELAY.value,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=FirePatternTimings.PRE_DELAY.value,
                duration=FirePatternTimings.DURATION.value,
                post_delay=FirePatternTimings.POST_DELAY.value,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def pos_cheops(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 300.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """A combination of pyramid shaped Ramp stimuli with a positive
        amplitude.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is None:
            if threshold_current is None or threshold_current == 0 or threshold_percentage is None:
                raise TypeError("You have to give either threshold_current or amplitude")
            amplitude = threshold_current * threshold_percentage / 100
        elif threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            logger.info(
                "amplitude, threshold_current and threshold_percentage are all set in pos_cheops."
                " Will only keep amplitude value."
            )
        result = (
            Empty(self.dt, duration=PosCheopsTimings.PRE_DELAY.value)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP1_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP1_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=PosCheopsTimings.INTER_DELAY.value)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP2_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP2_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=PosCheopsTimings.INTER_DELAY.value)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP3_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=PosCheopsTimings.RAMP3_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=PosCheopsTimings.POST_DELAY.value)
        )
        return result

    def neg_cheops(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 300.0,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """A combination of pyramid shaped Ramp stimuli with a negative
        amplitude.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
        """

        if amplitude is None:
            if threshold_current is None or threshold_current == 0 or threshold_percentage is None:
                raise TypeError("You have to give either threshold_current or amplitude")
            amplitude = - threshold_current * threshold_percentage / 100
        elif threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            logger.info(
                "amplitude, threshold_current and threshold_percentage are all set in neg_cheops."
                " Will only keep amplitude value."
            )
        result = (
            Empty(self.dt, duration=NegCheopsTimings.PRE_DELAY.value)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP1_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP1_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=NegCheopsTimings.INTER_DELAY.value)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP2_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP2_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=NegCheopsTimings.INTER_DELAY.value)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP3_DURATION.value, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=NegCheopsTimings.RAMP3_DURATION.value, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=NegCheopsTimings.POST_DELAY.value)
        )
        return result

    def sinespec(
        self,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = 60.0,
        amplitude: Optional[float] = None,
        pre_delay: float = 0,
    ) -> Stimulus:
        """Returns the SineSpec Stimulus object, a type of Zap stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
            pre_delay: delay before the start of the stimulus
        """

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in sinespec."
                    " Will only keep amplitude value."
                )
            return DelayedZap.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=SineSpecTimings.DURATION.value,
                post_delay=SineSpecTimings.POST_DELAY.value,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return DelayedZap.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=SineSpecTimings.DURATION.value,
                post_delay=SineSpecTimings.POST_DELAY.value,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def sinusoidal(
        self,
        pre_delay: float,
        post_delay: float,
        duration: float,
        frequency: float,
        amplitude: Optional[float] = None,
        amplitude_percent: Optional[float] = None,
        threshold_current: Optional[float] = None,
        dt: float = 0.025,
    ) -> Stimulus:
        """Creates a Sinusoidal stimulus (factory-compatible).

        Args:
            pre_delay: Delay before the sinusoidal stimulus starts (ms).
            post_delay: Delay after the stimulus ends (ms).
            duration: Duration of the stimulus (ms).
            frequency: Frequency of oscillation (Hz).
            amplitude: Absolute amplitude (nA). Used if provided.
            amplitude_percent: Amplitude as a percentage of threshold current.
            threshold_current: Reference threshold current for percentage-based calculation.
            dt: Time step of the stimulus (ms).

        Returns:
            A `Stimulus` object (Sinusoidal) that can be plotted and injected.

        Notes:
            - If `amplitude` is provided, `amplitude_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        is_amplitude_based = amplitude is not None
        is_threshold_based = (
            threshold_current is not None
            and threshold_current != 0
            and amplitude_percent is not None
        )

        if is_amplitude_based:
            if is_threshold_based:
                logger.info(
                    "amplitude, threshold_current, and amplitude_percent are all set in Sinusoidal."
                    " Using absolute amplitude and ignoring threshold-based parameters."
                )

            return Sinusoidal.amplitude_based(
                dt=dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                frequency=frequency,
                amplitude=amplitude,  # type: ignore[arg-type]
            )

        if is_threshold_based:
            return Sinusoidal.threshold_based(
                dt=dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                frequency=frequency,
                amplitude_percent=amplitude_percent,  # type: ignore[arg-type]
                threshold_current=threshold_current,  # type: ignore[arg-type]
            )

        raise TypeError("You have to provide either `amplitude` or `threshold_current` with `amplitude_percent`.")

    def ornstein_uhlenbeck(
        self,
        pre_delay: float,
        post_delay: float,
        duration: float,
        tau: float,
        sigma: Optional[float] = None,
        mean: Optional[float] = None,
        mean_percent: Optional[float] = None,
        sigma_percent: Optional[float] = None,
        threshold_current: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Stimulus:
        """Creates an Ornstein-Uhlenbeck process stimulus (factory-compatible).

        Args:
            pre_delay: Delay before the noise starts (ms).
            post_delay: Delay after the noise ends (ms).
            duration: Duration of the stimulus (ms).
            tau: Time constant of the noise process.
            sigma: Standard deviation of the noise (used when mean is provided).
            mean: Absolute mean current value (used if provided).
            mean_percent: Mean current as a percentage of threshold current (used if mean is None).
            sigma_percent: Standard deviation as a percentage of threshold current (used if sigma is None).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object (OrnsteinUhlenbeck) that can be plotted and injected.

        Notes:
            - If `mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        is_amplitude_based = mean is not None and sigma is not None
        is_threshold_based = (
            threshold_current is not None
            and threshold_current != 0
            and mean_percent is not None
            and sigma_percent is not None
        )

        if is_amplitude_based:
            if is_threshold_based:
                logger.info(
                    "mean, threshold_current, and mean_percent are all set in Ornstein-Uhlenbeck."
                    " Using mean and ignoring threshold-based parameters."
                )

            return OrnsteinUhlenbeck.amplitude_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                tau=tau,
                sigma=sigma,  # type: ignore[arg-type]
                mean=mean,  # type: ignore[arg-type]
                seed=seed
            )

        if is_threshold_based:
            return OrnsteinUhlenbeck.threshold_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                mean_percent=mean_percent,  # type: ignore[arg-type]
                sigma_percent=sigma_percent,  # type: ignore[arg-type]
                threshold_current=threshold_current,  # type: ignore[arg-type]
                tau=tau,
                seed=seed
            )

        raise TypeError("You have to give either `mean` and `sigma` or `threshold_current` and `mean_percent` and `sigma_percent`.")

    def shot_noise(
        self,
        pre_delay: float,
        post_delay: float,
        duration: float,
        rate: float,
        rise_time: float,
        decay_time: float,
        mean: Optional[float] = None,
        sigma: Optional[float] = None,
        mean_percent: Optional[float] = None,
        sigma_percent: Optional[float] = None,
        relative_skew: float = 0.5,
        threshold_current: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Stimulus:
        """Creates a ShotNoise instance, either with an absolute amplitude or
        relative to a threshold current.

        Args:
            pre_delay: Delay before the noise starts (ms).
            post_delay: Delay after the noise ends (ms).
            duration: Duration of the stimulus (ms).
            rate: Mean rate of synaptic events (Hz).
            mean: Mean amplitude of events (nA), used if provided.
            sigma: Standard deviation of event amplitudes.
            rise_time: Rise time of synaptic events (ms).
            decay_time: Decay time of synaptic events (ms).
            mean_percent: Mean current as a percentage of threshold current (used if mean is None).
            sigma_percent: Standard deviation as a percentage of threshold current (used if sigma is None).
            relative_skew: Skew factor for the shot noise process (default: 0.5).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object that can be plotted and injected.

        Notes:
            - If `mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        is_amplitude_based = mean is not None and sigma is not None
        is_threshold_based = (
            threshold_current is not None
            and threshold_current != 0
            and mean_percent is not None
            and sigma_percent is not None
        )

        if is_amplitude_based:
            if is_threshold_based:
                logger.info(
                    "mean, threshold_current, and mean_percent are all set in ShotNoise."
                    " Using mean and ignoring threshold-based parameters."
                )

            return ShotNoise.amplitude_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                rate=rate,
                mean=mean,  # type: ignore[arg-type]
                sigma=sigma,  # type: ignore[arg-type]
                rise_time=rise_time,
                decay_time=decay_time,
                seed=seed
            )

        if is_threshold_based:
            return ShotNoise.threshold_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                rise_time=rise_time,
                decay_time=decay_time,
                mean_percent=mean_percent,  # type: ignore[arg-type]
                sigma_percent=sigma_percent,  # type: ignore[arg-type]
                threshold_current=threshold_current,  # type: ignore[arg-type]
                relative_skew=relative_skew,
                seed=seed
            )

        raise TypeError("You must provide either `mean` and `sigma`, or `threshold_current` and `mean_percent` and `sigma_percent` with percentage values.")

    def step_noise(
        self,
        pre_delay: float,
        post_delay: float,
        duration: float,
        step_interval: float,
        mean: Optional[float] = None,
        sigma: Optional[float] = None,
        mean_percent: Optional[float] = None,
        sigma_percent: Optional[float] = None,
        threshold_current: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Stimulus:
        """Creates a StepNoise instance, either with an absolute amplitude or
        relative to a threshold current.

        Args:
            pre_delay: Delay before the step noise starts (ms).
            post_delay: Delay after the step noise ends (ms).
            duration: Duration of the stimulus (ms).
            step_interval: Interval at which noise amplitude changes.
            mean: Mean amplitude of step noise (nA), used if provided.
            sigma: Standard deviation of step noise.
            mean_percent: Mean current as a percentage of threshold current (used if mean is None).
            sigma_percent: Standard deviation as a percentage of threshold current (used if sigma is None).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object that can be plotted and injected.

        Notes:
            - If `mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        is_amplitude_based = mean is not None and sigma is not None
        is_threshold_based = (
            threshold_current is not None
            and threshold_current != 0
            and mean_percent is not None
            and sigma_percent is not None
        )

        if is_amplitude_based:
            if is_threshold_based:
                logger.info(
                    "mean, threshold_current, and mean_percent are all set in StepNoise."
                    " Using mean and ignoring threshold-based parameters."
                )
            return StepNoise.amplitude_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                step_interval=step_interval,
                mean=mean,  # type: ignore[arg-type]
                sigma=sigma,  # type: ignore[arg-type]
                seed=seed,
            )

        if is_threshold_based:
            return StepNoise.threshold_based(
                dt=self.dt,
                pre_delay=pre_delay,
                post_delay=post_delay,
                duration=duration,
                step_interval=step_interval,
                mean_percent=mean_percent,  # type: ignore[arg-type]
                sigma_percent=sigma_percent,  # type: ignore[arg-type]
                threshold_current=threshold_current,  # type: ignore[arg-type]
                seed=seed,
            )

        raise TypeError("You must provide either `mean` and `sigma`, or `threshold_current` and `mean_percent` and `sigma_percent`  with percentage values.")

    def pulse(
        self,
        pre_delay: float,
        duration: float,
        post_delay: float,
        frequency: float,
        width: float,
        threshold_current: Optional[float] = None,
        threshold_percentage: Optional[float] = None,
        amplitude: Optional[float] = None,
    ) -> Stimulus:
        """Creates a pulse stimulus.

        Args:
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
            amplitude: Raw amplitude of input current.
            pre_delay: Delay before the first pulse (ms).
            duration: Duration of the pulse train (ms).
            post_delay: Delay after the last pulse (ms).
            frequency: Frequency of the pulses (Hz).
            width: Width of each pulse (ms).
        """
        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in pulse."
                    " Will only keep amplitude value."
                )
            return Pulse.amplitude_based(self.dt, pre_delay, duration, post_delay, amplitude, frequency, width)

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Pulse.threshold_based(self.dt, pre_delay, duration, post_delay, threshold_current, threshold_percentage, frequency, width)

        raise TypeError("You have to give either threshold_current or amplitude")
