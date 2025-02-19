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
from typing import Optional
import logging
from bluecellulab.stimulus.stimulus import DelayedZap, Empty, Ramp, Slope, Step, StepNoise, Stimulus, OrnsteinUhlenbeck, ShotNoise
from bluecellulab.stimulus.circuit_stimulus_definitions import Stimulus as CircuitStimulus

logger = logging.getLogger(__name__)


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
        self, pre_delay: float, duration: float, post_delay: float, amplitude: float,
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
        pre_delay = 250.0
        duration = 50.0
        post_delay = 250.0

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in ap_waveform."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
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
        pre_delay = 250.0
        duration = 1350.0
        post_delay = 250.0

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in idrest."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
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
        pre_delay = 250.0
        duration = 3000.0
        post_delay = 250.0

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in iv."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
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
        pre_delay = 250.0
        duration = 3600.0
        post_delay = 250.0

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in fire_pattern."
                    " Will only keep amplitude value."
                )
            return Step.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return Step.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
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
        delay = 250.0
        ramp1_duration = 4000.0
        ramp2_duration = 2000.0
        ramp3_duration = 1333.0
        inter_delay = 2000.0
        post_delay = 250.0

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
            Empty(self.dt, duration=delay)
            + Slope(self.dt, duration=ramp1_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp1_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=inter_delay)
            + Slope(self.dt, duration=ramp2_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp2_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=inter_delay)
            + Slope(self.dt, duration=ramp3_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp3_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=post_delay)
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
        delay = 1750.0
        ramp1_duration = 3333.0
        ramp2_duration = 1666.0
        ramp3_duration = 1111.0
        inter_delay = 2000.0
        post_delay = 250.0

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
            Empty(self.dt, duration=delay)
            + Slope(self.dt, duration=ramp1_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp1_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=inter_delay)
            + Slope(self.dt, duration=ramp2_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp2_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=inter_delay)
            + Slope(self.dt, duration=ramp3_duration, amplitude_start=0.0, amplitude_end=amplitude)
            + Slope(self.dt, duration=ramp3_duration, amplitude_start=amplitude, amplitude_end=0.0)
            + Empty(self.dt, duration=post_delay)
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
        duration = 5000.0
        post_delay = 0

        if amplitude is not None:
            if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
                logger.info(
                    "amplitude, threshold_current and threshold_percentage are all set in sinespec."
                    " Will only keep amplitude value."
                )
            return DelayedZap.amplitude_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                amplitude=amplitude,
            )

        if threshold_current is not None and threshold_current != 0 and threshold_percentage is not None:
            return DelayedZap.threshold_based(
                self.dt,
                pre_delay=pre_delay,
                duration=duration,
                post_delay=post_delay,
                threshold_current=threshold_current,
                threshold_percentage=threshold_percentage,
            )

        raise TypeError("You have to give either threshold_current or amplitude")

    def ornstein_uhlenbeck(
        self,
        duration: float,
        tau: float,
        sigma: Optional[float] = None,
        mean: Optional[float] = None,
        mean_percent: Optional[float] = None,
        sd_percent: Optional[float] = None,
        threshold_current: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Stimulus:
        """Creates an Ornstein-Uhlenbeck process stimulus (factory-compatible).

        Args:
            duration: Duration of the stimulus (ms).
            tau: Time constant of the noise process.
            sigma: Standard deviation of the noise (used when mean is provided).
            mean: Absolute mean current value (used if provided).
            mean_percent: Mean current as a percentage of threshold current (used if mean is None).
            sd_percent: Standard deviation as a percentage of threshold current (used if sigma is None).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object (OrnsteinUhlenbeck) that can be plotted and injected.

        Notes:
            - If `mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        if mean is not None:
            if threshold_current is not None and threshold_current != 0 and mean_percent is not None:
                logger.info(
                    "amplitude, threshold_current and mean_percent are all set in Ornstein-Uhlenbeck."
                    " Will only keep amplitude value."
                )
            return OrnsteinUhlenbeck.amplitude_based(self.dt, duration, tau, sigma, mean, seed)

        if threshold_current is not None and threshold_current != 0 and mean_percent is not None and sd_percent is not None:
            return OrnsteinUhlenbeck.threshold_based(self.dt, duration, mean_percent, sd_percent, threshold_current, tau, seed)

        raise TypeError("You have to give either threshold_current or amplitude")

    def shot_noise(
            self, duration: float,
            rate: float,
            rise_time: float,
            decay_time: float,
            amp_mean: Optional[float] = None,
            amp_var: Optional[float] = None,
            mean_percent: Optional[float] = None,
            sd_percent: Optional[float] = None,
            relative_skew: Optional[float] = 0.5,
            threshold_current: Optional[float] = None,
            seed: Optional[int] = None
    ) -> Stimulus:
        """Creates a ShotNoise instance, either with an absolute amplitude or
        relative to a threshold current.

        Args:
            duration: Duration of the stimulus (ms).
            rate: Mean rate of synaptic events (Hz).
            amp_mean: Mean amplitude of events (nA), used if provided.
            amp_var: Variance in amplitude of events.
            rise_time: Rise time of synaptic events (ms).
            decay_time: Decay time of synaptic events (ms).
            mean_percent: Mean current as a percentage of threshold current (used if amp_mean is None).
            sd_percent: Standard deviation as a percentage of threshold current (used if amp_var is None).
            relative_skew: Skew factor for the shot noise process (default: 0.5).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object that can be plotted and injected.

        Notes:
            - If `amp_mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        if amp_mean is not None:
            if threshold_current is not None and threshold_current != 0 and mean_percent is not None:
                logger.info(
                    "amplitude, threshold_current and mean_percent are all set in Ornstein-Uhlenbeck."
                    " Will only keep amplitude value."
                )
            return ShotNoise.amplitude_based(self.dt, duration, amp_mean, rate, amp_var, rise_time, decay_time, seed)

        if threshold_current is not None and threshold_current != 0 and mean_percent is not None and sd_percent is not None:
            return ShotNoise.threshold_based(self.dt, duration, rise_time, decay_time, mean_percent, sd_percent, threshold_current, relative_skew, seed)

        raise TypeError("You have to give either threshold_current or amplitude")

    def step_noise(
        self,
        duration: float,
        step_duration: float,
        mean: Optional[float] = None,
        variance: Optional[float] = None,
        mean_percent: Optional[float] = None,
        sd_percent: Optional[float] = None,
        threshold_current: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Stimulus:
        """Creates a StepNoise instance, either with an absolute amplitude or
        relative to a threshold current.

        Args:
            duration: Duration of the stimulus (ms).
            step_duration: Duration of each step before noise changes (ms).
            mean: Mean amplitude of step noise (nA), used if provided.
            variance: Variance of step noise.
            mean_percent: Mean current as a percentage of threshold current (used if mean is None).
            sd_percent: Standard deviation as a percentage of threshold current (used if variance is None).
            threshold_current: Reference threshold current for percentage-based calculation.
            seed: Optional random seed for reproducibility.

        Returns:
            A `Stimulus` object that can be plotted and injected.

        Notes:
            - If `mean` is provided, `mean_percent` is ignored.
            - If `threshold_current` is not provided, threshold-based parameters cannot be used.
        """
        if mean is not None:
            if threshold_current is not None and threshold_current != 0 and mean_percent is not None:
                logger.info(
                    "amplitude, threshold_current and mean_percent are all set in StepNoise."
                    " Will only keep amplitude value."
                )
            return StepNoise.amplitude_based(self.dt, duration, step_duration, mean, variance, seed)

        if threshold_current is not None and threshold_current != 0 and mean_percent is not None and sd_percent is not None:
            return StepNoise.threshold_based(self.dt, duration, step_duration, mean_percent, sd_percent, threshold_current, seed)

        raise TypeError("You have to give either threshold_current or mean.")

    def from_sonata(cls, circuit_stimulus: CircuitStimulus):
        """Convert a SONATA stimulus into a factory-based stimulus."""
        raise ValueError(f"Unsupported circuit stimulus type: {type(circuit_stimulus)}")
