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
from bluecellulab.stimulus.stimulus import DelayedZap, Empty, Ramp, Slope, Step, Stimulus

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

