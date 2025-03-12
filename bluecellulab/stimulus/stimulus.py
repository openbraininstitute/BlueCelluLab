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
from abc import ABC, abstractmethod
from typing import Optional
import logging
import matplotlib.pyplot as plt
import neuron
import numpy as np
from bluecellulab.cell.stimuli_generator import get_relative_shotnoise_params
from bluecellulab.exceptions import BluecellulabError

logger = logging.getLogger(__name__)


class Stimulus(ABC):
    def __init__(self, dt: float) -> None:
        self.dt = dt

    @property
    @abstractmethod
    def time(self) -> np.ndarray:
        """Time values of the stimulus."""
        ...

    @property
    @abstractmethod
    def current(self) -> np.ndarray:
        """Current values of the stimulus."""
        ...

    def __len__(self) -> int:
        return len(self.time)

    @property
    def stimulus_time(self) -> float:
        return len(self) * self.dt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dt={self.dt})"

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.time, self.current, **kwargs)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (nA)")
        return ax

    def __add__(self, other: Stimulus) -> CombinedStimulus:
        """Override + operator to concatenate Stimulus objects."""
        if self.dt != other.dt:
            raise ValueError(
                "Stimulus objects must have the same dt to be concatenated"
            )
        if len(self.time) == 0:
            return CombinedStimulus(other.dt, other.time, other.current)
        elif len(other.time) == 0:
            return CombinedStimulus(self.dt, self.time, self.current)
        else:
            # shift other time
            other_time = other.time + self.time[-1] + self.dt
            combined_time = np.concatenate([self.time, other_time])
            # Concatenate the current arrays
            combined_current = np.concatenate([self.current, other.current])
            return CombinedStimulus(self.dt, combined_time, combined_current)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stimulus):
            return NotImplemented
        else:
            return (
                np.allclose(self.time, other.time)
                and np.allclose(self.current, other.current)
                and self.dt == other.dt
            )


class CombinedStimulus(Stimulus):
    """Represents the Stimulus created by combining multiple stimuli."""

    def __init__(self, dt: float, time: np.ndarray, current: np.ndarray) -> None:
        super().__init__(dt)
        self._time = time
        self._current = current

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current


class Empty(Stimulus):
    """Represents empty stimulus (all zeros) that has no impact on the cell.

    This is required by some Stimuli that expect the cell to rest.
    """

    def __init__(self, dt: float, duration: float) -> None:
        super().__init__(dt)
        self.duration = duration

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.zeros_like(self.time)


class Flat(Stimulus):
    def __init__(self, dt: float, duration: float, amplitude: float) -> None:
        super().__init__(dt)
        self.duration = duration
        self.amplitude = amplitude

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.full_like(self.time, self.amplitude)


class Slope(Stimulus):
    def __init__(
        self, dt: float, duration: float, amplitude_start: float, amplitude_end: float
    ) -> None:
        super().__init__(dt)
        self.duration = duration
        self.amplitude_start = amplitude_start
        self.amplitude_end = amplitude_end

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return np.linspace(self.amplitude_start, self.amplitude_end, len(self.time))


class Zap(Stimulus):
    def __init__(self, dt: float, duration: float, amplitude: float) -> None:
        super().__init__(dt)
        self.duration = duration
        self.amplitude = amplitude

    @property
    def time(self) -> np.ndarray:
        return np.arange(0.0, self.duration, self.dt)

    @property
    def current(self) -> np.ndarray:
        return self.amplitude * np.sin(
            2.0 * np.pi * (1.0 + (1.0 / (5.15 - (self.time - 0.1)))) * (self.time - 0.1)
        )


class OUProcess(Stimulus):
    """Generates an Ornstein-Uhlenbeck noise signal."""

    def __init__(self, dt: float, duration: float, tau: float, sigma: float, mean: float, seed: Optional[int] = None):
        super().__init__(dt)  # Ensure proper Stimulus initialization
        self.duration = duration
        self.tau = tau
        self.sigma = sigma
        self.mean = mean
        self.seed = seed

        # Generate OU noise upon initialization
        self._time, self._current = self._generate_ou_noise()

    @property
    def time(self) -> np.ndarray:
        """Returns the time array for the stimulus duration."""
        return self._time

    @property
    def current(self) -> np.ndarray:
        """Returns the Ornstein-Uhlenbeck noise signal."""
        return self._current

    def _generate_ou_noise(self):
        """Generates an Ornstein-Uhlenbeck noise signal."""
        from bluecellulab.cell.stimuli_generator import gen_ornstein_uhlenbeck
        from bluecellulab.rngsettings import RNGSettings

        rng_settings = RNGSettings.get_instance()
        rng = neuron.h.Random()

        if rng_settings.mode == "Random123":
            seed1, seed2, seed3 = 2997, 291204, self.seed if self.seed else 123
            rng.Random123(seed1, seed2, seed3)
        else:
            raise ValueError("Ornstein-Uhlenbeck stimulus requires Random123 RNG mode.")

        # Generate noise signal
        time, current = gen_ornstein_uhlenbeck(self.tau, self.sigma, self.mean, self.duration, self.dt, rng)
        return time, current


class ShotNoiseProcess(Stimulus):
    """Generates a shot noise signal, modeling discrete synaptic events
    occurring at random intervals."""

    def __init__(
        self, dt: float, duration: float, rate: float, mean: float, sigma: float,
        rise_time: float, decay_time: float, seed: Optional[int] = None
    ):
        super().__init__(dt)
        self.duration = duration
        self.rate = rate
        self.mean = mean
        self.sigma = sigma
        self.rise_time = rise_time
        self.decay_time = decay_time
        self.seed = seed

        # Generate shot noise signal
        self._time, self._current = self._generate_shot_noise()

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current

    def _generate_shot_noise(self):
        """Generates the shot noise time and current vectors."""
        from bluecellulab.cell.stimuli_generator import gen_shotnoise_signal
        from bluecellulab.rngsettings import RNGSettings

        rng_settings = RNGSettings.get_instance()
        rng = neuron.h.Random()

        if rng_settings.mode == "Random123":
            seed1, seed2, seed3 = 2997, 19216, self.seed if self.seed else 123
            rng.Random123(seed1, seed2, seed3)
        else:
            raise ValueError("Shot noise stimulus requires Random123 RNG mode.")

        variance = self.sigma ** 2
        tvec, svec = gen_shotnoise_signal(
            self.decay_time,
            self.rise_time,
            self.rate,
            self.mean,
            variance,
            self.duration,
            self.dt,
            rng=rng
        )

        return np.array(tvec.to_python()), np.array(svec.to_python())


class StepNoiseProcess(Stimulus):
    """Generates step noise: A step current with noise variations."""

    def __init__(
        self,
        dt: float,
        duration: float,
        step_interval: float,
        mean: float,
        sigma: float,
        seed: Optional[int] = None,
    ):
        super().__init__(dt)
        self.duration = duration
        self.step_interval = step_interval
        self.mean = mean
        self.sigma = sigma
        self.seed = seed

        # Generate step noise signal
        self._time, self._current = self._generate_step_noise()

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current

    def _generate_step_noise(self):
        """Generates the step noise time and current vectors using NEURON’s
        random generator."""
        from bluecellulab.rngsettings import RNGSettings

        # Get NEURON RNG settings
        rng_settings = RNGSettings.get_instance()
        rng = neuron.h.Random()

        if rng_settings.mode == "Random123":
            seed1, seed2, seed3 = 2997, 19216, self.seed if self.seed else 123
            rng.Random123(seed1, seed2, seed3)
        else:
            raise ValueError("StepNoise stimulus requires Random123 RNG mode.")

        num_steps = int(self.duration / self.step_interval)

        # Generate noise using NEURON's normal distribution function
        amplitudes = [self.mean + rng.normal(0, self.sigma) for _ in range(num_steps)]

        # Construct stimulus
        time_values = []
        current_values = []
        time = 0

        for amp in amplitudes:
            time_values.append(time)
            current_values.append(amp)
            time += self.step_interval

        return np.array(time_values), np.array(current_values)


class SinusoidalWave(Stimulus):
    """Generates a sinusoidal current wave."""

    def __init__(self, dt: float, duration: float, amplitude: float, frequency: float):
        super().__init__(dt)
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency

        self._time, self._current = self._generate_sinusoidal_signal()

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current

    def _generate_sinusoidal_signal(self):
        """Generate the sinusoidal waveform."""
        tvec = neuron.h.Vector()
        tvec.indgen(0.0, self.duration, self.dt)  # Time points using NEURON

        stim = neuron.h.Vector(len(tvec))
        stim.sin(self.frequency, 0.0, self.dt)  # Generate sinusoidal wave using NEURON
        stim.mul(self.amplitude)  # Scale by amplitude

        return np.array(tvec.to_python()), np.array(stim.to_python())


class PulseTrain(Stimulus):
    """Generates a pulse train signal."""

    def __init__(self, dt: float, duration: float, amplitude: float, frequency: float, width: float):
        super().__init__(dt)
        self.duration = duration
        self.amplitude = amplitude
        self.frequency = frequency
        self.width = width

        self._time, self._current = self._generate_pulse_train()

    @property
    def time(self) -> np.ndarray:
        return self._time

    @property
    def current(self) -> np.ndarray:
        return self._current

    def _generate_pulse_train(self):
        total_duration = self.duration
        time_steps = int(total_duration / self.dt)
        time = np.linspace(0, total_duration, time_steps)
        current = np.zeros_like(time)

        start_time = 0
        while start_time + self.width < self.duration:
            pulse_indices = (time >= start_time) & (time < start_time + self.width)
            current[pulse_indices] = self.amplitude
            start_time += 1000.0 / self.frequency

        return time, current


class Step(Stimulus):

    def __init__(self):
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use the class methods 'amplitude_based' "
            "or 'threshold_based' to create objects."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Step stimulus from given time events and amplitude.

        Args:
            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            amplitude: The amplitude of the step.
        """
        return (
            Empty(dt, duration=pre_delay)
            + Flat(dt, duration=duration, amplitude=amplitude)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Step stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )
        return res


class Ramp(Stimulus):

    def __init__(self):
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use the class methods 'amplitude_based' "
            "or 'threshold_based' to create objects."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a Ramp stimulus from given time events and amplitudes.

        Args:
            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the ramp.
            duration: The duration of the ramp.
            post_delay: The time to wait after the end of the ramp.
            amplitude: The final amplitude of the ramp.
        """
        return (
            Empty(dt, duration=pre_delay)
            + Slope(
                dt,
                duration=duration,
                amplitude_start=0.0,
                amplitude_end=amplitude,
            )
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a Ramp stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the ramp.
            duration: The duration of the ramp.
            post_delay: The time to wait after the end of the ramp.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )
        return res


class DelayedZap(Stimulus):

    def __init__(self):
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use the class methods 'amplitude_based' "
            "or 'threshold_based' to create objects."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
    ) -> CombinedStimulus:
        """Create a DelayedZap stimulus from given time events and amplitude.

        Args:
            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            amplitude: The amplitude of the step.
        """
        return (
            Empty(dt, duration=pre_delay)
            + Zap(dt, duration=duration, amplitude=amplitude)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        threshold_current: float,
        threshold_percentage: float,
    ) -> CombinedStimulus:
        """Creates a SineSpec stimulus with respect to the threshold current.

        Args:

            dt: The time step of the stimulus.
            pre_delay: The delay before the start of the step.
            duration: The duration of the step.
            post_delay: The time to wait after the end of the step.
            threshold_current: The threshold current of the Cell.
            threshold_percentage: Percentage of desired threshold_current amplification.
        """
        amplitude = threshold_current * threshold_percentage / 100
        res = cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
        )
        return res


class OrnsteinUhlenbeck(Stimulus):
    """Factory-compatible Ornstein-Uhlenbeck noise stimulus."""

    def __init__(self):
        """Prevents direct instantiation of the class."""
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use 'amplitude_based' or 'threshold_based' methods."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        tau: float,
        sigma: float,
        mean: float,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Create an Ornstein-Uhlenbeck stimulus from given time events and
        amplitude."""
        return (
            Empty(dt, duration=pre_delay)
            + OUProcess(dt, duration, tau, sigma, mean, seed)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        mean_percent: float,
        sigma_percent: float,
        threshold_current: float,
        tau: float,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Creates an Ornstein-Uhlenbeck stimulus with respect to the threshold
        current."""
        sigma = sigma_percent / 100 * threshold_current
        if sigma <= 0:
            raise BluecellulabError(f"Calculated standard deviation (sigma) must be positive, but got {sigma}. Ensure sigma_percent and threshold_current are both positive.")

        mean = mean_percent / 100 * threshold_current
        if mean < 0 and abs(mean) > 2 * sigma:
            logger.warning("Relative Ornstein-Uhlenbeck signal is mostly zero.")

        return cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            tau=tau,
            sigma=sigma,
            mean=mean,
            seed=seed,
        )


class ShotNoise(Stimulus):
    """Factory-compatible Shot Noise Stimulus."""

    def __init__(self):
        """Prevents direct instantiation."""
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use 'amplitude_based' or 'threshold_based' methods."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        rate: float,
        mean: float,
        sigma: float,
        rise_time: float,
        decay_time: float,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Creates a shot noise stimulus with a specified amplitude.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the noise starts.
            duration: Duration of the noise signal.
            post_delay: Delay after the noise ends.
            rate: Frequency of synaptic-like events.
            mean: Mean amplitude of the events.
            sigma: Standard deviation of event amplitudes.
            rise_time: Time constant for the event's rise phase.
            decay_time: Time constant for the event's decay phase.
            seed: Random seed for reproducibility.
        """
        return (
            Empty(dt, duration=pre_delay)
            + ShotNoiseProcess(dt, duration, rate, mean, sigma, rise_time, decay_time, seed)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        rise_time: float,
        decay_time: float,
        mean_percent: float,
        sigma_percent: float,
        threshold_current: float,
        relative_skew: float = 0.5,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Creates a shot noise stimulus based on a neuron's threshold current.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the noise starts.
            duration: Duration of the noise signal.
            post_delay: Delay after the noise ends.
            rise_time: Rise time constant of events.
            decay_time: Decay time constant of events.
            mean_percent: Mean value as a percentage of the threshold current.
            sigma_percent: Standard deviation as a percentage of the threshold current.
            threshold_current: Baseline threshold current.
            relative_skew: Skew factor affecting noise distribution.
            seed: Random seed for reproducibility.
        """
        _mean = mean_percent / 100 * threshold_current
        sd = sigma_percent / 100 * threshold_current

        rate, mean, sigma = get_relative_shotnoise_params(
            _mean, sd, decay_time, rise_time, relative_skew
        )

        return cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            rate=rate,
            mean=mean,
            sigma=sigma,
            rise_time=rise_time,
            decay_time=decay_time,
            seed=seed,
        )


class StepNoise(Stimulus):
    """Factory-compatible Step Noise Stimulus."""

    def __init__(self):
        """Prevents direct instantiation."""
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use 'amplitude_based' or 'threshold_based' methods."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        step_interval: float,
        mean: float,
        sigma: float,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Creates a step noise stimulus with a specified amplitude.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the step noise starts.
            duration: Duration of the noise signal.
            post_delay: Delay after the step noise ends.
            step_interval: Interval at which noise amplitude changes.
            mean: Mean amplitude of step noise.
            sigma: Standard deviation of step noise.
            seed: Random seed for reproducibility.
        """
        return (
            Empty(dt, duration=pre_delay)
            + StepNoiseProcess(dt, duration, step_interval, mean, sigma, seed)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        step_interval: float,
        mean_percent: float,
        sigma_percent: float,
        threshold_current: float,
        seed: Optional[int] = None,
    ) -> CombinedStimulus:
        """Creates a step noise stimulus relative to the threshold current.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the step noise starts.
            duration: Duration of the noise signal.
            post_delay: Delay after the step noise ends.
            step_interval: Interval at which noise amplitude changes.
            mean_percent: Mean current as a percentage of threshold current.
            sigma_percent: Standard deviation as a percentage of threshold current.
            threshold_current: Baseline threshold current.
            seed: Random seed for reproducibility.
        """
        mean = mean_percent / 100 * threshold_current
        sigma = sigma_percent / 100 * threshold_current

        return cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            step_interval=step_interval,
            mean=mean,
            sigma=sigma,
            seed=seed,
        )


class Sinusoidal(Stimulus):
    """Factory-compatible Sinusoidal Stimulus."""

    def __init__(self):
        """Prevents direct instantiation."""
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use 'amplitude_based' or 'threshold_based' methods."
        )

    @classmethod
    def amplitude_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        amplitude: float,
        frequency: float,
    ) -> CombinedStimulus:
        """Creates a sinusoidal stimulus with a specified amplitude.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the sinusoidal wave starts.
            duration: Duration of the sinusoidal signal.
            post_delay: Delay after the wave ends.
            amplitude: Amplitude of the sinusoidal wave.
            frequency: Frequency of the wave in Hz.
        """
        return (
            Empty(dt, duration=pre_delay)
            + SinusoidalWave(dt, duration, amplitude, frequency)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls,
        dt: float,
        pre_delay: float,
        duration: float,
        post_delay: float,
        frequency: float,
        threshold_current: float,
        amplitude_percent: float,
    ) -> CombinedStimulus:
        """Creates a sinusoidal stimulus relative to the threshold current.

        Args:
            dt: Time step of the stimulus.
            pre_delay: Delay before the sinusoidal wave starts.
            duration: Duration of the sinusoidal signal.
            post_delay: Delay after the wave ends.
            frequency: Frequency of the wave in Hz.
            threshold_current: Baseline threshold current.
            amplitude_percent: Amplitude as a percentage of the threshold current.
        """
        amplitude = (amplitude_percent / 100) * threshold_current

        return cls.amplitude_based(
            dt,
            pre_delay=pre_delay,
            duration=duration,
            post_delay=post_delay,
            amplitude=amplitude,
            frequency=frequency,
        )


class Pulse(Stimulus):
    """Factory-compatible Pulse Stimulus."""

    def __init__(self):
        """Prevents direct instantiation of the class."""
        raise NotImplementedError(
            "This class cannot be instantiated directly. "
            "Please use 'amplitude_based' or 'threshold_based' methods."
        )

    @classmethod
    def amplitude_based(
        cls, dt: float, pre_delay: float, duration: float, post_delay: float, amplitude: float, frequency: float, width: float
    ) -> CombinedStimulus:
        """Creates a Pulse stimulus from given time events and amplitude."""
        return (
            Empty(dt, duration=pre_delay)
            + PulseTrain(dt, duration, amplitude, frequency, width)
            + Empty(dt, duration=post_delay)
        )

    @classmethod
    def threshold_based(
        cls, dt: float, pre_delay: float, duration: float, post_delay: float, threshold_current: float, threshold_percentage: float,
        frequency: float, width: float
    ) -> CombinedStimulus:
        """Creates a Pulse stimulus with respect to the threshold current."""
        amplitude = threshold_current * (threshold_percentage / 100)
        return cls.amplitude_based(dt, pre_delay, duration, post_delay, amplitude, frequency, width)
