# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import pytest
import numpy as np

from bluecellulab.exceptions import BluecellulabError
from bluecellulab.stimulus.stimulus import (
    CombinedStimulus,
    Zap,
)
from bluecellulab.stimulus.factory import StimulusFactory


class TestStimulusFactory:

    def setup_method(self):
        self.dt = 0.1
        self.factory = StimulusFactory(dt=self.dt)

    def test_create_step(self):
        stim = self.factory.step(0, 1, 0, 0.55)
        assert isinstance(stim, CombinedStimulus)
        assert np.all(stim.time == np.arange(0, 1, self.dt))
        assert np.all(stim.current == np.full(10, 0.55))

    def test_create_ramp(self):
        pre_delay, duration, post_delay = 1, 2, 0
        total_time = sum([pre_delay, duration, post_delay])
        stim = self.factory.ramp(pre_delay, duration, post_delay, amplitude=3)
        assert isinstance(stim, CombinedStimulus)
        np.testing.assert_almost_equal(stim.time, np.arange(0, total_time, self.dt), decimal=9)
        assert stim.current[0] == 0.0
        assert stim.current[-1] == 3.0

        s = self.factory.ramp(pre_delay, duration, post_delay, threshold_current=1)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.ramp(pre_delay, duration, post_delay)

    def test_create_ap_waveform(self):
        s = self.factory.ap_waveform(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        # below write np almost equal
        np.testing.assert_allclose(s.time, np.arange(0, 550, self.dt))
        assert s.current[0] == 0.0
        assert s.current[2500] == 2.2
        assert s.current[-1] == 0.0

        s = self.factory.ap_waveform(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.ap_waveform(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.ap_waveform()

    def test_create_idrest(self):
        s = self.factory.idrest(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 1850

        s = self.factory.idrest(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 1850

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.idrest(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.idrest()

    def test_create_iv(self):
        s = self.factory.iv(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 3500
        # assert there are negative values
        assert np.any(s.current < 0)
        # assert no positive values
        assert not np.any(s.current > 0)

        s = self.factory.iv(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 3500

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.iv(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.iv()

    def test_create_fire_pattern(self):
        s = self.factory.fire_pattern(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 4100

        s = self.factory.fire_pattern(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 4100

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.fire_pattern(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.fire_pattern()

    def test_create_pos_cheops(self):
        s = self.factory.pos_cheops(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 19166.0

        s = self.factory.pos_cheops(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 19166.0

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.pos_cheops(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.pos_cheops()

    def test_create_neg_cheops(self):
        s = self.factory.neg_cheops(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 18220.0

        s = self.factory.neg_cheops(amplitude=0.1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 18220.0

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.neg_cheops(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.neg_cheops()

    def test_create_sinespec(self):
        s = self.factory.sinespec(threshold_current=1)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 5000.0

        test_amp = 0.1
        s = self.factory.sinespec(amplitude=test_amp)
        assert isinstance(s, CombinedStimulus)
        assert s.stimulus_time == 5000.0
        assert np.max(np.abs(s.current)) <= test_amp
        assert (s.current == Zap(dt=self.dt, duration=5000.0, amplitude=test_amp).current).all()

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.sinespec(threshold_current=0.0)
        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.sinespec()

    def test_create_step_noise(self):
        s = self.factory.step_noise(pre_delay=100, post_delay=100, duration=1000, step_interval=50, mean=1.0, sigma=0.3, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))
        assert len(s.time) == len(s.current)

        s = self.factory.step_noise(pre_delay=100, post_delay=100, duration=1000, step_interval=50, mean_percent=60.0, sigma_percent=20.0, threshold_current=0.8, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You must provide either `mean` and `sigma`, or `threshold_current` and `mean_percent` and `sigma_percent`  with percentage values."):
            self.factory.step_noise(pre_delay=100, post_delay=100, duration=1000, step_interval=50)

    def test_create_shot_noise(self):
        s = self.factory.shot_noise(pre_delay=100, post_delay=100, duration=1000, rate=10, mean=0.5, sigma=0.1, rise_time=1, decay_time=5, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))

        s = self.factory.shot_noise(pre_delay=100, post_delay=100, duration=1000, rate=10, mean_percent=60.0, sigma_percent=20.0, threshold_current=0.8, rise_time=1, decay_time=5, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You must provide either `mean` and `sigma`, or `threshold_current` and `mean_percent` and `sigma_percent` with percentage values."):
            self.factory.shot_noise(pre_delay=100, post_delay=100, duration=1000, rate=10, rise_time=1, decay_time=5)

    def test_create_ornstein_uhlenbeck(self, caplog):
        s = self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20, sigma=0.5, mean=1.0, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))

        s = self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20, mean_percent=60.0, sigma_percent=20.0, threshold_current=0.8, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either `mean` and `sigma` or `threshold_current` and `mean_percent` and `sigma_percent`."):
            self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20)

        with pytest.raises(TypeError, match="You have to give either `mean` and `sigma` or `threshold_current` and `mean_percent` and `sigma_percent`."):
            self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20, sigma=0)

        sigma_percent = 0.0
        threshold_current = 0.8
        sigma = sigma_percent / 100 * threshold_current
        with pytest.raises(BluecellulabError, match=f"Calculated standard deviation \\(sigma\\) must be positive, but got {sigma}\\. Ensure sigma_percent and threshold_current are both positive\\."):
            self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20, mean_percent=60.0, sigma_percent=0.0, threshold_current=0.8, seed=42)

        with caplog.at_level(logging.WARNING):
            self.factory.ornstein_uhlenbeck(pre_delay=100, post_delay=100, duration=1000, tau=20, mean_percent=-80.0, sigma_percent=20.0, threshold_current=0.8, seed=42)

        assert any("Relative Ornstein-Uhlenbeck signal is mostly zero." in record.message for record in caplog.records)

    def test_create_pulse(self):
        s = self.factory.pulse(pre_delay=50.0, duration=250.0, post_delay=50.0, width=10.0, frequency=10.0, amplitude=0.5)
        assert isinstance(s, CombinedStimulus)
        assert np.isclose(s.stimulus_time, 50 + 250 + 50, atol=self.dt), f"Expected {50 + 250 + 50} ms, but got {s.stimulus_time} ms"
        assert np.max(s.current) == 0.5, "Pulse amplitude is incorrect"

        pulse_intervals = 1000.0 / 10.0  # 100 ms intervals
        expected_pulse_times = np.arange(50, 50 + 250, pulse_intervals)
        actual_pulse_times = np.where(np.diff(s.current) > 0)[0] * self.dt

        assert len(actual_pulse_times) == len(expected_pulse_times), "Number of pulses does not match expectation"
        np.testing.assert_allclose(actual_pulse_times, expected_pulse_times, atol=self.dt)

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.pulse(pre_delay=50, duration=250, post_delay=50, width=10, frequency=10)


class TestSinusoidalStimulus:

    def setup_method(self):
        self.dt = 0.025  # Default dt for sinusoidal stimulus
        self.factory = StimulusFactory(dt=self.dt)

    def test_create_sinusoidal_amplitude_based(self):
        """Test sinusoidal stimulus with absolute amplitude."""
        duration = 500.0  # ms
        frequency = 5.0  # Hz
        amplitude = 0.2  # nA

        s = self.factory.sinusoidal(
            pre_delay=100.0,
            post_delay=100.0,
            duration=duration,
            frequency=frequency,
            amplitude=amplitude,
        )

        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))  # Ensure no NaN values
        assert np.isclose(
            s.stimulus_time,
            100 + duration + 100,
            atol=0.025  # Allow a tolerance equal to dt
        ), f"Expected {100 + duration + 100} ms, but got {s.stimulus_time} ms"
        assert np.max(np.abs(s.current)) <= amplitude  # Ensure max amplitude is correct

    def test_create_sinusoidal_threshold_based(self):
        """Test sinusoidal stimulus with threshold-based amplitude."""
        duration = 500.0  # ms
        frequency = 5.0  # Hz
        threshold_current = 0.8  # Reference threshold
        amplitude_percent = 50.0  # 50% of threshold current

        s = self.factory.sinusoidal(
            pre_delay=100.0,
            post_delay=100.0,
            duration=duration,
            frequency=frequency,
            amplitude_percent=amplitude_percent,
            threshold_current=threshold_current,
        )

        expected_amplitude = (amplitude_percent / 100) * threshold_current
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))  # Ensure no NaN values
        assert np.isclose(
            s.stimulus_time,
            100 + duration + 100,
            atol=0.025  # Allow a tolerance equal to dt
        ), f"Expected {100 + duration + 100} ms, but got {s.stimulus_time} ms"
        assert np.max(np.abs(s.current)) <= expected_amplitude  # Ensure amplitude scaling is correct

    def test_create_sinusoidal_invalid_parameters(self):
        """Test invalid parameter combinations for sinusoidal stimulus."""
        with pytest.raises(TypeError, match="You have to provide either `amplitude` or `threshold_current` with `amplitude_percent`."):
            self.factory.sinusoidal(pre_delay=100, post_delay=100, duration=500, frequency=5)

        with pytest.raises(TypeError, match="You have to provide either `amplitude` or `threshold_current` with `amplitude_percent`."):
            self.factory.sinusoidal(pre_delay=100, post_delay=100, duration=500, frequency=5, amplitude_percent=50.0)

        with pytest.raises(TypeError, match="You have to provide either `amplitude` or `threshold_current` with `amplitude_percent`."):
            self.factory.sinusoidal(pre_delay=100, post_delay=100, duration=500, frequency=5, threshold_current=0.8)

    def test_sinusoidal_zero_amplitude(self):
        """Ensure a zero-amplitude sinusoidal stimulus results in a flat signal."""
        s = self.factory.sinusoidal(
            pre_delay=100.0,
            post_delay=100.0,
            duration=500.0,
            frequency=5.0,
            amplitude=0.0,
        )

        assert isinstance(s, CombinedStimulus)
        assert np.all(s.current == 0.0)  # Ensure it remains zero throughout

    def test_sinusoidal_waveform_shape(self):
        """Validate the sinusoidal waveform shape."""
        s = self.factory.sinusoidal(
            pre_delay=0.0,
            post_delay=0.0,
            duration=1000.0,  # 1 second
            frequency=10.0,  # 10 Hz
            amplitude=0.5,  # nA
        )

        time_values = s.time
        current_values = s.current

        # Check zero-crossings occur at expected frequency
        zero_crossings = np.where(np.diff(np.sign(current_values)))[0]
        assert len(zero_crossings) > 1, "Not enough zero crossings detected."

        # Compute estimated period using zero crossings
        estimated_period = np.mean(np.diff(time_values[zero_crossings])) * 2  # Full period
        expected_period = (1 / 10.0) * 1000  # Convert Hz to ms

        assert np.isclose(estimated_period, expected_period, rtol=0.3), (
            f"Estimated period {estimated_period} ms does not match expected {expected_period} ms"
        )
