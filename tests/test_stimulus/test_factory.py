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
import pytest
import numpy as np

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
        s = self.factory.step_noise(duration=1000, step_duration=50, mean=1.0, variance=0.3, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))
        assert len(s.time) == len(s.current)

        s = self.factory.step_noise(duration=1000, step_duration=50, mean_percent=60.0, sd_percent=20.0, threshold_current=0.8, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either threshold_current or mean."):
            self.factory.step_noise(duration=1000, step_duration=50)

    def test_create_shot_noise(self):
        s = self.factory.shot_noise(duration=1000, rate=10, amp_mean=0.5, amp_var=0.1, rise_time=1, decay_time=5, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))

        s = self.factory.shot_noise(duration=1000, rate=10, mean_percent=60.0, sd_percent=20.0, threshold_current=0.8, rise_time=1, decay_time=5, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.shot_noise(duration=1000, rate=10, rise_time=1, decay_time=5)

    def test_create_ornstein_uhlenbeck(self):
        s = self.factory.ornstein_uhlenbeck(duration=1000, tau=20, sigma=0.5, mean=1.0, seed=42)
        assert isinstance(s, CombinedStimulus)
        assert np.all(np.isfinite(s.current))

        s = self.factory.ornstein_uhlenbeck(duration=1000, tau=20, mean_percent=60.0, sd_percent=20.0, threshold_current=0.8, seed=42)
        assert isinstance(s, CombinedStimulus)

        with pytest.raises(TypeError, match="You have to give either threshold_current or amplitude"):
            self.factory.ornstein_uhlenbeck(duration=1000, tau=20)
