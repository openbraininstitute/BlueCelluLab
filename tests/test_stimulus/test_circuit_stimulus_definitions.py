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

import pytest
from bluecellulab.stimulus.circuit_stimulus_definitions import Noise, Pattern, Stimulus


def test_pattern_from_sonata_valid():
    """Test valid mappings from SONATA strings to Pattern enum values."""
    valid_patterns = {
        "noise": Pattern.NOISE,
        "hyperpolarizing": Pattern.HYPERPOLARIZING,
        "pulse": Pattern.PULSE,
        "linear": Pattern.LINEAR,
        "relative_linear": Pattern.RELATIVE_LINEAR,
        "synapse_replay": Pattern.SYNAPSE_REPLAY,
        "shot_noise": Pattern.SHOT_NOISE,
        "relative_shot_noise": Pattern.RELATIVE_SHOT_NOISE,
        "ornstein_uhlenbeck": Pattern.ORNSTEIN_UHLENBECK,
        "relative_ornstein_uhlenbeck": Pattern.RELATIVE_ORNSTEIN_UHLENBECK,
    }

    for sonata_pattern, expected_enum in valid_patterns.items():
        assert Pattern.from_sonata(sonata_pattern) == expected_enum


def test_pattern_from_sonata_invalid():
    """Test that an invalid pattern raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown pattern unknown_pattern"):
        Pattern.from_sonata("unknown_pattern")


def test_noise_requires_exactly_one_mean_field():
    """Noise dataclass should validate exactly one of mean/mean_percent."""
    with pytest.raises(ValueError, match="Noise stimulus must define exactly one of 'mean' or 'mean_percent'."):
        Noise(target="T", delay=0.0, duration=1.0, variance=0.1)

    with pytest.raises(ValueError, match="Noise stimulus must define exactly one of 'mean' or 'mean_percent'."):
        Noise(target="T", delay=0.0, duration=1.0, variance=0.1, mean=0.01, mean_percent=5.0)


def test_noise_negative_variance_raises():
    """Noise variance must be non-negative."""
    with pytest.raises(ValueError, match="'variance' must be >= 0."):
        Noise(target="T", delay=0.0, duration=1.0, variance=-0.1, mean=0.01)


def test_from_sonata_noise_requires_one_mean_field():
    """Parsing SONATA noise stimulus enforces exactly one mean field."""
    base = {
        "module": "noise",
        "delay": 0.0,
        "duration": 1.0,
        "variance": 0.1,
        "node_set": "T",
    }

    with pytest.raises(ValueError, match="Noise input must contain exactly one of 'mean' or 'mean_percent'."):
        Stimulus.from_sonata(dict(base))

    with pytest.raises(ValueError, match="Noise input must contain exactly one of 'mean' or 'mean_percent'."):
        Stimulus.from_sonata({**base, "mean": 0.01, "mean_percent": 5.0})
