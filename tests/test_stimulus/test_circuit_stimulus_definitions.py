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
from bluecellulab.stimulus.circuit_stimulus_definitions import (
    Noise, 
    Pattern, 
    Stimulus,
    SpatiallyUniformEField,
)


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


def test_pattern_spatially_uniform_e_field():
    """Test that spatially_uniform_e_field maps correctly."""
    assert Pattern.from_sonata("spatially_uniform_e_field") == Pattern.SPATIALLY_UNIFORM_E_FIELD


def test_spatially_uniform_e_field_valid():
    """Test valid SpatiallyUniformEField construction."""
    stim = SpatiallyUniformEField(
        target="T",
        delay=0.0,
        duration=10.0,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 75}],
        ramp_up_time=2.0,
        ramp_down_time=3.0,
        node_set="T",
        compartment_set=None,
    )
    assert stim.fields == [{"Ex": 100, "Ey": -50, "Ez": 75}]
    assert stim.ramp_up_time == 2.0
    assert stim.ramp_down_time == 3.0


def test_spatially_uniform_e_field_empty_fields():
    """Test that empty fields list raises validation error."""
    with pytest.raises(ValueError, match="fields list cannot be empty"):
        SpatiallyUniformEField(
            target="T",
            delay=0.0,
            duration=10.0,
            fields=[],
            node_set="T",
            compartment_set=None,
        )


def test_spatially_uniform_e_field_missing_components():
    """Test that missing Ex/Ey/Ez raises validation error."""
    with pytest.raises(ValueError, match="Field 0 must contain Ex, Ey, and Ez components"):
        SpatiallyUniformEField(
            target="T",
            delay=0.0,
            duration=10.0,
            fields=[{"Ex": 100, "Ey": -50}],
            node_set="T",
            compartment_set=None,
        )


def test_spatially_uniform_e_field_negative_frequency():
    """Test that negative frequency raises validation error."""
    with pytest.raises(ValueError, match="Field 0 frequency must be non-negative"):
        SpatiallyUniformEField(
            target="T",
            delay=0.0,
            duration=10.0,
            fields=[{"Ex": 100, "Ey": -50, "Ez": 75, "frequency": -10}],
            node_set="T",
            compartment_set=None,
        )


def test_from_sonata_spatially_uniform_e_field():
    """Test parsing SONATA spatially_uniform_e_field stimulus."""
    entry = {
        "module": "spatially_uniform_e_field",
        "delay": 0.0,
        "duration": 10.0,
        "node_set": "TestTarget",
        "fields": [
            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
        ],
        "ramp_up_time": 2.0,
        "ramp_down_time": 3.0,
    }
    
    stim = Stimulus.from_sonata(entry)
    assert isinstance(stim, SpatiallyUniformEField)
    assert stim.delay == 0.0
    assert stim.duration == 10.0
    assert stim.node_set == "TestTarget"
    assert stim.compartment_set is None
    assert len(stim.fields) == 2
    assert stim.ramp_up_time == 2.0
    assert stim.ramp_down_time == 3.0


def test_from_sonata_spatially_uniform_e_field_defaults():
    """Test parsing with default ramp times."""
    entry = {
        "module": "spatially_uniform_e_field",
        "delay": 0.0,
        "duration": 10.0,
        "node_set": "TestTarget",
        "fields": [{"Ex": 50, "Ey": -25, "Ez": 75}],
    }
    
    stim = Stimulus.from_sonata(entry)
    assert isinstance(stim, SpatiallyUniformEField)
    assert stim.ramp_up_time == 0.0
    assert stim.ramp_down_time == 0.0
