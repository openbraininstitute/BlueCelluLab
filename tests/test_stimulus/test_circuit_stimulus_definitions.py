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
from bluecellulab.stimulus.circuit_stimulus_definitions import Pattern


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
        "seclamp": Pattern.SECLAMP,
    }

    for sonata_pattern, expected_enum in valid_patterns.items():
        assert Pattern.from_sonata(sonata_pattern) == expected_enum


def test_pattern_from_sonata_invalid():
    """Test that an invalid pattern raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown pattern unknown_pattern"):
        Pattern.from_sonata("unknown_pattern")
