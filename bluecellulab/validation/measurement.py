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

"""Measurements extract quantitative values from simulation recordings."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import efel

from bluecellulab.analysis.inject_sequence import Recording


class Measurement(ABC):
    """Abstract base for extracting a value from a simulation recording."""

    @abstractmethod
    def extract(self, recording: Recording, stim_start: float, stim_end: float):
        """Extract a measurement value from a recording.

        Args:
            recording: The simulation recording containing time, voltage, current.
            stim_start: Stimulus onset time in ms.
            stim_end: Stimulus offset time in ms.

        Returns:
            The measured value (type depends on the specific measurement).
        """


@dataclass
class EfelMeasurement(Measurement):
    """Extract an eFEL feature from a recording.

    Attributes:
        feature_name: The eFEL feature to extract (e.g. "Spikecount", "AP_amplitude").
        efel_settings: Optional dict of eFEL settings to apply before extraction
            (e.g. {"depol_block_min_duration": 150}).
    """

    feature_name: str
    efel_settings: dict = field(default_factory=dict)

    def extract(self, recording: Recording, stim_start: float, stim_end: float):
        """Extract the eFEL feature value from the recording.

        Returns:
            The scalar feature value, or None if extraction fails.
        """
        for key, value in self.efel_settings.items():
            efel.set_setting(key, value)

        trace = {
            "T": recording.time,
            "V": recording.voltage,
            "stim_start": [stim_start],
            "stim_end": [stim_end],
        }
        features_results = efel.get_feature_values([trace], [self.feature_name])
        result = features_results[0][self.feature_name]
        if result is None or len(result) == 0:
            return None
        # For scalar features (e.g. Spikecount), return the single value.
        if len(result) == 1:
            return result[0]
        return result
