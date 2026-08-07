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

"""Criteria for determining pass/fail of a validation measurement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Criterion(ABC):
    """Abstract base for pass/fail criteria applied to a measurement value."""

    @abstractmethod
    def evaluate(self, value) -> bool:
        """Evaluate whether the measured value meets this criterion."""

    @abstractmethod
    def describe(self, value) -> str:
        """Human-readable description of the evaluation result."""


@dataclass
class GreaterThan(Criterion):
    """Passes if the measured value is strictly greater than a threshold.

    For array-valued measurements (e.g. AP_amplitude per spike), all values
    must exceed the threshold.

    Attributes:
        threshold: The value that must be exceeded.
    """

    threshold: float

    def evaluate(self, value) -> bool:
        result = np.all(np.asarray(value) > self.threshold)
        return bool(result)

    def describe(self, value) -> str:
        passed = self.evaluate(value)
        # Summarize array values
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.size == 1:
            display = f"{float(arr):.4g}"
        else:
            display = f"min={float(arr.min()):.4g}, mean={float(arr.mean()):.4g}, max={float(arr.max()):.4g}"

        if passed:
            return f"Value ({display}) is greater than {self.threshold}."
        return f"Value ({display}) is not greater than {self.threshold}."
