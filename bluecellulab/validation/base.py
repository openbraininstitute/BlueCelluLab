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

"""Base abstractions for the validation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TestResult:
    """Standard result of a validation test.

    Contains everything needed for OBI-One to register a ValidationResult entity.

    Attributes:
        name: Stable identifier/name for this validation (used for dedup and display).
        passed: Whether the validation criteria were met.
        details: Human-readable explanation of the result.
        figures: Paths to generated figure files (plots, traces, etc.).
    """

    name: str
    passed: bool
    details: str
    figures: list[Path] = field(default_factory=list)


class ValidationTest(ABC):
    """Abstract base class for all validation tests.

    A validation test encapsulates a complete check: stimulate a cell,
    measure a property, and determine pass/fail against a criterion.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for this validation test."""

    @abstractmethod
    def run(self, template_params, rheobase: float, out_dir: Path) -> TestResult:
        """Execute the validation test.

        Args:
            template_params: BlueCelluLab TemplateParams for creating the cell.
            rheobase: The rheobase (threshold) current in nA.
            out_dir: Directory to write output figures.

        Returns:
            A TestResult with the test result.
        """
