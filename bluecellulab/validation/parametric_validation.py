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
"""Parametric validation test: composes protocol + measurement + criterion."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bluecellulab.validation.base import TestResult, ValidationTest
from bluecellulab.validation.criterion import Criterion
from bluecellulab.validation.measurement import Measurement
from bluecellulab.validation.plotting import plot_trace
from bluecellulab.validation.protocol import Protocol

logger = logging.getLogger(__name__)


@dataclass
class ParametricValidation(ValidationTest):
    """A composable validation test built from protocol + measurement +
    criterion.

    This class allows users to define custom validations by combining:
    - A Protocol (how to stimulate the cell)
    - A Measurement (what to extract from the recording)
    - A Criterion (how to judge pass/fail)

    Attributes:
        validation_name: Stable identifier for this validation.
        protocol: The stimulation protocol to execute.
        measurement: The measurement to extract from the recording.
        criterion: The pass/fail criterion to apply to the measurement.
        figure_filename: Optional filename for the output trace plot. If omitted,
            a filename is derived from validation_name.
    """

    validation_name: str
    protocol: Protocol
    measurement: Measurement
    criterion: Criterion
    figure_filename: str | None = None

    @property
    def name(self) -> str:
        return self.validation_name

    def _resolved_figure_filename(self) -> str:
        """Return an explicit filename or a safe derived default."""
        if self.figure_filename is not None:
            return self.figure_filename

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.name).strip("._-")
        return f"{safe_name or 'validation'}.pdf"

    def run(self, template_params, rheobase: float, out_dir: Path) -> TestResult:
        """Execute protocol, extract measurement, evaluate criterion.

        Args:
            template_params: BlueCelluLab TemplateParams for creating the cell.
            rheobase: The rheobase (threshold) current in nA.
            out_dir: Directory to write output figures.

        Returns:
            A TestResult with the test result.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Execute protocol. A cell that cannot be stimulated is a failed
        # validation rather than a crash, so a batch run still reports on it.
        try:
            recording = self.protocol.execute(template_params, rheobase)
        except Exception as error:  # noqa: BLE001 - a failed protocol is a failed validation
            logger.warning(
                "Protocol %s failed for validation %r: %s",
                type(self.protocol).__name__,
                self.name,
                error,
            )
            return TestResult(
                name=self.name,
                passed=False,
                details=(
                    "Validation failed: the protocol could not be executed "
                    f"({type(error).__name__}: {error})."
                ),
                figures=[],
            )

        # 2. Extract measurement
        extraction_error = None
        try:
            value = self.measurement.extract(
                recording,
                stim_start=self.protocol.stim_start,
                stim_end=self.protocol.stim_end,
            )
        except Exception as error:  # noqa: BLE001 - a failed measurement is a failed validation
            value = None
            extraction_error = error

        # 3. Evaluate criterion
        if value is None:
            if extraction_error is None:
                details = (
                    "Validation failed: measurement could not be extracted "
                    "(feature returned None)."
                )
            else:
                details = (
                    "Validation failed: measurement could not be extracted "
                    f"({type(extraction_error).__name__}: {extraction_error})."
                )
            passed = False
        else:
            passed = self.criterion.evaluate(value)
            details = self.criterion.describe(value)

        # 4. Generate figure. Plotting is reporting, not evaluation, so a
        # failure here is logged and the verdict is still returned.
        figures = []
        try:
            figures.append(
                plot_trace(
                    recording,
                    out_dir,
                    fname=self._resolved_figure_filename(),
                    title=f"{self.name} - {self.protocol.__class__.__name__}",
                )
            )
        except Exception as error:  # noqa: BLE001 - a figure is not part of the verdict
            logger.warning(
                "Could not plot the trace for validation %r: %s", self.name, error
            )
            details += " (Figure could not be generated.)"

        return TestResult(
            name=self.name,
            passed=passed,
            details=details,
            figures=figures,
        )
