"""Unit tests for the parametric validation framework."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from bluecellulab.validation.base import TestResult
from bluecellulab.validation.criterion import GreaterThan
from bluecellulab.validation.measurement import EfelMeasurement
from bluecellulab.validation.parametric_test import ParametricValidation
from bluecellulab.validation.plotting import plot_trace
from bluecellulab.validation.protocol import StepProtocol


class TestTestResult:
    def test_creation_minimal(self):
        outcome = TestResult(name="test", passed=True, details="ok")
        assert outcome.name == "test"
        assert outcome.passed is True
        assert outcome.details == "ok"
        assert outcome.figures == []

    def test_creation_with_figures(self, tmp_path):
        figs = [tmp_path / "fig1.pdf", tmp_path / "fig2.png"]
        outcome = TestResult(name="test", passed=False, details="failed", figures=figs)
        assert outcome.figures == figs
        assert outcome.passed is False

    def test_figures_default_is_not_shared(self):
        """Ensure each instance gets its own figures list."""
        o1 = TestResult(name="a", passed=True, details="")
        o2 = TestResult(name="b", passed=True, details="")
        o1.figures.append(Path("x"))
        assert o2.figures == []


class TestGreaterThan:
    def test_passes_when_above(self):
        c = GreaterThan(threshold=5.0)
        assert c.evaluate(6) is True

    def test_fails_when_equal(self):
        c = GreaterThan(threshold=5.0)
        assert c.evaluate(5) is False

    def test_fails_when_below(self):
        c = GreaterThan(threshold=5.0)
        assert c.evaluate(3) is False

    def test_describe_pass(self):
        c = GreaterThan(threshold=0)
        desc = c.describe(5)
        assert "5" in desc
        assert "greater than" in desc

    def test_describe_fail(self):
        c = GreaterThan(threshold=10)
        desc = c.describe(3)
        assert "not greater than" in desc


class TestEfelMeasurement:
    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_scalar_feature(self, mock_efel):
        mock_efel.return_value = [{"Spikecount": [5]}]
        m = EfelMeasurement(feature_name="Spikecount")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)
        value = m.extract(recording, stim_start=10.0, stim_end=90.0)
        assert value == 5
        mock_efel.assert_called_once()

    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_returns_none_when_feature_is_none(self, mock_efel):
        mock_efel.return_value = [{"Spikecount": None}]
        m = EfelMeasurement(feature_name="Spikecount")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)
        value = m.extract(recording, stim_start=10.0, stim_end=90.0)
        assert value is None

    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_returns_none_when_feature_is_empty(self, mock_efel):
        mock_efel.return_value = [{"Spikecount": []}]
        m = EfelMeasurement(feature_name="Spikecount")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)
        value = m.extract(recording, stim_start=10.0, stim_end=90.0)
        assert value is None

    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_multi_value_feature(self, mock_efel):
        mock_efel.return_value = [{"peak_time": [10.5, 20.3, 30.1]}]
        m = EfelMeasurement(feature_name="peak_time")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)
        value = m.extract(recording, stim_start=10.0, stim_end=90.0)
        assert value == [10.5, 20.3, 30.1]


class TestStepProtocol:
    def test_default_values(self):
        p = StepProtocol()
        assert p.threshold_percentage == 130.0
        assert p.dt == 1.0
        assert p.section == "soma[0]"
        assert p.segment == 0.5
        assert p.add_hypamp is True

    def test_stim_timing(self):
        p = StepProtocol()
        assert p.stim_start == 250.0
        assert p.stim_end == 1600.0

    def test_custom_parameters(self):
        p = StepProtocol(
            threshold_percentage=200.0,
            dt=0.5,
            section="axon[0]",
            segment=0.1,
            add_hypamp=False,
        )
        assert p.threshold_percentage == 200.0
        assert p.dt == 0.5
        assert p.section == "axon[0]"
        assert p.segment == 0.1
        assert p.add_hypamp is False

    @patch("bluecellulab.validation.protocol.run_stimulus")
    @patch("bluecellulab.validation.protocol.StimulusFactory")
    def test_execute_calls_stimulus_factory(self, mock_factory_cls, mock_run):
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        mock_stimulus = MagicMock()
        mock_factory.idrest.return_value = mock_stimulus
        mock_recording = MagicMock()
        mock_run.return_value = mock_recording

        p = StepProtocol(threshold_percentage=150.0)
        result = p.execute("template_params", 0.5)

        mock_factory_cls.assert_called_once_with(dt=1.0)
        mock_factory.idrest.assert_called_once_with(
            threshold_current=0.5, threshold_percentage=150.0
        )
        mock_run.assert_called_once_with(
            "template_params", mock_stimulus, "soma[0]", 0.5, add_hypamp=True
        )
        assert result == mock_recording


class TestParametricValidation:
    @patch("bluecellulab.validation.parametric_test.plot_trace")
    def test_run_passes(self, mock_plot, tmp_path):
        mock_plot.return_value = tmp_path / "fig.pdf"

        protocol = MagicMock()
        protocol.execute.return_value = MagicMock()
        protocol.stim_start = 250.0
        protocol.stim_end = 1600.0
        protocol.__class__.__name__ = "StepProtocol"

        measurement = MagicMock()
        measurement.extract.return_value = 5

        criterion = MagicMock()
        criterion.evaluate.return_value = True
        criterion.describe.return_value = "Value 5 is greater than 0."

        pv = ParametricValidation(
            validation_name="Test Validation",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
            figure_filename="test.pdf",
        )

        outcome = pv.run("tparams", 1.0, tmp_path)

        assert outcome.name == "Test Validation"
        assert outcome.passed is True
        assert outcome.details == "Value 5 is greater than 0."
        assert len(outcome.figures) == 1

        protocol.execute.assert_called_once_with("tparams", 1.0)
        measurement.extract.assert_called_once()
        criterion.evaluate.assert_called_once_with(5)

    @patch("bluecellulab.validation.parametric_test.plot_trace")
    def test_run_fails(self, mock_plot, tmp_path):
        mock_plot.return_value = tmp_path / "fig.pdf"

        protocol = MagicMock()
        protocol.execute.return_value = MagicMock()
        protocol.stim_start = 250.0
        protocol.stim_end = 1600.0
        protocol.__class__.__name__ = "StepProtocol"

        measurement = MagicMock()
        measurement.extract.return_value = 0

        criterion = MagicMock()
        criterion.evaluate.return_value = False
        criterion.describe.return_value = "Value 0 is not greater than 0."

        pv = ParametricValidation(
            validation_name="Fail Test",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
        )

        outcome = pv.run("tparams", 1.0, tmp_path)
        assert outcome.passed is False
        assert "not greater than" in outcome.details

    @patch("bluecellulab.validation.parametric_test.plot_trace")
    def test_run_with_none_measurement(self, mock_plot, tmp_path):
        mock_plot.return_value = tmp_path / "fig.pdf"

        protocol = MagicMock()
        protocol.execute.return_value = MagicMock()
        protocol.stim_start = 250.0
        protocol.stim_end = 1600.0
        protocol.__class__.__name__ = "StepProtocol"

        measurement = MagicMock()
        measurement.extract.return_value = None

        criterion = MagicMock()

        pv = ParametricValidation(
            validation_name="None Test",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
        )

        outcome = pv.run("tparams", 1.0, tmp_path)
        assert outcome.passed is False
        assert "could not be extracted" in outcome.details
        criterion.evaluate.assert_not_called()

    def test_name_property(self):
        pv = ParametricValidation(
            validation_name="My Name",
            protocol=MagicMock(),
            measurement=MagicMock(),
            criterion=MagicMock(),
        )
        assert pv.name == "My Name"


class TestPlotTrace:
    def test_creates_figure_file(self, tmp_path):
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.random.randn(100) * 10 - 70
        recording.current = np.full(100, 0.5)

        outpath = plot_trace(recording, tmp_path, "test.pdf", "Test Title")
        assert outpath.exists()
        assert outpath.name == "test.pdf"

    def test_creates_figure_without_current(self, tmp_path):
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.random.randn(100) * 10 - 70
        recording.current = np.full(100, 0.5)

        outpath = plot_trace(
            recording, tmp_path, "no_current.pdf", "No Current", plot_current=False
        )
        assert outpath.exists()
