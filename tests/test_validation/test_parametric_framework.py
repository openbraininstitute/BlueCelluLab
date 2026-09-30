"""Unit tests for the parametric validation framework."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import efel
import numpy as np
import pytest

from bluecellulab.validation.base import TestResult
from bluecellulab.validation.criterion import GreaterThan
from bluecellulab.validation.measurement import EfelMeasurement
from bluecellulab.validation.parametric_validation import ParametricValidation
from bluecellulab.validation.plotting import plot_trace
from bluecellulab.validation.protocol import SequenceProtocol, StepProtocol


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

    def test_empty_measurement_fails_with_description(self):
        c = GreaterThan(threshold=5.0)
        empty = np.array([])

        assert c.evaluate(empty) is False
        assert "No observations" in c.describe(empty)

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
    def test_extract_returns_none_when_efel_raises(self, mock_efel):
        mock_efel.side_effect = RuntimeError("unknown feature")
        m = EfelMeasurement(feature_name="UnknownFeature")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)

        assert m.extract(recording, stim_start=10.0, stim_end=90.0) is None

    @pytest.mark.parametrize(
        "efel_result",
        [[], [{}], [{"OtherFeature": [1]}]],
    )
    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_returns_none_for_malformed_result(self, mock_efel, efel_result):
        mock_efel.return_value = efel_result
        m = EfelMeasurement(feature_name="Spikecount")
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)

        assert m.extract(recording, stim_start=10.0, stim_end=90.0) is None

    @patch("bluecellulab.validation.measurement.efel.get_feature_values")
    def test_extract_restores_efel_settings(self, mock_efel):
        settings = efel.get_settings()
        original_threshold = settings.Threshold
        original_interp_step = settings.interp_step
        requested_threshold = original_threshold + 5.0
        requested_interp_step = original_interp_step + 0.01

        def extract_feature(*args, **kwargs):
            assert efel.get_settings().Threshold == requested_threshold
            assert efel.get_settings().interp_step == requested_interp_step
            return [{"Spikecount": [1]}]

        mock_efel.side_effect = extract_feature
        m = EfelMeasurement(
            feature_name="Spikecount",
            efel_settings={
                "Threshold": requested_threshold,
                "interp_step": requested_interp_step,
            },
        )
        recording = MagicMock()
        recording.time = np.arange(0, 100, 1.0)
        recording.voltage = np.full(100, -70.0)

        try:
            assert m.extract(recording, stim_start=10.0, stim_end=90.0) == 1
            assert efel.get_settings().Threshold == original_threshold
            assert efel.get_settings().interp_step == original_interp_step
        finally:
            # Keep this test isolated even if an assertion fails before the
            # measurement's restoration code runs.
            efel.set_setting("Threshold", original_threshold)
            efel.set_setting("interp_step", original_interp_step)

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

    def test_stim_timing_is_aligned_to_sample_grid(self):
        p = StepProtocol(dt=0.3)
        assert p.stim_start == pytest.approx(250.2)
        assert p.stim_end == pytest.approx(1600.2)

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_dt(self, dt):
        with pytest.raises(ValueError):
            StepProtocol(dt=dt)

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


class TestSequenceProtocol:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"phases": []},
            {"phases": [(0.0, 1.0)]},
            {"phases": [(1.0, 1.0)], "pre_delay": -1.0},
            {"phases": [(1.0, 1.0)], "post_delay": -1.0},
            {"phases": [(1.0, 1.0)], "dt": 0.0},
            {"phases": [(float("nan"), 1.0)]},
            {"phases": [(1.0, float("inf"))]},
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            SequenceProtocol(**kwargs)

    def test_rejects_out_of_range_measurement_phase(self):
        with pytest.raises(IndexError):
            SequenceProtocol(phases=[(10.0, 1.0)], measurement_phase=1)

    def test_timing_is_aligned_to_sample_grid(self):
        whole_protocol = SequenceProtocol(
            phases=[(5.0, 1.0), (15.0, 2.0)],
            pre_delay=5.0,
            post_delay=5.0,
            dt=10.0,
            absolute_amplitudes=True,
        )
        phase_protocol = SequenceProtocol(
            phases=[(5.0, 1.0), (15.0, 2.0)],
            pre_delay=5.0,
            post_delay=5.0,
            dt=10.0,
            absolute_amplitudes=True,
            measurement_phase=1,
        )

        assert whole_protocol.stim_start == 10.0
        assert whole_protocol.stim_end == 40.0
        assert phase_protocol.stim_start == 20.0
        assert phase_protocol.stim_end == 40.0

    @patch("bluecellulab.validation.protocol.run_stimulus")
    def test_execute_waveform_matches_sample_grid(self, mock_run):
        mock_run.return_value = MagicMock()
        protocol = SequenceProtocol(
            phases=[(5.0, 1.0), (15.0, 2.0)],
            pre_delay=5.0,
            post_delay=5.0,
            dt=10.0,
            absolute_amplitudes=True,
        )

        protocol.execute("template_params", 0.5)

        stimulus = mock_run.call_args.args[1]
        np.testing.assert_allclose(stimulus.current, [0.0, 1.0, 2.0, 2.0, 0.0])
        assert stimulus.stimulus_time == 50.0


class TestParametricValidation:
    @patch("bluecellulab.validation.parametric_validation.plot_trace")
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

    @patch("bluecellulab.validation.parametric_validation.plot_trace")
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

    @patch("bluecellulab.validation.parametric_validation.plot_trace")
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

    @patch("bluecellulab.validation.parametric_validation.plot_trace")
    def test_run_converts_measurement_exception_to_failure(self, mock_plot, tmp_path):
        mock_plot.return_value = tmp_path / "fig.pdf"

        protocol = MagicMock()
        protocol.execute.return_value = MagicMock()
        protocol.stim_start = 250.0
        protocol.stim_end = 1600.0

        measurement = MagicMock()
        measurement.extract.side_effect = RuntimeError("unknown feature")
        criterion = MagicMock()

        pv = ParametricValidation(
            validation_name="Exception Test",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
        )

        outcome = pv.run("tparams", 1.0, tmp_path)

        assert outcome.passed is False
        assert "could not be extracted" in outcome.details
        assert "unknown feature" in outcome.details
        criterion.evaluate.assert_not_called()

    def test_run_converts_protocol_exception_to_failure(self, tmp_path):
        protocol = MagicMock()
        protocol.execute.side_effect = RuntimeError("simulation failed")
        measurement = MagicMock()
        criterion = MagicMock()
        pv = ParametricValidation(
            validation_name="Protocol Exception",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
        )

        outcome = pv.run("tparams", 1.0, tmp_path)

        assert outcome.passed is False
        assert "protocol could not be executed" in outcome.details
        assert "simulation failed" in outcome.details
        assert outcome.figures == []
        measurement.extract.assert_not_called()

    @patch("bluecellulab.validation.parametric_validation.plot_trace")
    def test_run_preserves_verdict_when_plotting_fails(self, mock_plot, tmp_path):
        mock_plot.side_effect = OSError("disk full")
        protocol = MagicMock()
        protocol.execute.return_value = MagicMock()
        protocol.stim_start = 250.0
        protocol.stim_end = 1600.0
        measurement = MagicMock()
        measurement.extract.return_value = 5
        criterion = MagicMock()
        criterion.evaluate.return_value = True
        criterion.describe.return_value = "ok"
        pv = ParametricValidation(
            validation_name="Plot Exception",
            protocol=protocol,
            measurement=measurement,
            criterion=criterion,
        )

        outcome = pv.run("tparams", 1.0, tmp_path)

        assert outcome.passed is True
        assert outcome.figures == []
        assert "Figure could not be generated" in outcome.details

    @patch("bluecellulab.validation.parametric_validation.plot_trace")
    def test_default_figure_filename_is_validation_specific(self, mock_plot, tmp_path):
        mock_plot.side_effect = lambda recording, out_dir, fname, title: (
            Path(out_dir) / fname
        )

        def make_validation(name):
            protocol = MagicMock()
            protocol.execute.return_value = MagicMock()
            protocol.stim_start = 250.0
            protocol.stim_end = 1600.0

            measurement = MagicMock()
            measurement.extract.return_value = 1
            criterion = MagicMock()
            criterion.evaluate.return_value = True
            criterion.describe.return_value = "ok"
            return ParametricValidation(
                validation_name=name,
                protocol=protocol,
                measurement=measurement,
                criterion=criterion,
            )

        first = make_validation("First Validation")
        second = make_validation("Second Validation")

        first_result = first.run("tparams", 1.0, tmp_path)
        second_result = second.run("tparams", 1.0, tmp_path)

        assert first_result.figures == [tmp_path / "First_Validation.pdf"]
        assert second_result.figures == [tmp_path / "Second_Validation.pdf"]
        assert first_result.figures[0] != second_result.figures[0]

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
