"""Unit test for the validation module."""

import itertools
import pytest
from unittest.mock import MagicMock, patch
import pathlib

import bluecellulab.validation.validation as validation


@pytest.fixture
def dummy_recording():
    rec = MagicMock()
    rec.time = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rec.voltage = [-70, -70, 0, 0, -70, -70, -70, -70, -70]
    rec.current = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    rec.spike = [1]
    return rec


@pytest.fixture
def dummy_recordings(dummy_recording):
    rec2 = MagicMock()
    rec2.time = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rec2.voltage = [-70, -70, 0, 0, -70, -70, 0, 0, -70]
    rec2.current = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    rec2.spike = [2]
    return [dummy_recording, rec2]


@pytest.fixture
def dummy_template_params():
    return {"foo": "bar"}


@pytest.fixture
def dummy_out_dir(tmp_path):
    return tmp_path


def test_plot_trace(dummy_recording, dummy_out_dir):
    out = validation.plot_trace(dummy_recording, dummy_out_dir, "trace.pdf", "title")
    assert pathlib.Path(out).exists()


def test_plot_traces(dummy_recordings, dummy_out_dir):
    out = validation.plot_traces(
        dummy_recordings,
        dummy_out_dir,
        "traces.pdf",
        "title",
        ["rec1", "rec2"],
        (0.1, 0.8),
    )
    assert pathlib.Path(out).exists()


@patch("bluecellulab.validation.validation.plot_trace")
@patch("bluecellulab.validation.validation.run_stimulus")
def test_spiking_test(
    mock_run_stimulus, mock_plot_trace, dummy_template_params, dummy_out_dir
):
    # passed case
    rec = MagicMock()
    rec.spike = [1]
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "spiking_test.pdf"
    result = validation.spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "spiking_test.pdf"
    assert result["validation_details"] == "Validation passed: Spikes detected."
    assert result["name"] == "Simulatable Neuron Spiking Validation"

    # failed with None case
    rec.spike = None
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "spiking_test.pdf"
    result = validation.spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Spiking Validation"
    assert result["validation_details"] == "Validation failed: No spikes detected."

    # failed with empty list case
    rec.spike = []
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "spiking_test.pdf"
    result = validation.spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Spiking Validation"
    assert result["validation_details"] == "Validation failed: No spikes detected."


@patch("bluecellulab.validation.validation.plot_trace")
@patch("bluecellulab.validation.validation.run_stimulus")
@patch("bluecellulab.validation.validation.efel.get_feature_values")
def test_depolarization_block_test(
    mock_efel, mock_run_stimulus, mock_plot_trace, dummy_template_params, dummy_out_dir
):
    # passed case
    rec = MagicMock()
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "depol.pdf"
    mock_efel.return_value = [{"depol_block_bool": [0]}]
    result = validation.depolarization_block_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron Depolarization Block Validation"
    assert (
        result["validation_details"]
        == "Validation passed: No depolarization block detected."
    )
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "depol.pdf"

    # failed case
    mock_run_stimulus.return_value = rec
    mock_efel.return_value = [{"depol_block_bool": [1]}]
    result = validation.depolarization_block_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Depolarization Block Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Depolarization block detected."
    )


@patch("bluecellulab.validation.validation.BPAP")
@patch("bluecellulab.validation.validation.Cell")
def test_bpap_test(mock_Cell, mock_Bpap, dummy_template_params, dummy_out_dir):
    bpap = MagicMock()
    bpap.get_amplitudes_and_distances.return_value = (1, [2], [3], [4], [5])
    bpap.validate.return_value = (True, "notes")
    bpap.plot_amp_vs_dist.return_value = dummy_out_dir / "bpap.pdf"
    bpap.plot_recordings.return_value = dummy_out_dir / "bpap_recordings.pdf"
    mock_Bpap.return_value = bpap
    mock_Cell.from_template_parameters.return_value = MagicMock()
    result = validation.bpap_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert (
        result["name"]
        == "Simulatable Neuron Back-propagating Action Potential Validation"
    )
    assert result["validation_details"] == "notes"
    assert len(result["figures"]) == 2
    assert result["figures"][0] == dummy_out_dir / "bpap.pdf"
    assert result["figures"][1] == dummy_out_dir / "bpap_recordings.pdf"


@patch("bluecellulab.validation.validation.efel.get_feature_values")
@patch("bluecellulab.validation.validation.plot_traces")
@patch("bluecellulab.validation.validation.run_multirecordings_stimulus")
@patch("bluecellulab.validation.validation.Cell")
def test_ais_spiking_test(
    mock_Cell, mock_run_multi, mock_plot_traces, mock_efel, dummy_template_params, dummy_out_dir
):
    # passed case
    cell = MagicMock()
    cell.axonal = [1]
    cell.sections = {"axon[0]": MagicMock()}
    mock_Cell.from_template_parameters.return_value = cell
    rec1 = MagicMock(spike=[1])
    rec2 = MagicMock(spike=[2])
    mock_run_multi.return_value = [rec1, rec2]
    mock_plot_traces.side_effect = itertools.cycle(
        [dummy_out_dir / "ais1.pdf", dummy_out_dir / "ais2.pdf"]
    )
    mock_efel.return_value = [{"peak_time": [1]}, {"peak_time": [2]}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert result["validation_details"] == "Validation passed: Axon spikes before soma."
    assert len(result["figures"]) == 2
    assert result["figures"][0] == dummy_out_dir / "ais1.pdf"
    assert result["figures"][1] == dummy_out_dir / "ais2.pdf"

    # soma spikes first case
    rec1 = MagicMock(spike=[3])
    mock_run_multi.return_value = [rec1, rec2]
    mock_efel.return_value = [{"peak_time": [3]}, {"peak_time": [2]}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Axon does not spike before soma."
    )

    # efel extraction failed case
    mock_efel.return_value = [{"peak_time": None}, {"peak_time": [2]}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine spike times for axon or soma."
    )
    mock_efel.return_value = [{"peak_time": []}, {"peak_time": [2]}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine spike times for axon or soma."
    )
    mock_efel.return_value = [{"peak_time": [1]}, {"peak_time": None}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine spike times for axon or soma."
    )
    mock_efel.return_value = [{"peak_time": [1]}, {"peak_time": []}]
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine spike times for axon or soma."
    )

    # no axon case
    cell.axonal = []
    result = validation.ais_spiking_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron AIS Spiking Validation"
    assert (
        result["validation_details"]
        == "Validation skipped: Cell does not have an axon section."
    )
    assert len(result["figures"]) == 0


@patch("bluecellulab.validation.validation.plot_trace")
@patch("bluecellulab.validation.validation.run_stimulus")
@patch("bluecellulab.validation.validation.efel.get_feature_values")
def test_hyperpolarization_test(
    mock_efel, mock_run_stimulus, mock_plot_trace, dummy_template_params, dummy_out_dir
):
    # passed case
    rec = MagicMock()
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "hyper.pdf"
    mock_efel.return_value = [
        {"voltage_base": [0], "steady_state_voltage_stimend": [-1]}
    ]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation passed: Hyperpolarized voltage (-1.00 mV) is lower than RMP (0.00 mV)."
    )
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "hyper.pdf"

    # hyperpolarized voltage > RMP case
    mock_efel.return_value = [
        {"voltage_base": [-80], "steady_state_voltage_stimend": [-70]}
    ]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Hyperpolarized voltage (-70.00 mV) is not lower than RMP (-80.00 mV)."
    )

    # empty rmp / steady state voltage cases
    mock_efel.return_value = [
        {"voltage_base": None, "steady_state_voltage_stimend": [-1]}
    ]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine RMP or steady state voltage."
    )
    mock_efel.return_value = [
        {"voltage_base": [], "steady_state_voltage_stimend": [-1]}
    ]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine RMP or steady state voltage."
    )
    mock_efel.return_value = [
        {"voltage_base": [0], "steady_state_voltage_stimend": None}
    ]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine RMP or steady state voltage."
    )
    mock_efel.return_value = [{"voltage_base": [0], "steady_state_voltage_stimend": []}]
    result = validation.hyperpolarization_test(
        dummy_template_params, 1.0, dummy_out_dir
    )
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Hyperpolarization Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Could not determine RMP or steady state voltage."
    )


def test_rin_test():
    result = validation.rin_test(500)
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron Input Resistance Validation"
    assert (
        result["validation_details"]
        == "Validation passed: Input resistance (Rin) = 500.00 MOhm is smaller than 1000 MOhm."
    )
    assert len(result["figures"]) == 0

    result = validation.rin_test(1500)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron Input Resistance Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Input resistance (Rin) = 1500.00 MOhm is higher than 1000 MOhm, which is not realistic."
    )
    assert len(result["figures"]) == 0


@patch("bluecellulab.validation.validation.compute_plot_iv_curve")
@patch("bluecellulab.validation.validation.Cell")
def test_iv_test(mock_Cell, mock_compute, dummy_template_params, dummy_out_dir):
    # passed case
    mock_Cell.from_template_parameters.return_value = MagicMock()
    mock_compute.return_value = ([1, 2], [1, 2])
    result = validation.iv_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron IV Curve Validation"
    assert "Validation passed" in result["validation_details"]
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "iv_curve.pdf"

    # failed case
    mock_compute.return_value = ([1, 2], [2, 1])
    result = validation.iv_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron IV Curve Validation"
    assert "Validation failed" in result["validation_details"]

    # not enough data points case
    mock_compute.return_value = ([1], [2])
    result = validation.iv_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron IV Curve Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Not enough data points to determine slope."
    )


@patch("bluecellulab.validation.validation.compute_plot_fi_curve")
@patch("bluecellulab.validation.validation.Cell")
def test_fi_test(mock_Cell, mock_compute, dummy_template_params, dummy_out_dir):
    # passed case
    mock_Cell.from_template_parameters.return_value = MagicMock()
    mock_compute.return_value = ([1, 2], [1, 2])
    result = validation.fi_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert result["name"] == "Simulatable Neuron FI Curve Validation"
    assert "Validation passed" in result["validation_details"]
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "fi_curve.pdf"

    # failed case
    mock_compute.return_value = ([1, 2], [2, 1])
    result = validation.fi_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron FI Curve Validation"
    assert "Validation failed" in result["validation_details"]

    # not enough data points case
    mock_compute.return_value = ([1], [2])
    result = validation.fi_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is False
    assert result["name"] == "Simulatable Neuron FI Curve Validation"
    assert (
        result["validation_details"]
        == "Validation failed: Not enough data points to determine slope."
    )


@patch("bluecellulab.validation.validation.plot_trace")
@patch("bluecellulab.validation.validation.run_stimulus")
def test_thumnail_test(
    mock_run_stimulus, mock_plot_trace, dummy_template_params, dummy_out_dir
):
    # passed case
    rec = MagicMock()
    rec.spike = [1]
    mock_run_stimulus.return_value = rec
    mock_plot_trace.return_value = dummy_out_dir / "thumbnail.png"
    result = validation.thumbnail_test(dummy_template_params, 1.0, dummy_out_dir)
    assert result["passed"] is True
    assert len(result["figures"]) == 1
    assert result["figures"][0] == dummy_out_dir / "thumbnail.png"
    assert result["validation_details"] == ""
    assert result["name"] == "thumbnail"


class DummyPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def apply_async(self, func, args=(), kwds=None):
        # Directly call the function synchronously for testing
        class DummyResult:
            def get(self):
                return func(*args, **(kwds or {}))
        return DummyResult()


@patch("bluecellulab.validation.validation.calculate_rheobase")
@patch("bluecellulab.validation.validation.calculate_input_resistance")
@patch("bluecellulab.validation.validation.spiking_test")
@patch("bluecellulab.validation.validation.depolarization_block_test")
@patch("bluecellulab.validation.validation.bpap_test")
@patch("bluecellulab.validation.validation.ais_spiking_test")
@patch("bluecellulab.validation.validation.hyperpolarization_test")
@patch("bluecellulab.validation.validation.rin_test")
@patch("bluecellulab.validation.validation.iv_test")
@patch("bluecellulab.validation.validation.fi_test")
@patch("bluecellulab.validation.validation.thumbnail_test")
@patch("bluecellulab.validation.validation.NestedPool", new=DummyPool)
def test_run_validations(
    mock_thumbnail,
    mock_fi,
    mock_iv,
    mock_rin,
    mock_hyper,
    mock_ais,
    mock_bpap,
    mock_depol,
    mock_spike,
    mock_rin_calc,
    mock_rheo_calc,
):
    template_params = MagicMock()
    template_params.template_filepath = "template.hoc"
    template_params.morph_filepath = "morph.asc"
    template_params.template_format = "v6"
    template_params.emodel_properties = MagicMock()
    cell = MagicMock()
    cell.hypamp = -0.1
    cell.threshold = 2.0
    cell.template_params = template_params
    mock_rin_calc.return_value = 100
    mock_rheo_calc.return_value = 0.5
    mock_spike.return_value = {"passed": True}
    mock_depol.return_value = {"passed": True}
    mock_bpap.return_value = {"passed": True}
    mock_ais.return_value = {"passed": True}
    mock_hyper.return_value = {"passed": True}
    mock_rin.return_value = {"passed": True}
    mock_iv.return_value = {"passed": True}
    mock_fi.return_value = {"passed": True}
    mock_thumbnail.return_value = {"passed": True}
    result = validation.run_validations(cell, "cellname")
    assert result["spiking_test"]["passed"] is True
    assert result["depolarization_block_test"]["passed"] is True
    assert result["bpap_test"]["passed"] is True
    assert result["ais_spiking_test"]["passed"] is True
    assert result["hyperpolarization_test"]["passed"] is True
    assert result["rin_test"]["passed"] is True
    assert result["iv_test"]["passed"] is True
    assert result["fi_test"]["passed"] is True
    assert result["memodel_properties"]["holding_current"] == -0.1
    assert result["memodel_properties"]["rheobase"] == 2.0
    assert result["memodel_properties"]["rin"] == 100

    # test no holding and threshold currents
    cell.hypamp = None
    cell.threshold = None
    result = validation.run_validations(cell, "cellname")
    assert result["memodel_properties"]["holding_current"] == 0.0
    assert result["memodel_properties"]["rheobase"] == 0.5
