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

import logging
import matplotlib.pyplot as plt
import numpy
import pathlib

import efel

from bluecellulab.analysis.analysis import BPAP
from bluecellulab.analysis.analysis import compute_plot_fi_curve
from bluecellulab.analysis.analysis import compute_plot_iv_curve
from bluecellulab.analysis.inject_sequence import run_multirecordings_stimulus
from bluecellulab.analysis.inject_sequence import run_stimulus
from bluecellulab.cell.core import Cell
from bluecellulab.stimulus.factory import IDRestTimings
from bluecellulab.stimulus.factory import StimulusFactory
from bluecellulab.tools import calculate_input_resistance
from bluecellulab.tools import calculate_rheobase

logger = logging.getLogger(__name__)


def plot_trace(recording, out_dir, fname, title):
    """Plot a trace with inout current given a recording."""
    outpath = out_dir / fname
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.plot(recording.time, recording.voltage, color="black")
    current_axis = ax1.twinx()
    current_axis.plot(recording.time, recording.current, color="gray", alpha=0.6)
    current_axis.set_ylabel("Stimulus Current [nA]")
    fig.suptitle(title)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Voltage [mV]")
    fig.tight_layout()
    fig.savefig(outpath)

    return outpath


def plot_traces(recordings, out_dir, fname, title, labels=None, xlim=None):
    """Plot a trace with inout current given a recording."""
    outpath = out_dir / fname
    fig, ax1 = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    N_colors = len(colors)
    for i, recording in enumerate(recordings):
        if i == 0:
            color = "black"
        else:
            color = colors[(i - 1) % N_colors]
        label = labels[i] if labels is not None else None
        plt.plot(recording.time, recording.voltage, color=color, label=label)
    current_axis = ax1.twinx()
    current_axis.plot(recordings[0].time, recordings[0].current, color="gray", alpha=0.6)
    current_axis.set_ylabel("Stimulus Current [nA]")
    fig.suptitle(title)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Voltage [mV]")
    if labels is not None:
        ax1.legend()
    if xlim is not None:
        ax1.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(outpath)

    return outpath


def spiking_test(template_params, rheobase, out_dir, spike_threshold_voltage=-30.):
    """Spiking test: cell should spike."""
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recording = run_stimulus(
        template_params,
        step_stimulus,
        "soma[0]",
        0.5,
        add_hypamp=True,
        enable_spike_detection=True,
        threshold_spike_detection=spike_threshold_voltage,
    )
    passed = recording.spike is not None and len(recording.spike) > 0

    # plotting
    outpath = plot_trace(
        recording,
        out_dir,
        fname="spiking_test.pdf",
        title="Spiking Test - Step at 200% of Rheobase",
    )

    notes = "Validation passed: Spikes detected." if passed else "Validation failed: No spikes detected."
    return {
        "name": "Simulatable Neuron Spiking Validation",
        "passed": passed,
        "validation_details": notes,
        "figures": [outpath],
    }


def depolarization_block_test(template_params, rheobase, out_dir):
    """Depolarization block test: no depolarization block should be detected."""
    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recording = run_stimulus(
        template_params,
        step_stimulus,
        "soma[0]",
        0.5,
        add_hypamp=True,
        enable_spike_detection=False,
    )
    # Check for depolarization block
    trace = {
        "T": recording.time,
        "V": recording.voltage,
        "stim_start": [IDRestTimings.PRE_DELAY.value],
        "stim_end": [IDRestTimings.PRE_DELAY.value + IDRestTimings.DURATION.value],
    }
    features_results = efel.get_feature_values([trace], ["depol_block_bool"])
    depol_block = bool(features_results[0]["depol_block_bool"][0])

    # plotting
    outpath = plot_trace(
        recording,
        out_dir,
        fname="depolarization_block_test.pdf",
        title="Depolarization Block Test - Step at 200% of Rheobase",
    )

    notes = "Validation passed: No depolarization block detected." if not depol_block else "Validation failed: Depolarization block detected."
    return {
        "name": "Simulatable Neuron Depolarization Block Validation",
        "passed": not depol_block,
        "validation_details": notes,
        "figures": [outpath],
    }


def bpap_test(template_params, rheobase, out_dir="./"):
    """Back-propagating action potential test: exponential fit should decay.

    Args:
        template_params (dict): The template parameters for creating the cell.
        rheobase (float): The rheobase current to use for the test.
        out_dir (str): Directory to save the figure.
    """
    amplitude = 10. * rheobase  # Use 1000% of the rheobase current
    bpap = BPAP(Cell.from_template_parameters(template_params))
    bpap.run(duration=1500, amplitude=amplitude)
    soma_amp, dend_amps, dend_dist, apic_amps, apic_dist = bpap.get_amplitudes_and_distances()
    validated, notes = bpap.validate(soma_amp, dend_amps, dend_dist, apic_amps, apic_dist)
    outpath_amp_dist = bpap.plot_amp_vs_dist(
        soma_amp,
        dend_amps,
        dend_dist,
        apic_amps,
        apic_dist,
        show_figure=False,
        save_figure=True,
        output_dir=out_dir,
        output_fname="bpap.pdf"
    )
    outpath_recordings = bpap.plot_recordings(
        show_figure=False,
        save_figure=True,
        output_dir=out_dir,
        output_fname="bpap_recordings.pdf",
    )

    return {
        "name": "Simulatable Neuron Back-propagating Action Potential Validation",
        "validation_details": notes,
        "passed": validated,
        "figures": [outpath_amp_dist, outpath_recordings],
    }


def ais_spiking_test(template_params, rheobase, out_dir, spike_threshold_voltage=-30.):
    """AIS spiking test: axon should spike before soma."""
    name = "Simulatable Neuron AIS Spiking Validation"
    # Check that the cell has an axon
    if len(Cell.from_template_parameters(template_params).axonal) == 0:
        return {
            "name": name,
            "passed": True,
            "validation_details": "Validation skipped: Cell does not have an axon section.",
            "figures": [],
        }

    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recordings = run_multirecordings_stimulus(
        template_params,
        step_stimulus,
        "soma[0]",
        0.5,
        add_hypamp=True,
        recording_locations=[("axon[0]", 0.5), ("soma[0]", 0.5)],
        enable_spike_detection=True,
        threshold_spike_detection=spike_threshold_voltage,
    )
    axon_recording, soma_recording = recordings

    # plotting
    outpath1 = plot_traces(
        recordings,
        out_dir,
        fname="ais_spiking_test.pdf",
        title="AIS Spiking Test - Step at 200% of Rheobase",
        labels=["axon[0]", "soma[0]"],
    )
    outpath2 = plot_traces(
        recordings,
        out_dir,
        fname="ais_spiking_test_zoomed.pdf",
        title="AIS Spiking Test - Step at 200% of Rheobase (zoomed)",
        labels=["axon[0]", "soma[0]"],
        xlim=(IDRestTimings.PRE_DELAY.value, IDRestTimings.PRE_DELAY.value + 100),
    )

    # Check for spiking
    for recording in recordings:
        if recording.spike is None or len(recording.spike) == 0:
            return {
                "name": name,
                "passed": False,
                "validation_details": "Validation failed: No spikes detected in one or both recordings.",
                "figures": [outpath1, outpath2],
            }

    # Check if axon spike happens before soma spike
    passed = bool(axon_recording.spike[0] < soma_recording.spike[0])
    notes = (
        "Validation passed: Axon spikes before soma."
        if passed
        else "Validation failed: Axon does not spike before soma."
    )
    return {
        "name": name,
        "passed": passed,
        "validation_details": notes,
        "figures": [outpath1, outpath2],
    }


def hyperpolarization_test(template_params, rheobase, out_dir):
    """Hyperpolarization test: hyperpolarized voltage should be lower than RMP."""
    name = "Simulatable Neuron Hyperpolarization Validation"
    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.iv(threshold_current=rheobase, threshold_percentage=-40)
    recording = run_stimulus(
        template_params,
        step_stimulus,
        "soma[0]",
        0.5,
        add_hypamp=True,
        enable_spike_detection=False,
    )

    # plotting
    outpath = plot_trace(
        recording,
        out_dir,
        fname="hyperpolarization_test.pdf",
        title="Hyperpolarization Test - Step at -40% of Rheobase",
    )

    # Check for hyperpolarization
    trace = {
        "T": recording.time,
        "V": recording.voltage,
        "stim_start": [IDRestTimings.PRE_DELAY.value],
        "stim_end": [IDRestTimings.PRE_DELAY.value + IDRestTimings.DURATION.value],
    }
    features_results = efel.get_feature_values([trace], ["voltage_base", "steady_state_voltage_stimend"])
    rmp = features_results[0]["voltage_base"][0]
    ss_voltage = features_results[0]["steady_state_voltage_stimend"][0]
    if rmp is None or ss_voltage is None:
        return {
            "name": name,
            "passed": False,
            "validation_details": "Validation failed: Could not determine RMP or steady state voltage.",
            "figures": [outpath],
        }
    hyperpol_bool = bool(ss_voltage < rmp)

    notes = (
        f"Validation passed: Hyperpolarized voltage ({ss_voltage:.2f} mV) is lower than RMP ({rmp:.2f} mV)."
        if hyperpol_bool
        else f"Validation failed: Hyperpolarized voltage ({ss_voltage:.2f} mV) is not lower than RMP ({rmp:.2f} mV)."
    )
    return {
        "name": name,
        "passed": hyperpol_bool,
        "validation_details": notes,
        "figures": [outpath],
    }


def rin_test(rin):
    """Rin should have an acceptable biological range (< 1000 MOhm)"""
    passed = bool(rin < 1000)

    notes = (
        f"Validation passed: Input resistance (Rin) = {rin:.2f} MOhm is smaller than 1000 MOhm."
        if passed
        else f"Validation failed: Input resistance (Rin) = {rin:.2f} MOhm is higher than 1000 MOhm, which is not realistic."
    )
    return {
        "name": "Simulatable Neuron Input Resistance Validation",
        "passed": passed,
        "validation_details": notes,
        "figures": [],
    }


def iv_test(template_params, rheobase, out_dir, spike_threshold_voltage=-30.):
    """IV curve should have a positive slope."""
    name = "Simulatable Neuron IV Curve Validation"
    amps, steady_states = compute_plot_iv_curve(
        Cell.from_template_parameters(template_params),
        rheobase=rheobase,
        threshold_voltage=spike_threshold_voltage,
        nb_bins=5,
        show_figure=False,
        save_figure=True,
        output_dir=out_dir,
        output_fname="iv_curve.pdf")

    outpath = pathlib.Path(out_dir) / "iv_curve.pdf"

    # Check for positive slope
    if len(amps) < 2 or len(steady_states) < 2:
        return {
            "name": name,
            "passed": False,
            "validation_details": "Validation failed: Not enough data points to determine slope.",
            "figures": [outpath],
        }
    slope = numpy.polyfit(amps, steady_states, 1)[0]
    passed = bool(slope > 0)
    notes = (
        f"Validation passed: Slope of IV curve = {slope:.2f} is positive."
        if passed
        else f"Validation failed: Slope of IV curve = {slope:.2f} is not positive."
    )
    return {
        "name": name,
        "validation_details": notes,
        "passed": passed,
        "figures": [outpath],
    }


def fi_test(template_params, rheobase, out_dir, spike_threshold_voltage=-30.):
    """FI curve should have a positive slope."""
    name = "Simulatable Neuron FI Curve Validation"
    amps, spike_counts = compute_plot_fi_curve(
        Cell.from_template_parameters(template_params),
        rheobase=rheobase,
        threshold_voltage=spike_threshold_voltage,
        nb_bins=5,
        show_figure=False,
        save_figure=True,
        output_dir=out_dir,
        output_fname="fi_curve.pdf")

    outpath = pathlib.Path(out_dir) / "fi_curve.pdf"

    # Check for positive slope
    if len(amps) < 2 or len(spike_counts) < 2:
        return {
            "name": name,
            "passed": False,
            "validation_details": "Validation failed: Not enough data points to determine slope.",
            "figures": [outpath],
        }
    slope = numpy.polyfit(amps, spike_counts, 1)[0]
    passed = bool(slope > 0)
    notes = (
        f"Validation passed: Slope of FI curve = {slope:.2f} is positive."
        if passed
        else f"Validation failed: Slope of FI curve = {slope:.2f} is not positive."
    )
    return {
        "name": name,
        "validation_details": notes,
        "passed": passed,
        "figures": [outpath],
    }


def run_validations(
    cell, cell_name, spike_threshold_voltage=-30, output_dir="./memodel_validation_figures"
):
    """Run all the validations on the cell.

    Args:
        cell (Cell): The cell to validate.
        cell_name (str): The name of the cell, used in the output directory.
        spike_threshold_voltage (float): The voltage threshold for spike detection.
        output_dir (str): The directory to save the validation figures.
    """
    out_dir = pathlib.Path(output_dir) / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # get me-model properties
    holding_current = cell.hypamp if cell.hypamp else 0.0
    if cell.threshold:
        rheobase = cell.threshold
    else:
        rheobase = calculate_rheobase(
            cell=cell, section="soma[0]", segx=0.5, threshold_voltage=spike_threshold_voltage
        )
    rin = calculate_input_resistance(
        template_path=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
    )

    logger.debug("Running validations...")
    from bluecellulab.utils import NestedPool
    with NestedPool(processes=8) as pool:
        # Validation 1: Spiking Test
        spiking_test_result_future = pool.apply_async(
            spiking_test,
            (cell.template_params, rheobase, out_dir, spike_threshold_voltage)
        )

        # Validation 2: Depolarization Block Test
        depolarization_block_result_future = pool.apply_async(
            depolarization_block_test,
            (cell.template_params, rheobase, out_dir)
        )

        # Validation 3: Backpropagating AP Test
        bpap_result_future = pool.apply_async(
            bpap_test,
            (cell.template_params, rheobase, out_dir)
        )

        # Validation 4: Postsynaptic Potential Test
        # We have to wait for ProbAMPANMDA_EMS to be present in entitycore to implement this test

        # Validation 5: AIS Spiking Test
        ais_spiking_test_result_future = pool.apply_async(
            ais_spiking_test,
            (cell.template_params, rheobase, out_dir, spike_threshold_voltage)
        )

        # Validation 6: Hyperpolarization Test
        hyperpolarization_result_future = pool.apply_async(
            hyperpolarization_test,
            (cell.template_params, rheobase, out_dir)
        )

        # Validation 7: Rin Test
        rin_result_future = pool.apply_async(
            rin_test,
            (rin,)
        )

        # # Validation 8: IV Test
        iv_test_result_future = pool.apply_async(
            iv_test,
            (cell.template_params, rheobase, out_dir, spike_threshold_voltage)
        )

        # # Validation 9: FI Test
        fi_test_result_future = pool.apply_async(
            fi_test,
            (cell.template_params, rheobase, out_dir, spike_threshold_voltage)
        )

        # Wait for all validations to complete
        spiking_test_result = spiking_test_result_future.get()
        depolarization_block_result = depolarization_block_result_future.get()
        bpap_result = bpap_result_future.get()
        ais_spiking_test_result = ais_spiking_test_result_future.get()
        hyperpolarization_result = hyperpolarization_result_future.get()
        rin_result = rin_result_future.get()
        iv_test_result = iv_test_result_future.get()
        fi_test_result = fi_test_result_future.get()

    return {
        "memodel_properties": {
            "holding_current": holding_current,
            "rheobase": rheobase,
            "rin": rin,
        },
        "spiking_test": spiking_test_result,
        "depolarization_block_test": depolarization_block_result,
        "bpap_test": bpap_result,
        "ais_spiking_test": ais_spiking_test_result,
        "hyperpolarization_test": hyperpolarization_result,
        "rin_test": rin_result,
        "iv_test": iv_test_result,
        "fi_test": fi_test_result,
    }
