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

from bluecellulab.analysis.analysis import compute_plot_fi_curve
from bluecellulab.analysis.analysis import compute_plot_iv_curve
from bluecellulab.analysis.inject_sequence import run_multirecordings_stimulus
from bluecellulab.analysis.inject_sequence import run_stimulus
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


def spiking_test(cell, rheobase, out_dir, spike_threshold_voltage=-30.):
    """Spiking test: cell should spike."""
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recording = run_stimulus(
        cell.template_params,
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

    return {
        "skipped": False,
        "passed": passed,
        "figures": [outpath],
    }


def depolarization_block_test(cell, rheobase, out_dir):
    """Depolarization block test: no depolarization block should be detected."""
    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recording = run_stimulus(
        cell.template_params,
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

    return {
        "skipped": False,
        "passed": not depol_block,
        "figures": [outpath],
    }


def ais_spiking_test(cell, rheobase, out_dir, spike_threshold_voltage=-30.):
    """AIS spiking test: axon should spike before soma."""
    # Check that the cell has an axon
    if len(cell.axonal) == 0:
        return {
            "skipped": True,
            "passed": False,
            "figures": [],
        }

    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recordings = run_multirecordings_stimulus(
        cell.template_params,
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
                "skipped": False,
                "passed": False,
                "figures": [outpath1, outpath2],
            }

    # Check if axon spike happens before soma spike
    passed = bool(axon_recording.spike[0] < soma_recording.spike[0])
    return {
        "skipped": False,
        "passed": passed,
        "figures": [outpath1, outpath2],
    }


def hyperpolarization_test(cell, rheobase, out_dir):
    """Hyperpolarization test: hyperpolarized voltage should be lower than RMP."""
    # Run the stimulus
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.iv(threshold_current=rheobase, threshold_percentage=-40)
    recording = run_stimulus(
        cell.template_params,
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
            "skipped": False,
            "passed": False,
            "figures": [outpath],
        }
    hyperpol_bool = bool(ss_voltage < rmp)

    return {
        "skipped": False,
        "passed": hyperpol_bool,
        "figures": [outpath],
    }


def rin_test(rin):
    """Rin should have an acceptable biological range (< 1000 MOhm)"""
    passed = bool(rin < 1000)

    return {
        "skipped": False,
        "passed": passed,
        "figures": [],
    }


def iv_test(cell, rheobase, out_dir, spike_threshold_voltage=-30.):
    """IV curve should have a positive slope."""
    amps, steady_states = compute_plot_iv_curve(
        cell,
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
            "skipped": False,
            "passed": False,
            "figures": [outpath],
        }
    slope = numpy.polyfit(amps, steady_states, 1)[0]
    passed = bool(slope > 0)
    return {
        "skipped": False,
        "passed": passed,
        "figures": [outpath],
    }


def fi_test(cell, rheobase, out_dir, spike_threshold_voltage=-30.):
    """FI curve should have a positive slope."""
    amps, spike_counts = compute_plot_fi_curve(
        cell,
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
            "skipped": False,
            "passed": False,
            "figures": [outpath],
        }
    slope = numpy.polyfit(amps, spike_counts, 1)[0]
    passed = bool(slope > 0)
    return {
        "skipped": False,
        "passed": passed,
        "figures": [outpath],
    }


def run_validations(cell, cell_name, spike_threshold_voltage=-30):
    """Run all the validations on the cell.

    Args:
        cell (Cell): The cell to validate.
        cell_name (str): The name of the cell, used in the output directory.
        spike_threshold_voltage (float): The voltage threshold for spike detection.
    """
    out_dir = pathlib.Path("memodel_validation_figures") / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # cell = Cell.from_template_parameters(template_params)
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

    # Validation 1: Spiking Test
    logger.debug("Running spiking test")
    spiking_test_result = spiking_test(cell, rheobase, out_dir, spike_threshold_voltage)

    # Validation 2: Depolarization Block Test
    logger.debug("Running depolarization block test")
    depolarization_block_result = depolarization_block_test(cell, rheobase, out_dir)

    # Validation 3: Backpropagating AP Test
    # logger.debug("Running backpropagating AP test")

    # Validation 4: Postsynaptic Potential Test
    # logger.debug("Running postsynaptic potential test")

    # Validation 5: AIS Spiking Test
    logger.debug("Running AIS spiking test")
    ais_spiking_test_result = ais_spiking_test(cell, rheobase, out_dir, spike_threshold_voltage)

    # Validation 6: Hyperpolarization Test
    logger.debug("Running hyperpolarization test")
    hyperpolarization_result = hyperpolarization_test(cell, rheobase, out_dir)

    # Validation 7: Rin Test
    logger.debug("Running Rin test")
    rin_result = rin_test(rin)

    # Validation 8: IV Test
    logger.debug("Running IV test")
    iv_test_result = iv_test(cell, rheobase, out_dir, spike_threshold_voltage)

    # Validation 9: FI Test
    logger.debug("Running FI test")
    fi_test_result = fi_test(cell, rheobase, out_dir, spike_threshold_voltage)

    return {
        "memodel_properties": {
            "holding_current": holding_current,
            "rheobase": rheobase,
            "rin": rin,
        },
        "spiking_test": spiking_test_result,
        "depolarization_block_test": depolarization_block_result,
        "ais_spiking_test": ais_spiking_test_result,
        "hyperpolarization_test": hyperpolarization_result,
        "rin_test": rin_result,
        "iv_test": iv_test_result,
        "fi_test": fi_test_result,
    }
