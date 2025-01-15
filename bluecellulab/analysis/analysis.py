"""Module for analyzing cell simulation results."""
import numpy as np
from bluecellulab.stimulus import StimulusFactory

from bluecellulab.tools import search_threshold_current
from bluecellulab.tools import steady_state_voltage_stimend
from bluecellulab.analysis.inject_sequence import run_stimulus


def compute_iv_curve(cell, stim_start=100.0, duration=500.0, stim_delay=100.0, nb_bins=11):
    """Compute the IV curve from a given cell by injecting a predefined range
    of currents.

    Args:
        cell (bluecellulab.cell.Cell): The initialized cell model.
        stim_start (float): Start time for current injection (in ms). Default is 100.0 ms.
        duration (float): Duration of current injection (in ms). Default is 500.0 ms.
        stim_delay (float): Delay after the stimulation ends (in ms). Default is 100.0 ms.
        nb_bins (int): Number of current injection levels. Default is 11.

    Returns:
        tuple: A tuple containing:
            - list_amp (np.ndarray): The predefined injected step current levels (nA).
            - steady_states (np.ndarray): The corresponding steady-state voltages (mV).
    """

    # compute rheobase
    rheobase = search_threshold_current(
        template_name=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
        hyp_level=0,
        inj_start=stim_start,
        inj_stop=stim_start + duration,
        min_current=0.001,
        max_current=0.5,
        current_precision=0.001
    )

    list_amp = np.linspace(rheobase - 2, rheobase - 0.1, nb_bins)  # [nA]

    steps = []
    times = []
    voltages = []
    # inject step current and record voltage response
    for amp in list_amp:
        stim_factory = StimulusFactory(dt=0.1)
        step_stimulus = stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=stim_delay, amplitude=amp)
        recording = run_stimulus(cell.template_params, step_stimulus, section="soma[0]", segment=0.5)
        steps.append(step_stimulus)
        times.append(recording.time)
        voltages.append(recording.voltage)

    steady_states = []
    # compute steady state response
    for voltage, t in zip(voltages, times):
        steady_state = steady_state_voltage_stimend(stim_start, duration, voltage, t)
        steady_states.append(steady_state)

    return np.array(list_amp), np.array(steady_states)
