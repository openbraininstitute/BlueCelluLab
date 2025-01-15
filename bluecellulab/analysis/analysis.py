"""Module for analyzing cell simulation results."""
try:
    import efel
except ImportError:
    efel = None
import numpy as np
import matplotlib.pyplot as plt

from bluecellulab.stimulus import StimulusFactory
from bluecellulab.tools import calculate_SS_voltage, calculate_input_resistance, search_threshold_current
from bluecellulab.analysis.inject_sequence import run_stimulus


def plot_iv_curve(cell, stim_start=100.0, duration=500.0, post_delay=100.0, threshold_voltage=-30, nb_bins=11):
    """Compute and plot the IV curve from a given cell by injecting a
    predefined range of currents.

    Args:
        cell (bluecellulab.cell.Cell): The initialized cell model.
        stim_start (float): Start time for current injection (in ms). Default is 100.0 ms.
        duration (float): Duration of current injection (in ms). Default is 500.0 ms.
        post_delay (float): Delay after the stimulation ends (in ms). Default is 100.0 ms.
        nb_bins (int): Number of current injection levels. Default is 11.

    Returns:
        tuple: A tuple containing:
            - list_amp (np.ndarray): The predefined injected step current levels (nA).
            - steady_states (np.ndarray): The corresponding steady-state voltages (mV).
    """
    threshold_search_stim_start = 300
    threshold_search_stim_stop = 1000

    # Calculate resting membrane potential (rmp)
    rmp = calculate_SS_voltage(
        template_path=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
        step_level=0.0,
    )

    # Calculate input resistance (rin)
    rin = calculate_input_resistance(
        template_path=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
    )

    # Calculate upperbound threshold current
    upperbound_threshold_current = (threshold_voltage - rmp) / rin
    upperbound_threshold_current = np.min([upperbound_threshold_current, 2.0])

    # compute rheobase
    rheobase = search_threshold_current(
        template_name=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
        hyp_level=cell.template_params.emodel_properties.holding_current,
        inj_start=threshold_search_stim_start,
        inj_stop=threshold_search_stim_stop,
        min_current=cell.template_params.emodel_properties.holding_current,
        max_current=upperbound_threshold_current,
        current_precision=0.005,
    )

    list_amp = np.linspace(rheobase - 2, rheobase - 0.1, nb_bins)  # [nA]

    steps = []
    times = []
    voltages = []
    # inject step current and record voltage response
    for amp in list_amp:
        stim_factory = StimulusFactory(dt=0.1)
        step_stimulus = stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=post_delay, amplitude=amp)
        recording = run_stimulus(cell.template_params, step_stimulus, section="soma[0]", segment=0.5)
        steps.append(step_stimulus)
        times.append(recording.time)
        voltages.append(recording.voltage)

    steady_states = []
    # compute steady state response
    efel.set_setting('Threshold', threshold_voltage)
    for voltage, t in zip(voltages, times):
        trace = {
            'T': t,
            'V': voltage,
            'stim_start': [stim_start],
            'stim_end': [stim_start + duration]
        }
        features_results = efel.get_feature_values([trace], ['steady_state_voltage_stimend'])
        steady_state = features_results[0]['steady_state_voltage_stimend']
        steady_states.append(steady_state)

    # plot I-V curve
    plt.figure(figsize=(10, 6))
    plt.plot(steady_states, list_amp, marker='o', linestyle='-', color='b')
    plt.title("I-V Curve")
    plt.ylabel("Injected current [nA]")
    plt.xlabel("Steady state voltage [mV]")
    plt.tight_layout()
    plt.show()

    return np.array(list_amp), np.array(steady_states)
