"""Module for analyzing cell simulation results."""
try:
    import efel
except ImportError:
    efel = None
import numpy as np

from bluecellulab.analysis.inject_sequence import run_stimulus
from bluecellulab.analysis.plotting import plot_iv_curve, plot_fi_curve
from bluecellulab.stimulus import StimulusFactory
from bluecellulab.tools import calculate_rheobase


def compute_plot_iv_curve(cell,
                          injecting_section="soma[0]",
                          injecting_segment=0.5,
                          recording_section="soma[0]",
                          recording_segment=0.5,
                          stim_start=100.0,
                          duration=500.0,
                          post_delay=100.0,
                          threshold_voltage=-20,
                          nb_bins=11,
                          rheobase=None,
                          show_figure=True,
                          save_figure=False,
                          output_dir="./",
                          output_fname="iv_curve.pdf"):
    """Compute and plot the Current-Voltage (I-V) curve for a given cell by
    injecting a range of currents.

    This function evaluates the relationship between the injected current amplitude and the resulting
    steady-state membrane potential of a neuronal cell model. Currents are injected at a specified section
    and segment, and the steady-state voltage at the recording location is used to construct the I-V curve.

    Args:
        cell (bluecellulab.cell.Cell): The initialized BlueCelluLab cell model.
        injecting_section (str, optional): The name of the section where the stimulus is injected.
            Default is "soma[0]".
        injecting_segment (float, optional): The position along the injecting section (0.0 to 1.0)
            where the stimulus is applied. Default is 0.5.
        recording_section (str, optional): The name of the section where the voltage is recorded.
            Default is "soma[0]".
        recording_segment (float, optional): The position along the recording section (0.0 to 1.0)
            where the voltage is recorded. Default is 0.5.
        stim_start (float, optional): The start time of the current injection (in ms). Default is 100.0 ms.
        duration (float, optional): The duration of the current injection (in ms). Default is 500.0 ms.
        post_delay (float, optional): The delay after the stimulation ends before the simulation stops
            (in ms). Default is 100.0 ms.
        threshold_voltage (float, optional): The voltage threshold (in mV) for detecting a steady-state
            response. Default is -30 mV.
        nb_bins (int, optional): The number of discrete current levels between 0 and the maximum current.
            Default is 11.
        rheobase (float, optional): The rheobase current (in nA) for the cell. If not provided, it will
            be calculated using the `calculate_rheobase` function.
        show_figure (bool): Whether to display the figure. Default is True.
        save_figure (bool): Whether to save the figure. Default is False.
        output_dir (str): The directory to save the figure if save_figure is True. Default is "./".
        output_fname (str): The filename to save the figure as if save_figure is True. Default is "iv_curve.png".

    Returns:
        tuple: A tuple containing:
            - list_amp (np.ndarray): The injected current amplitudes (nA).
            - steady_states (np.ndarray): The corresponding steady-state voltages (mV) recorded at the
              specified location.

    Raises:
        ValueError: If the cell object is invalid, the specified sections/segments are not found, or if
            the simulation results are inconsistent.
    """
    if rheobase is None:
        rheobase = calculate_rheobase(cell=cell, section=injecting_section, segx=injecting_segment)

    list_amp = np.linspace(rheobase - 2, rheobase - 0.1, nb_bins)  # [nA]

    steps = []
    times = []
    voltages = []
    # inject step current and record voltage response
    stim_factory = StimulusFactory(dt=0.1)
    for amp in list_amp:
        step_stimulus = stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=post_delay, amplitude=amp)
        recording = run_stimulus(cell.template_params,
                                 step_stimulus,
                                 section=injecting_section,
                                 segment=injecting_segment,
                                 recording_section=recording_section,
                                 recording_segment=recording_segment)
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
        steady_state = features_results[0]['steady_state_voltage_stimend'][0]
        steady_states.append(steady_state)

    plot_iv_curve(list_amp,
                  steady_states,
                  injecting_section=injecting_section,
                  injecting_segment=injecting_segment,
                  recording_section=recording_section,
                  recording_segment=recording_segment,
                  show_figure=show_figure,
                  save_figure=save_figure,
                  output_dir=output_dir,
                  output_fname=output_fname)

    return np.array(list_amp), np.array(steady_states)


def compute_plot_fi_curve(cell,
                          injecting_section="soma[0]",
                          injecting_segment=0.5,
                          recording_section="soma[0]",
                          recording_segment=0.5,
                          stim_start=100.0,
                          duration=500.0,
                          post_delay=100.0,
                          max_current=0.8,
                          threshold_voltage=-20,
                          nb_bins=11,
                          rheobase=None,
                          show_figure=True,
                          save_figure=False,
                          output_dir="./",
                          output_fname="fi_curve.pdf"):
    """Compute and plot the Frequency-Current (F-I) curve for a given cell by
    injecting a range of currents.

    This function evaluates the relationship between injected current amplitude and the firing rate
    of a neuronal cell model. Currents are injected at a specified section and segment, and the number
    of spikes recorded in the specified recording location is used to construct the F-I curve.

    Args:
        cell (bluecellulab.cell.Cell): The initialized BlueCelluLab cell model.
        injecting_section (str, optional): The name of the section where the stimulus is injected.
            Default is "soma[0]".
        injecting_segment (float, optional): The position along the injecting section (0.0 to 1.0)
            where the stimulus is applied. Default is 0.5.
        recording_section (str, optional): The name of the section where spikes are recorded.
            Default is "soma[0]".
        recording_segment (float, optional): The position along the recording section (0.0 to 1.0)
            where spikes are recorded. Default is 0.5.
        stim_start (float, optional): The start time of the current injection (in ms). Default is 100.0 ms.
        duration (float, optional): The duration of the current injection (in ms). Default is 500.0 ms.
        post_delay (float, optional): The delay after the stimulation ends before the simulation stops
            (in ms). Default is 100.0 ms.
        max_current (float, optional): The maximum amplitude of the injected current (in nA).
            Default is 0.8 nA.
        threshold_voltage (float, optional): The voltage threshold (in mV) for detecting a steady-state
            response. Default is -30 mV.
        nb_bins (int, optional): The number of discrete current levels between 0 and `max_current`.
            Default is 11.
        rheobase (float, optional): The rheobase current (in nA) for the cell. If not provided, it will
            be calculated using the `calculate_rheobase` function.
        show_figure (bool): Whether to display the figure. Default is True.
        save_figure (bool): Whether to save the figure. Default is False.
        output_dir (str): The directory to save the figure if save_figure is True. Default is "./".
        output_fname (str): The filename to save the figure as if save_figure is True. Default is "iv_curve.png".

    Returns:
        tuple: A tuple containing:
            - list_amp (np.ndarray): The injected current amplitudes (nA).
            - spike_count (np.ndarray): The corresponding spike counts for each current amplitude.

    Raises:
        ValueError: If the cell object is invalid or the specified sections/segments are not found.
    """
    if rheobase is None:
        rheobase = calculate_rheobase(cell=cell, section=injecting_section, segx=injecting_segment)

    list_amp = np.linspace(rheobase, max_current, nb_bins)  # [nA]
    steps = []
    spikes = []
    # inject step current and record spike response
    stim_factory = StimulusFactory(dt=0.1)
    for amp in list_amp:
        step_stimulus = stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=post_delay, amplitude=amp)
        recording = run_stimulus(cell.template_params,
                                 step_stimulus,
                                 section=injecting_section,
                                 segment=injecting_segment,
                                 recording_section=recording_section,
                                 recording_segment=recording_segment,
                                 enable_spike_detection=True,
                                 threshold_spike_detection=threshold_voltage)
        steps.append(step_stimulus)
        spikes.append(recording.spike)

    spike_count = [len(spike) for spike in spikes]

    plot_fi_curve(list_amp,
                  spike_count,
                  injecting_section=injecting_section,
                  injecting_segment=injecting_segment,
                  recording_section=recording_section,
                  recording_segment=recording_segment,
                  show_figure=show_figure,
                  save_figure=save_figure,
                  output_dir=output_dir,
                  output_fname=output_fname)

    return np.array(list_amp), np.array(spike_count)
