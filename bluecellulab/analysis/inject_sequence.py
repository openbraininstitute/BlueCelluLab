"""Module for injecting a sequence of protocols to the cell."""
from __future__ import annotations
from enum import Enum, auto
from typing import NamedTuple, Sequence, Dict, Optional

import neuron
import numpy as np
from bluecellulab.cell.core import Cell
from bluecellulab.cell.template import TemplateParams
from bluecellulab.simulation.parallel import IsolatedProcess
from bluecellulab.simulation.simulation import Simulation
from bluecellulab.stimulus.circuit_stimulus_definitions import Hyperpolarizing
from bluecellulab.stimulus.factory import Stimulus, StimulusFactory
from bluecellulab.tools import validate_section_and_segment


class StimulusName(Enum):
    """Allowed values for the StimulusName."""
    AP_WAVEFORM = auto()
    IDREST = auto()
    IV = auto()
    FIRE_PATTERN = auto()
    POS_CHEOPS = auto()
    NEG_CHEOPS = auto()


class Recording(NamedTuple):
    """A tuple of the current, voltage and time recordings with optional spike
    recordings."""
    current: np.ndarray
    voltage: np.ndarray
    time: np.ndarray
    spike: Optional[np.ndarray] = None


StimulusRecordings = Dict[str, Recording]


def run_multirecordings_stimulus(
    template_params: TemplateParams,
    stimulus: Stimulus,
    section: str,
    segment: float,
    cvode: bool = True,
    add_hypamp: bool = True,
    recording_locations: list[tuple[str, float]] = [("soma[0]", 0.5)],
    enable_spike_detection: bool = False,
    threshold_spike_detection: float = -20.0,
) -> list[Recording]:
    """Creates a cell from template parameters, applies a stimulus, and records
    the response.

    This function simulates the electrical activity of a neuronal cell model by injecting
    a stimulus into a specified section and segment, recording the voltage response, and
    optionally detecting spikes.

    Args:
        template_params (TemplateParams): Parameters required to create the cell from a
            specified template, including morphology and mechanisms.
        stimulus (Stimulus): The stimulus waveform to inject, defined by time and current arrays.
        section (str): Name of the section of the cell where the stimulus is applied.
            (e.g. soma[0])
        segment (float): The normalized position (0.0 to 1.0) along the injecting
            section where the stimulus is applied.
        cvode (bool, optional): Whether to use variable time-step integration. Defaults to True.
        add_hypamp (bool, optional): If True, adds a hyperpolarizing stimulus before applying
            the main stimulus. Defaults to True.
        recording_location (list): List of tuples containing the name of the section of the cell
            where voltage is recorded and the normalized position (0.0 to 1.0) along the recording
            section where voltage is recorded.
            (e.g. [("soma[0]", 0.5), ("dend[0]", 0.5)])
        enable_spike_detection (bool, optional): If True, enables spike detection at the
            recording location. Defaults to False.
        threshold_spike_detection (float, optional): The voltage threshold (mV) for spike detection.
            Defaults to -20 mV.

    Returns:
        list[Recording]: `Recording` objects containing the following:
            - `current` (np.ndarray): The injected current waveform (nA).
            - `voltage` (np.ndarray): The recorded membrane potential (mV) over time.
            - `time` (np.ndarray): The simulation time points (ms).
            - `spike` (np.ndarray or None): The detected spikes, if spike detection is enabled.

    Raises:
        ValueError: If the time, current, and voltage arrays do not have the same length,
            or if the specified sections or segments are not found in the cell model.
    """
    cell = Cell.from_template_parameters(template_params)

    validate_section_and_segment(cell, section, segment)
    for recording_section, recording_segment in recording_locations:
        validate_section_and_segment(cell, recording_section, recording_segment)

    if add_hypamp:
        hyp_stim = Hyperpolarizing(target="", delay=0.0, duration=stimulus.stimulus_time)
        cell.add_replay_hypamp(hyp_stim)

    for recording_section, recording_segment in recording_locations:
        cell.add_voltage_recording(cell.sections[recording_section], recording_segment)

    # Set up spike detection if enabled
    spikes: Optional[np.ndarray] = None
    if enable_spike_detection:
        for recording_section, recording_segment in recording_locations:
            recording_location = f"{recording_section}({str(recording_segment)})"
            cell.start_recording_spikes(None, location=recording_location, threshold=threshold_spike_detection)

    # Inject the stimulus and run the simulation
    iclamp, _ = cell.inject_current_waveform(
        stimulus.time, stimulus.current, section=cell.sections[section], segx=segment
    )
    current_vector = neuron.h.Vector()
    current_vector.record(iclamp._ref_i)

    simulation = Simulation(cell)
    simulation.run(stimulus.stimulus_time, cvode=cvode)

    # Retrieve simulation results
    recordings = []
    current = np.array(current_vector.to_python())
    for recording_section, recording_segment in recording_locations:
        recording_location = f"{recording_section}({str(recording_segment)})"
        voltage = cell.get_voltage_recording(cell.sections[recording_section], recording_segment)
        time = cell.get_time()

        if len(time) != len(voltage) or len(time) != len(current):
            raise ValueError("Time, current, and voltage arrays are not the same length")

        if enable_spike_detection:
            results = cell.get_recorded_spikes(location=recording_location, threshold=threshold_spike_detection)
            if results is not None:
                spikes = np.array(results)
            else:
                spikes = None

        recordings.append(
            Recording(current=current, voltage=voltage, time=time, spike=spikes)
        )

    return recordings


def run_stimulus(
    template_params: TemplateParams,
    stimulus: Stimulus,
    section: str,
    segment: float,
    cvode: bool = True,
    add_hypamp: bool = True,
    recording_section: str = "soma[0]",
    recording_segment: float = 0.5,
    enable_spike_detection: bool = False,
    threshold_spike_detection: float = -20.0,
) -> Recording:
    """Creates a cell from template parameters, applies a stimulus, and records
    the response.

    This function simulates the electrical activity of a neuronal cell model by injecting
    a stimulus into a specified section and segment, recording the voltage response, and
    optionally detecting spikes.

    Args:
        template_params (TemplateParams): Parameters required to create the cell from a
            specified template, including morphology and mechanisms.
        stimulus (Stimulus): The stimulus waveform to inject, defined by time and current arrays.
        section (str): Name of the section of the cell where the stimulus is applied.
            (e.g. soma[0])
        segment (float): The normalized position (0.0 to 1.0) along the injecting
            section where the stimulus is applied.
        cvode (bool, optional): Whether to use variable time-step integration. Defaults to True.
        add_hypamp (bool, optional): If True, adds a hyperpolarizing stimulus before applying
            the main stimulus. Defaults to True.
        recording_section (str): Name of the section of the cell where voltage is recorded.
        recording_segment (float): The normalized position (0.0 to 1.0) along the recording
            section where voltage is recorded.
        enable_spike_detection (bool, optional): If True, enables spike detection at the
            recording location. Defaults to False.
        threshold_spike_detection (float, optional): The voltage threshold (mV) for spike detection.
            Defaults to -20 mV.

    Returns:
        Recording: A `Recording` object containing the following:
            - `current` (np.ndarray): The injected current waveform (nA).
            - `voltage` (np.ndarray): The recorded membrane potential (mV) over time.
            - `time` (np.ndarray): The simulation time points (ms).
            - `spike` (np.ndarray or None): The detected spikes, if spike detection is enabled.

    Raises:
        ValueError: If the time, current, and voltage arrays do not have the same length,
            or if the specified sections or segments are not found in the cell model.
    """
    return run_multirecordings_stimulus(
        template_params=template_params,
        stimulus=stimulus,
        section=section,
        segment=segment,
        cvode=cvode,
        add_hypamp=add_hypamp,
        recording_locations=[(recording_section, recording_segment)],
        enable_spike_detection=enable_spike_detection,
        threshold_spike_detection=threshold_spike_detection,
    )[0]


def apply_multiple_stimuli(
    cell: Cell,
    stimulus_name: StimulusName,
    amplitudes: Sequence[float],
    threshold_based: bool = True,
    section_name: Optional[str] = None,
    segment: float = 0.5,
    n_processes: Optional[int] = None,
    cvode: bool = True,
    add_hypamp: bool = True,
) -> StimulusRecordings:
    """Apply multiple stimuli to the cell on isolated processes.

    Args:
        cell: The cell to which the stimuli are applied.
        stimulus_name: The name of the stimulus to apply.
        amplitudes: The amplitudes of the stimuli to apply.
        threshold_based: Whether to consider amplitudes to be
            threshold percentages or to be raw amplitudes.
        section_name: Section name of the cell where the stimuli are applied.
          If None, the stimuli are applied at the soma[0] of the cell.
        segment: The segment of the section where the stimuli are applied.
        n_processes: The number of processes to use for running the stimuli.
        cvode: True to use variable time-steps. False for fixed time-steps.
        add_hypamp: True to add the cell's holding current stimulus

    Returns:
        A dictionary where the keys are the names of the stimuli and the values
        are the recordings of the cell's response to each stimulus.

    Raises:
        ValueError: If the stimulus name is not recognized.
    """
    res: StimulusRecordings = {}
    stim_factory = StimulusFactory(dt=1.0)
    task_args = []
    section_name = section_name if section_name is not None else "soma[0]"

    # Prepare arguments for each stimulus
    for amplitude in amplitudes:
        if threshold_based:
            thres_perc = amplitude
            amp = None
        else:
            thres_perc = None
            amp = amplitude

        if stimulus_name == StimulusName.AP_WAVEFORM:
            stimulus = stim_factory.ap_waveform(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.IDREST:
            stimulus = stim_factory.idrest(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.IV:
            stimulus = stim_factory.iv(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.FIRE_PATTERN:
            stimulus = stim_factory.fire_pattern(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.POS_CHEOPS:
            stimulus = stim_factory.pos_cheops(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        elif stimulus_name == StimulusName.NEG_CHEOPS:
            stimulus = stim_factory.neg_cheops(
                threshold_current=cell.threshold, threshold_percentage=thres_perc, amplitude=amp
            )
        else:
            raise ValueError("Unknown stimulus name.")

        task_args.append((cell.template_params, stimulus, section_name, segment, cvode, add_hypamp))

    with IsolatedProcess(processes=n_processes) as pool:
        # Map expects a function and a list of argument tuples
        results = pool.starmap(run_stimulus, task_args)

    # Associate each result with a key
    for amplitude, result in zip(amplitudes, results):
        key = f"{stimulus_name}_{amplitude}"
        res[key] = result

    return res
