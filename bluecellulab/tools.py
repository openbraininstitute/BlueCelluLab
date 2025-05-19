# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for calculating certain properties of Neurons."""


from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import logging

import neuron
import numpy as np

import bluecellulab
from bluecellulab.cell import Cell
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.exceptions import UnsteadyCellError
from bluecellulab.simulation.parallel import IsolatedProcess
from bluecellulab.utils import CaptureOutput
from bluecellulab.type_aliases import NeuronSection


logger = logging.getLogger(__name__)


def calculate_input_resistance(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    current_delta: float = 0.01,
    section: str = "soma[0]",
    segx: float = 0.5
) -> float:
    """Calculate the input resistance at rest of the cell."""
    rest_voltage = calculate_SS_voltage(
        template_path, morphology_path, template_format, emodel_properties, 0.0,
        section=section, segx=segx
    )
    step_voltage = calculate_SS_voltage(
        template_path,
        morphology_path,
        template_format,
        emodel_properties,
        current_delta,
        section=section,
        segx=segx
    )

    voltage_delta = step_voltage - rest_voltage

    return voltage_delta / current_delta


def calculate_SS_voltage(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    step_level: float,
    check_for_spiking=False,
    spike_threshold=-20.0,
    section: str = "soma[0]",
    segx: float = 0.5
) -> float:
    """Calculate the steady state voltage at a certain current step."""
    with IsolatedProcess() as runner:
        SS_voltage = runner.apply(
            calculate_SS_voltage_subprocess,
            [
                template_path,
                morphology_path,
                template_format,
                emodel_properties,
                step_level,
                check_for_spiking,
                spike_threshold,
                section,
                segx
            ],
        )
    return SS_voltage


def calculate_SS_voltage_subprocess(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    step_level: float,
    check_for_spiking: bool,
    spike_threshold: float,
    section: str = "soma[0]",
    segx: float = 0.5
) -> float:
    """Subprocess wrapper of calculate_SS_voltage.

    This code should be run in a separate process. If check_for_spiking
    is True, this function will return None if the cell spikes from
    100ms to the end of the simulation indicating no steady state was
    reached.
    """
    cell = bluecellulab.Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
    )

    sec = get_section(cell, section)
    neuron_section = sec
    cell.add_voltage_recording(cell.sections[section], segx)
    cell.add_ramp(500, 5000, step_level, step_level, section=neuron_section, segx=segx)
    simulation = bluecellulab.Simulation()
    simulation.run(1000, cvode=template_accepts_cvode(template_path))
    time = cell.get_time()
    voltage = cell.get_voltage_recording(section=neuron_section, segx=segx)
    SS_voltage = np.mean(voltage[np.where((time < 1000) & (time > 800))])
    cell.delete()

    if check_for_spiking:
        # check for voltage crossings
        if len(np.nonzero(voltage[np.where(time > 100.0)] > spike_threshold)[0]) > 0:
            raise UnsteadyCellError(
                "Cell spikes from 100ms to the end of the simulation."
            )

    return SS_voltage


def holding_current_subprocess(v_hold, enable_ttx, cell_kwargs):
    """Subprocess wrapper of holding_current."""
    cell = bluecellulab.Cell(**cell_kwargs)

    if enable_ttx:
        cell.enable_ttx()

    vclamp = neuron.h.SEClamp(0.5, sec=cell.soma)
    vclamp.rs = 0.01
    vclamp.dur1 = 2000
    vclamp.amp1 = v_hold

    simulation = bluecellulab.Simulation()
    simulation.run(1000, cvode=False)

    i_hold = vclamp.i
    v_control = vclamp.vc

    cell.delete()

    return i_hold, v_control


def holding_current(
    v_hold: float,
    cell_id: int | tuple[str, int],
    circuit_path: str | Path,
    enable_ttx=False,
) -> Tuple[float, float]:
    """Calculate the holding current necessary for a given holding voltage."""
    cell_id = bluecellulab.circuit.node_id.create_cell_id(cell_id)
    circuit_sim = bluecellulab.CircuitSimulation(circuit_path)

    cell_kwargs = circuit_sim.fetch_cell_kwargs(cell_id)
    with IsolatedProcess() as runner:
        i_hold, v_control = runner.apply(
            holding_current_subprocess, [v_hold, enable_ttx, cell_kwargs]
        )

    return i_hold, v_control


def template_accepts_cvode(template_name: str | Path) -> bool:
    """Return True if template_name can be run with cvode."""
    with open(template_name, "r") as template_file:
        template_content = template_file.read()
    if "StochKv" in template_content:
        accepts_cvode = False
    else:
        accepts_cvode = True
    return accepts_cvode


def search_hyp_current(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: Optional[EmodelProperties],
    target_voltage: float,
    min_current: float,
    max_current: float,
) -> float:
    """Search current necessary to bring cell to -85 mV."""
    med_current = min_current + abs(min_current - max_current) / 2
    new_target_voltage = calculate_SS_voltage(
        template_path,
        morphology_path,
        template_format,
        emodel_properties,
        med_current,
    )
    logger.info("Detected voltage: %f" % new_target_voltage)
    if abs(new_target_voltage - target_voltage) < 0.5:
        return med_current
    elif new_target_voltage > target_voltage:
        return search_hyp_current(
            template_path=template_path,
            morphology_path=morphology_path,
            template_format=template_format,
            emodel_properties=emodel_properties,
            target_voltage=target_voltage,
            min_current=min_current,
            max_current=med_current,
        )
    else:  # new_target_voltage < target_voltage:
        return search_hyp_current(
            template_path=template_path,
            morphology_path=morphology_path,
            template_format=template_format,
            emodel_properties=emodel_properties,
            target_voltage=target_voltage,
            min_current=med_current,
            max_current=max_current,
        )


def detect_hyp_current(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    target_voltage: float,
) -> float:
    """Search current necessary to bring cell to -85 mV.

    Compared to using NEURON's SEClamp object, the binary search better
    replicates what experimentalists use
    """
    return search_hyp_current(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
        target_voltage=target_voltage,
        min_current=-1.0,
        max_current=0.0,
    )


def detect_spike_step(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    hyp_level: float,
    inj_start: float,
    inj_stop: float,
    step_level: float,
    section: str = "soma[0]",
    segx: float = 0.5,
    step_thresh: float = -20.
) -> bool:
    """Detect if there is a spike at a certain step level."""
    with IsolatedProcess() as runner:
        spike_detected = runner.apply(
            detect_spike_step_subprocess,
            [
                template_path,
                morphology_path,
                template_format,
                emodel_properties,
                hyp_level,
                inj_start,
                inj_stop,
                step_level,
                section,
                segx,
                step_thresh
            ],
        )
    return spike_detected


def detect_spike_step_subprocess(
    template_path: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    hyp_level: float,
    inj_start: float,
    inj_stop: float,
    step_level: float,
    section: str = "soma[0]",
    segx: float = 0.5,
    step_thresh: float = -20.
) -> bool:
    """Detect if there is a spike at a certain step level."""
    cell = bluecellulab.Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties)

    sec = get_section(cell, section)
    cell.add_voltage_recording(cell.sections[section], segx)

    neuron_section = sec
    cell.add_ramp(0, 5000, hyp_level, hyp_level, section=neuron_section, segx=segx)
    cell.add_ramp(inj_start, inj_stop, step_level, step_level, section=neuron_section, segx=segx)
    simulation = bluecellulab.Simulation()
    simulation.run(int(inj_stop), cvode=template_accepts_cvode(template_path))

    time = cell.get_time()
    voltage = cell.get_voltage_recording(section=neuron_section, segx=segx)
    voltage_step = voltage[np.where((time > inj_start) & (time < inj_stop))]
    spike_detected = detect_spike(voltage_step, step_thresh)

    cell.delete()

    return spike_detected


def detect_spike(voltage: np.ndarray, step_thresh: float = -20.) -> bool:
    """Detect if there is a spike in the voltage trace."""
    if len(voltage) == 0:
        return False
    else:
        return bool(np.max(voltage) > step_thresh)  # bool not np.bool_


def search_threshold_current(
    template_name: str | Path,
    morphology_path: str | Path,
    template_format: str,
    emodel_properties: EmodelProperties | None,
    hyp_level: float,
    inj_start: float,
    inj_stop: float,
    min_current: float,
    max_current: float,
    current_precision: float = 0.01,
    section: str = "soma[0]",
    segx: float = 0.5,
    step_thresh: float = -20.
):
    """Search current necessary to reach threshold."""
    if abs(max_current - min_current) < current_precision:
        return max_current
    med_current = min_current + abs(min_current - max_current) / 2
    logger.info("Med current %d" % med_current)

    spike_detected = detect_spike_step(
        template_name, morphology_path, template_format, emodel_properties,
        hyp_level, inj_start, inj_stop, med_current,
        section=section, segx=segx, step_thresh=step_thresh
    )
    logger.info("Spike threshold detection at: %f nA" % med_current)

    if spike_detected:
        return search_threshold_current(template_name, morphology_path,
                                        template_format, emodel_properties,
                                        hyp_level, inj_start, inj_stop,
                                        min_current, med_current,
                                        current_precision,
                                        section=section, segx=segx,
                                        step_thresh=step_thresh)
    else:
        return search_threshold_current(template_name, morphology_path,
                                        template_format, emodel_properties,
                                        hyp_level, inj_start, inj_stop,
                                        med_current, max_current,
                                        current_precision,
                                        section=section, segx=segx,
                                        step_thresh=step_thresh)


def check_empty_topology() -> bool:
    """Return true if NEURON simulator topology command is empty."""
    with CaptureOutput() as stdout:
        neuron.h.topology()

    return stdout == ['', '']


def calculate_max_thresh_current(cell: Cell,
                                 threshold_voltage: float = -30.0,
                                 section: str = "soma[0]",
                                 segx: float = 0.5) -> float:
    """Calculate the upper bound threshold current.

    Args:
        cell (bluecellulab.cell.Cell): The initialized cell model.
        threshold_voltage (float, optional): Voltage threshold for spike detection. Default is -30.0 mV.
        section (str, optional): The section where current is injected.
        segx (float, optional): Fractional location within the section for current injection.

    Returns:
        float: The upper bound threshold current.
    """
    # Calculate resting membrane potential (rmp)
    rmp = calculate_SS_voltage(
        template_path=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
        step_level=0.0,
        section=section,
        segx=segx
    )

    # Calculate input resistance (rin)
    rin = calculate_input_resistance(
        template_path=cell.template_params.template_filepath,
        morphology_path=cell.template_params.morph_filepath,
        template_format=cell.template_params.template_format,
        emodel_properties=cell.template_params.emodel_properties,
        section=section,
        segx=segx
    )

    # Calculate upperbound threshold current
    upperbound_threshold_current = (threshold_voltage - rmp) / rin
    upperbound_threshold_current = np.min([upperbound_threshold_current, 2.0])

    return upperbound_threshold_current


def calculate_rheobase(cell: Cell,
                       threshold_voltage: float = -20.0,
                       threshold_search_stim_start: float = 300.0,
                       threshold_search_stim_stop: float = 1000.0,
                       section: str = "soma[0]",
                       segx: float = 0.5) -> float:
    """Calculate the rheobase by first computing the upper bound threshold
    current.

    Args:
        cell (bluecellulab.cell.Cell): The initialized cell model.
        threshold_voltage (float, optional): Voltage threshold for spike detection. Default is -20.0 mV.
        threshold_search_stim_start (float, optional): Start time for threshold search stimulation (in ms). Default is 300.0 ms.
        threshold_search_stim_stop (float, optional): Stop time for threshold search stimulation (in ms). Default is 1000.0 ms.
        section (str, optional): The section where current is injected.
        segx (float, optional): Fractional location within the section for current injection.

    Returns:
        float: The rheobase current.
    """
    if cell.template_params.emodel_properties is None:
        raise ValueError("emodel_properties cannot be None")

    # Calculate upper bound threshold current
    upperbound_threshold_current = calculate_max_thresh_current(
        cell,
        threshold_voltage,
        section,
        segx
    )

    # Compute rheobase
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
        section=section,
        segx=segx,
        step_thresh=threshold_voltage
    )

    return rheobase


def validate_section_and_segment(cell: Cell, section_name: str, segment_position: float):
    """Validate a single section and segment position.

    Args:
        cell: The cell model to validate against.
        section_name: The name of the section to validate (e.g., 'soma', 'axon[1]').
        segment_position: The position within the section (e.g., 0.5 for the middle).

    Raises:
        ValueError: If the section or position is invalid.
    """
    # Validate the section
    if section_name not in cell.sections:
        raise ValueError(f"Section '{section_name}' not found in the cell model.")

    # Validate the segment position
    if not (0.0 <= segment_position <= 1.0):
        raise ValueError(f"Segment position must be between 0.0 and 1.0, got {segment_position}.")


def get_section(cell: Cell, section_name: str) -> NeuronSection:
    """Retrieve a NEURON section from the cell by its name.

    Args:
        cell (Cell): The cell object containing the sections.
        section_name (str): The name of the section to retrieve.

    Returns:
        NeuronSection: The NEURON section corresponding to the provided name.

    Raises:
        ValueError: If the section with the specified name does not exist.
    """
    try:
        return cell.sections[section_name]
    except KeyError:
        raise ValueError(f"Section '{section_name}' not found in the cell.")
