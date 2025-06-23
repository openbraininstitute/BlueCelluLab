"""Module for plotting analysis results of cell simulations."""

import matplotlib.pyplot as plt
import pathlib
import numpy as np
from typing import Optional, Tuple
from bluecellulab.tools import calculate_rheobase
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.cell import Cell
from bluecellulab.simulation.neuron_globals import NeuronGlobals
from bluecellulab import Simulation


def generate_cell_thumbnail(
    template_path: str,
    morphology_path: Optional[str] = None,
    threshold_current: Optional[float] = None,
    holding_current: Optional[float] = None,
    template_format: str = "v6",
    save_figure: bool = True,
    output_path: str = "cell_thumbnail.png",
    show_figure: bool = False,
    v_init: Optional[float] = -80.0,
    celsius: Optional[float] = 34.0,
) -> Tuple[np.ndarray, np.ndarray]: 
    """Generate a thumbnail image of a Cell object's step response.

    Args:
        template_path: Path to the hoc file.
        morphology_path: Path to the morphology file. Default is None.
        threshold_current: Threshold current for the cell. Default is None.
        holding_current: Holding current for the cell. Default is None.
        template_format: Template format. Default is "v6".
        save_figure: Whether to save the figure. Default is True.
        output_path: Path where to save the thumbnail image.
        show_figure: Whether to display the figure. Default is False.
        v_init: Initial membrane potential. Default is -80.0 mV.
        celsius: Temperature in Celsius. Default is 34.0.

    Returns:
        Tuple containing (time, voltage) arrays of the simulation.
    """
    # If the threshold and holding_current is not known,
    # we use calculate_rheobase() to find the threshold_current.
    # For this case, we set the holding_current to 0.0 nA.
    # if threshold_current is None: 
    threshold_current = 1.0
    if holding_current is None:
        holding_current = 0.0

    emodel_properties = EmodelProperties(
        threshold_current=threshold_current,
        holding_current=holding_current
    )

    # Initialise cell
    thumbnail_cell = Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
    )
    holding_current = 0.0
    # Calculate rheobase
    newthreshold_current = calculate_rheobase(
        thumbnail_cell,
        section="soma[0]",
        segx=0.5
    )
    print(f"Threshold current: {newthreshold_current}")
    print(f"Holding current: {holding_current}")
    #delete thumbnail_cell
    thumbnail_cell.delete()

    # Update emodel properties
    emodel_properties = EmodelProperties(
        threshold_current=newthreshold_current,
        holding_current=holding_current
    )

    # Create cell
    thumbnail_cell = Cell(
        template_path=template_path,
        morphology_path=morphology_path,
        template_format=template_format,
        emodel_properties=emodel_properties,
    )
    
    # Add 130% threshold current injection
    step_amplitude = newthreshold_current * 1.1
    step_duration = 2000
    start_time = 700
    thumbnail_cell.add_step(
        start_time=start_time,
        stop_time=start_time+step_duration,
        level=step_amplitude
    )

    # Set up simulation
    sim = thumbnail_cell.simulation if hasattr(thumbnail_cell, 'simulation') else None
    if sim is None:
        sim = Simulation()
        sim.add_cell(thumbnail_cell)

    neuron_globals = NeuronGlobals.get_instance()
    # neuron_globals.temperature = celsius
    # neuron_globals.v_init = v_init

    # Run simulation
    tstop = step_duration+300
    sim.run(tstop)
    # Get recording data
    time = thumbnail_cell.get_time()
    voltage = thumbnail_cell.get_soma_voltage()

    # Create figure with default size (matplotlib's default is [6.4, 4.8])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(time, voltage, 'k-', linewidth=1.5)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Membrane potential (mV)', fontsize=10)
    plt.tight_layout()

    if save_figure:
        # Create parent directories if they don't exist
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure with 300 DPI
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_figure:
        plt.show()
    else:
        plt.close()
    
    thumbnail_cell.delete()
    
    return time, voltage


def plot_iv_curve(
    currents,
    voltages,
    injecting_section,
    injecting_segment,
    recording_section,
    recording_segment,
    show_figure=True,
    save_figure=False,
    output_dir="./",
    output_fname="iv_curve.pdf",
):
    """Plots the IV curve.

    Args:
        currents (list): The injected current levels (nA).
        voltages (list): The corresponding steady-state voltages (mV).
        injecting_section (str): The section in the cell where the current was injected.
        injecting_segment (float): The segment position (0.0 to 1.0) where the current was injected.
        recording_section (str): The section in the cell where spikes were recorded.
        recording_segment (float): The segment position (0.0 to 1.0) where spikes were recorded.
        show_figure (bool): Whether to display the figure. Default is True.
        save_figure (bool): Whether to save the figure. Default is False.
        output_dir (str): The directory to save the figure if save_figure is True. Default is "./".
        output_fname (str): The filename to save the figure as if save_figure is True. Default is "iv_curve.pdf".

    Raises:
        ValueError: If the lengths of currents and voltages do not match.
    """
    if len(currents) != len(voltages):
        raise ValueError("currents and voltages must have the same length")

    plt.figure(figsize=(10, 6))
    plt.plot(currents, voltages, marker='o', linestyle='-', color='b')
    plt.title("I-V Curve")
    plt.xlabel(f"Injected Current [nA] at {injecting_section}({injecting_segment:.2f})")
    plt.ylabel(f"Steady state voltage [mV] at {recording_section}({recording_segment:.2f})")
    plt.grid(True)
    plt.tight_layout()
    if show_figure:
        plt.show()

    if save_figure:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(pathlib.Path(output_dir) / output_fname, format='pdf')
        plt.close()


def plot_fi_curve(
    currents,
    spike_count,
    injecting_section,
    injecting_segment,
    recording_section,
    recording_segment,
    show_figure=True,
    save_figure=False,
    output_dir="./",
    output_fname="fi_curve.pdf",
):
    """Plots the F-I (Frequency-Current) curve.

    Args:
        currents (list): The injected current levels (nA).
        spike_count (list): The number of spikes recorded for each current level.
        injecting_section (str): The section in the cell where the current was injected.
        injecting_segment (float): The segment position (0.0 to 1.0) where the current was injected.
        recording_section (str): The section in the cell where spikes were recorded.
        recording_segment (float): The segment position (0.0 to 1.0) where spikes were recorded.
        show_figure (bool): Whether to display the figure. Default is True.
        save_figure (bool): Whether to save the figure. Default is False.
        output_dir (str): The directory to save the figure if save_figure is True. Default is "./".
        output_fname (str): The filename to save the figure as if save_figure is True. Default is "fi_curve.pdf".

    Raises:
        ValueError: If the lengths of currents and spike counts do not match.
    """
    if len(currents) != len(spike_count):
        raise ValueError("currents and spike count must have the same length")

    plt.figure(figsize=(10, 6))
    plt.plot(currents, spike_count, marker='o')
    plt.title("F-I Curve")
    plt.xlabel(f"Injected Current [nA] at {injecting_section}({injecting_segment:.2f})")
    plt.ylabel(f"Spike Count recorded at {recording_section}({recording_segment:.2f})")
    plt.grid(True)
    plt.tight_layout()
    if show_figure:
        plt.show()

    if save_figure:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(pathlib.Path(output_dir) / output_fname, format='pdf')
        plt.close()
