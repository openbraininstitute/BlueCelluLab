"""Module for plotting analysis results of cell simulations."""

import matplotlib.pyplot as plt
import pathlib
import numpy as np
from typing import Optional, Tuple
from bluecellulab.tools import calculate_rheobase


def generate_cell_thumbnail(
    cell,
    output_path: str = "cell_thumbnail.png"
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a thumbnail image of a Cell object's step response.

    Args:
        cell: The Cell object to generate a thumbnail for.
        output_path: Path where to save the thumbnail image.

    Returns:
        Tuple containing (time, voltage) arrays of the simulation.
    """
    # Calculate step amplitude (130% of threshold current)
    threshold_current = getattr(cell, 'threshold', 0.0)
    if not threshold_current:  # If threshold is 0, None, or False
        threshold_current = calculate_rheobase(cell)
    
    step_amplitude = threshold_current * 1.3  # 130% of threshold
    step_duration = 100.0  # ms

    # Add step current injection
    stim = cell.add_step(
        start_time=50.0,  # Start after a short delay
        stop_time=50.0 + step_duration,
        level=step_amplitude
    )

    # Set up simulation
    sim = cell.simulation if hasattr(cell, 'simulation') else None
    if sim is None:
        from bluecellulab import Simulation
        sim = Simulation()
        sim.add_cell(cell)

    # Run simulation
    total_duration = 50.0 + step_duration + 50.0  # Add some buffer time after step
    sim.run(total_duration)
    # Get recording data
    time = cell.get_time()
    voltage = cell.get_soma_voltage()

    # Create figure with default size (matplotlib's default is [6.4, 4.8])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(time, voltage, 'k-', linewidth=1.5)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Membrane potential (mV)', fontsize=10)
    plt.tight_layout()

    # Create parent directories if they don't exist
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure with 300 DPI
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
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
