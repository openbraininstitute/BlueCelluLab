"""Module for plotting analysis results of cell simulations."""

import matplotlib.pyplot as plt


def plot_iv_curve(currents, voltages, injecting_section, injecting_segment, recording_section, recording_segment):
    """Plots the IV curve.

    Args:
        currents (list): The injected current levels (nA).
        voltages (list): The corresponding steady-state voltages (mV).
        injecting_section (str): The section in the cell where the current was injected.
        injecting_segment (float): The segment position (0.0 to 1.0) where the current was injected.
        recording_section (str): The section in the cell where spikes were recorded.
        recording_segment (float): The segment position (0.0 to 1.0) where spikes were recorded.

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
    plt.show()


def plot_fi_curve(currents, spike_count, injecting_section, injecting_segment, recording_section, recording_segment):
    """Plots the F-I (Frequency-Current) curve.

    Args:
        currents (list): The injected current levels (nA).
        spike_count (list): The number of spikes recorded for each current level.
        injecting_section (str): The section in the cell where the current was injected.
        injecting_segment (float): The segment position (0.0 to 1.0) where the current was injected.
        recording_section (str): The section in the cell where spikes were recorded.
        recording_segment (float): The segment position (0.0 to 1.0) where spikes were recorded.

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
    plt.show()
