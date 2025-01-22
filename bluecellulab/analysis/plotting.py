"""Module for plotting analysis results of cell simulations."""

import matplotlib.pyplot as plt


def plot_iv_curve(currents, voltages):
    """Plots the IV curve.

    Args:
        currents (list): The injected current levels (nA).
        voltages (list): The corresponding steady-state voltages (mV).
    Raises:
        ValueError: If the lengths of currents and voltages do not match.
    """
    if len(currents) != len(voltages):
        raise ValueError("currents and voltages must have the same length")

    plt.figure(figsize=(10, 6))
    plt.plot(voltages, currents, marker='o', linestyle='-', color='b')
    plt.title("I-V Curve")
    plt.ylabel("Injected current [nA]")
    plt.xlabel("Steady state voltage [mV]")
    plt.tight_layout()
    plt.show()


def plot_fi_curve(currents, spike_count):
    """Plots the FI curve.

    Args:
        currents (list): The injected current levels (nA).
        spike_count (list): list of spike count for each amplitude.
    Raises:
        ValueError: If the lengths of currents and spikes do not match.
    """
    if len(currents) != len(spike_count):
        raise ValueError("currents and spike count must have the same length")

    plt.figure(figsize=(10, 6))
    plt.plot(currents, spike_count, marker='o')
    plt.title("F-I Curve")
    plt.xlabel("Injected current [nA]")
    plt.ylabel("Spike count")
    plt.tight_layout()
    plt.show()
