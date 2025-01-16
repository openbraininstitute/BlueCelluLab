"""Module for plotting analysis results of cell simulations."""

import matplotlib.pyplot as plt


def plot_iv_curve(currents, voltages):
    """Plots the IV curve.

    Args:
        currents (iterable): The injected current levels (nA).
        voltages (iterable): The corresponding steady-state voltages (mV).
    Raises:
        ValueError: If the lengths of currents and voltages do not match.
    """
    if len(currents) != len(voltages):
        raise ValueError("currents and voltages must have the same length")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, currents, marker='o', linestyle='-', color='b')
    plt.title("I-V Curve")
    plt.ylabel("Injected current [nA]")
    plt.xlabel("Steady state voltage [mV]")
    plt.tight_layout()
    plt.show()
