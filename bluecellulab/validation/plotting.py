# Copyright 2026 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plotting utilities for validation results."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_trace(recording, out_dir, fname, title, plot_current=True):
    """Plot a trace with input current given a recording.

    Args:
        recording: A Recording object with time, voltage, and current attributes.
        out_dir: Output directory for the figure.
        fname: Filename for the saved figure.
        title: Title for the plot.
        plot_current: Whether to overlay the stimulus current.

    Returns:
        Path to the saved figure.
    """
    out_dir = Path(out_dir)
    outpath = out_dir / fname
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.plot(recording.time, recording.voltage, color="black")
    if plot_current:
        current_axis = ax1.twinx()
        current_axis.plot(recording.time, recording.current, color="gray", alpha=0.6)
        current_axis.set_ylabel("Stimulus Current [nA]")
    if title:
        fig.suptitle(title)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Voltage [mV]")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return outpath


def plot_traces(recordings, out_dir, fname, title, labels=None, xlim=None):
    """Plot multiple traces overlaid with stimulus current.

    Args:
        recordings: List of Recording objects.
        out_dir: Output directory for the figure.
        fname: Filename for the saved figure.
        title: Title for the plot.
        labels: Optional labels for each recording.
        xlim: Optional x-axis limits as (min, max).

    Returns:
        Path to the saved figure.
    """
    out_dir = Path(out_dir)
    outpath = out_dir / fname
    fig, ax1 = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    n_colors = len(colors)
    for i, recording in enumerate(recordings):
        if i == 0:
            color = "black"
        else:
            color = colors[(i - 1) % n_colors]
        label = labels[i] if labels is not None else None
        plt.plot(recording.time, recording.voltage, color=color, label=label)
    current_axis = ax1.twinx()
    current_axis.plot(recordings[0].time, recordings[0].current, color="gray", alpha=0.6)
    current_axis.set_ylabel("Stimulus Current [nA]")
    fig.suptitle(title)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Voltage [mV]")
    if labels is not None:
        ax1.legend()
    if xlim is not None:
        ax1.set_xlim(xlim)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return outpath
