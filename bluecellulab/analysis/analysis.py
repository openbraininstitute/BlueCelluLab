"""Module for analyzing cell simulation results."""
try:
    import efel
except ImportError:
    efel = None
from itertools import islice
from itertools import repeat
import logging
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from multiprocessing import Pool
import neuron
import numpy as np
import pathlib
import seaborn as sns


from bluecellulab import Cell
from bluecellulab.analysis.inject_sequence import run_stimulus
from bluecellulab.analysis.plotting import plot_iv_curve, plot_fi_curve
from bluecellulab.analysis.utils import exp_decay
from bluecellulab.simulation import Simulation
from bluecellulab.stimulus import StimulusFactory
from bluecellulab.stimulus.circuit_stimulus_definitions import Hyperpolarizing
from bluecellulab.tools import calculate_rheobase


logger = logging.getLogger(__name__)


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
            response. Default is -20 mV.
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

    # inject step current and record voltage response
    stim_factory = StimulusFactory(dt=0.1)
    steps = [
        stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=post_delay, amplitude=amp)
        for amp in list_amp
    ]

    with Pool(len(steps)) as p:
        recordings = p.starmap(
            run_stimulus,
            zip(
                repeat(cell.template_params),
                steps,
                repeat(injecting_section),
                repeat(injecting_segment),
                repeat(True),  # cvode
                repeat(True),  # add_hypamp
                repeat(recording_section),
                repeat(recording_segment),
            )
        )

    steady_states = []
    # compute steady state response
    efel.set_setting('Threshold', threshold_voltage)
    for recording in recordings:
        trace = {
            'T': recording.time,
            'V': recording.voltage,
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
            response. Default is -20 mV.
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
    stim_factory = StimulusFactory(dt=0.1)
    steps = [
        stim_factory.step(pre_delay=stim_start, duration=duration, post_delay=post_delay, amplitude=amp)
        for amp in list_amp
    ]

    with Pool(len(steps)) as p:
        recordings = p.starmap(
            run_stimulus,
            zip(
                repeat(cell.template_params),
                steps,
                repeat(injecting_section),
                repeat(injecting_segment),
                repeat(True),  # cvode
                repeat(True),  # add_hypamp
                repeat(recording_section),
                repeat(recording_segment),
                repeat(True),  # enable_spike_detection
                repeat(threshold_voltage),  # threshold_spike_detection
            )
        )

    spike_count = [len(recording.spike) for recording in recordings]

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


class BPAP:
    # taken from the examples

    def __init__(self, cell: Cell) -> None:
        self.cell = cell
        self.dt = 0.025
        self.stim_start = 1000
        self.stim_duration = 3
        self.basal_cmap = sns.color_palette("crest", as_cmap=True)
        self.apical_cmap = sns.color_palette("YlOrBr_r", as_cmap=True)

    @property
    def start_index(self) -> int:
        """Get the index of the start of the stimulus."""
        return int(self.stim_start / self.dt)

    @property
    def end_index(self) -> int:
        """Get the index of the end of the stimulus."""
        return int((self.stim_start + self.stim_duration) / self.dt)

    def get_recordings(self):
        """Get the soma, basal and apical recordings."""
        all_recordings = self.cell.get_allsections_voltagerecordings()
        soma_rec = None
        dend_rec = {}
        apic_rec = {}
        for key, value in all_recordings.items():
            if "soma" in key:
                soma_rec = value
            elif "dend" in key:
                dend_rec[key] = value
            elif "apic" in key:
                apic_rec[key] = value

        return soma_rec, dend_rec, apic_rec

    def run(self, duration: float, amplitude: float) -> None:
        """Apply depolarization and hyperpolarization at the same time."""
        sim = Simulation()
        sim.add_cell(self.cell)
        self.cell.add_allsections_voltagerecordings()
        self.cell.add_step(start_time=self.stim_start, stop_time=self.stim_start + self.stim_duration, level=amplitude)
        hyperpolarizing = Hyperpolarizing("single-cell", delay=0, duration=duration)
        self.cell.add_replay_hypamp(hyperpolarizing)
        sim.run(duration, dt=self.dt, cvode=False)

    def amplitudes(self, recs) -> list[float]:
        """Return amplitude across given sections."""
        efel_feature_name = "maximum_voltage_from_voltagebase"
        traces = [
            {
                'T': self.cell.get_time(),
                'V': rec,
                'stim_start': [self.stim_start],
                'stim_end': [self.stim_start + self.stim_duration]
            }
            for rec in recs.values()
        ]
        features_results = efel.get_feature_values(traces, [efel_feature_name])
        amps = [feat_res[efel_feature_name][0] for feat_res in features_results]

        return amps

    def distances_to_soma(self, recs) -> list[float]:
        """Return the distance to the soma for each section."""
        res = []
        soma = self.cell.soma
        for key in recs.keys():
            section_name = key.rsplit(".")[-1].split("[")[0]  # e.g. "dend"
            section_idx = int(key.rsplit(".")[-1].split("[")[1].split("]")[0])  # e.g. 0
            attribute_value = getattr(self.cell.cell.getCell(), section_name)
            section = next(islice(attribute_value, section_idx, None))
            # section e.g. cADpyr_L2TPC_bluecellulab_x[0].dend[0]
            res.append(neuron.h.distance(soma(0.5), section(0.5)))
        return res

    def get_amplitudes_and_distances(self):
        soma_rec, dend_rec, apic_rec = self.get_recordings()
        soma_amp = self.amplitudes({"soma": soma_rec})[0]
        dend_amps = None
        dend_dist = None
        apic_amps = None
        apic_dist = None
        if dend_rec:
            dend_amps = self.amplitudes(dend_rec)
            dend_dist = self.distances_to_soma(dend_rec)
        if apic_rec:
            apic_amps = self.amplitudes(apic_rec)
            apic_dist = self.distances_to_soma(apic_rec)

        return soma_amp, dend_amps, dend_dist, apic_amps, apic_dist

    def fit(self, soma_amp, dend_amps, dend_dist, apic_amps, apic_dist):
        """Fit the amplitudes vs distances to an exponential decay function."""
        from scipy.optimize import curve_fit

        popt_dend = None
        if dend_amps and dend_dist:
            dist = [0] + dend_dist  # add soma distance
            amps = [soma_amp] + dend_amps  # add soma amplitude
            popt_dend, _ = curve_fit(exp_decay, dist, amps)

        popt_apic = None
        if apic_amps and apic_dist:
            dist = [0] + apic_dist  # add soma distance
            amps = [soma_amp] + apic_amps  # add soma amplitude
            popt_apic, _ = curve_fit(exp_decay, dist, amps)

        return popt_dend, popt_apic

    def validate(self, soma_amp, dend_amps, dend_dist, apic_amps, apic_dist):
        """Check that the exponential fit is decaying."""
        validated = True
        notes = ""
        popt_dend, popt_apic = self.fit(soma_amp, dend_amps, dend_dist, apic_amps, apic_dist)
        if popt_dend is None:
            logger.debug("No dendritic recordings found.")
            notes += "No dendritic recordings found.\n"
        elif popt_dend[1] <= 0:
            logger.debug("Dendritic fit is not decaying.")
            validated = False
            notes += "Dendritic fit is not decaying.\n"
        else:
            notes += "Dendritic validation passed: dendritic amplitude is decaying with distance relative to soma.\n"
        if popt_apic is None:
            logger.debug("No apical recordings found.")
            notes += "No apical recordings found.\n"
        elif popt_apic[1] <= 0:
            logger.debug("Apical fit is not decaying.")
            validated = False
            notes += "Apical fit is not decaying.\n"
        else:
            notes += "Apical validation passed: apical amplitude is decaying with distance relative to soma.\n"

        return validated, notes

    def plot_amp_vs_dist(
        self,
        soma_amp,
        dend_amps,
        dend_dist,
        apic_amps,
        apic_dist,
        show_figure=True,
        save_figure=False,
        output_dir="./",
        output_fname="bpap.pdf",
    ):
        """Plot the results of the BPAP analysis."""
        popt_dend, popt_apic = self.fit(soma_amp, dend_amps, dend_dist, apic_amps, apic_dist)

        outpath = pathlib.Path(output_dir) / output_fname
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter([0], [soma_amp], marker="^", color='black', label='Soma')
        if dend_amps and dend_dist:
            ax1.scatter(
                dend_dist,
                dend_amps,
                c=dend_dist,
                cmap=self.basal_cmap,
                label='Basal Dendrites',
            )
            if popt_dend is not None:
                x = np.linspace(0, max(dend_dist), 100)
                y = exp_decay(x, *popt_dend)
                ax1.plot(x, y, color='darkgreen', linestyle='--', label='Basal Dendritic Fit')
        if apic_amps and apic_dist:
            ax1.scatter(
                apic_dist,
                apic_amps,
                c=apic_dist,
                cmap=self.apical_cmap,
                label='Apical Dendrites'
            )
            if popt_apic is not None:
                x = np.linspace(0, max(apic_dist), 100)
                y = exp_decay(x, *popt_apic)
                ax1.plot(x, y, color='goldenrod', linestyle='--', label='Apical Fit')
        ax1.set_xlabel('Distance to Soma (um)')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.legend()
        fig.suptitle('Back-propagating Action Potential Analysis')
        fig.tight_layout()
        if save_figure:
            fig.savefig(outpath)
        if show_figure:
            plt.show()

        return outpath

    def plot_one_axis_recordings(self, fig, ax, rec_list, dist, cmap):
        """Plot the soma and dendritic recordings on one axis.

        Args:
            fig (matplotlib.figure.Figure): The figure to plot on.
            ax (matplotlib.axes.Axes): The axis to plot on.
            rec_list (list): List of recordings to plot.
            dist (list): List of distances from the soma for each recording.
            cmap (matplotlib.colors.Colormap): Colormap to use for the recordings.
        """
        time = self.cell.get_time()
        line_collection = LineCollection(
            [np.column_stack([time, rec]) for rec in rec_list],
            array=dist,
            cmap=cmap,
        )
        ax.set_xlim(
            self.stim_start - 0.1,
            self.stim_start + 30
        )
        ax.set_ylim(
            min([min(rec[self.start_index:]) for rec in rec_list]) - 2,
            max([max(rec[self.start_index:]) for rec in rec_list]) + 2
        )
        ax.add_collection(line_collection)
        fig.colorbar(line_collection, label="soma distance (um)", ax=ax)

    def plot_recordings(
        self,
        show_figure=True,
        save_figure=False,
        output_dir="./",
        output_fname="bpap_recordings.pdf",
    ):
        """Plot the recordings from all dendrites."""
        soma_rec, dend_rec, apic_rec = self.get_recordings()
        dend_dist = []
        apic_dist = []
        if dend_rec:
            dend_dist = self.distances_to_soma(dend_rec)
        if apic_rec:
            apic_dist = self.distances_to_soma(apic_rec)
        # add soma_rec to the lists
        dend_rec_list = [soma_rec] + list(dend_rec.values())
        dend_dist = [0] + dend_dist
        apic_rec_list = [soma_rec] + list(apic_rec.values())
        apic_dist = [0] + apic_dist

        outpath = pathlib.Path(output_dir) / output_fname
        fig, (ax1, ax2) = plt.subplots(figsize=(10, 12), nrows=2, sharex=True)

        self.plot_one_axis_recordings(fig, ax1, dend_rec_list, dend_dist, self.basal_cmap)
        self.plot_one_axis_recordings(fig, ax2, apic_rec_list, apic_dist, self.apical_cmap)

        # plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_title('Basal Dendritic Recordings')
        ax2.set_title('Apical Dendritic Recordings')
        ax1.set_ylabel('Voltage (mV)')
        ax2.set_ylabel('Voltage (mV)')
        ax2.set_xlabel('Time (ms)')
        fig.suptitle('Back-propagating Action Potential Recordings')
        fig.tight_layout()
        if save_figure:
            fig.savefig(outpath)
        if show_figure:
            plt.show()

        return outpath
