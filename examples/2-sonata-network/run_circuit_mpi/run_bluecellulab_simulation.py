"""
Usage:
    python run_bluecellulab_simulation.py --simulation_config <simulation_config> [--save-nwb]
"""

# Following script in Bluenaas
# https://github.com/openbraininstitute/Bluenaas/blob/main/app/core/circuit/simulation-mpi-entrypoint.py

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib
import numpy as np
from bluecellulab import CircuitSimulation
from bluecellulab.reports.manager import ReportManager
import logging
import sys
from neuron import h
from pynwb import NWBHDF5IO, H5DataIO, NWBFile
from pynwb.icephys import (
    CurrentClampSeries,
    IntracellularElectrode,
    VoltageClampSeries,
    VoltageClampStimulusSeries,
)

from bluecellulab.reports.utils import (
    collect_local_payload,
    collect_local_spikes,
    gather_payload_to_rank0,
    payload_to_cells,
    gather_recording_sites,
    prepare_recordings_for_reports,
)

# Use non-interactive backend for matplotlib to avoid display issues
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create module-level logger
logger = logging.getLogger(__name__)

DEFAULT_AGENT_NAME = "obi_small_scale_simulator_v1"
CURRENT_REPORT_WRITER_AGENT_NAME = "obi_small_scale_simulator_v1__current_report_writer"


def _get_report_metadata(simulation_config_data: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    reports = simulation_config_data.get("reports", {}) or {}
    out: Dict[str, Dict[str, str]] = {}

    for report_name, report_cfg in reports.items():
        if not report_cfg.get("enabled", True):
            continue
        if report_cfg.get("type") != "compartment":
            continue

        variable_name = report_cfg.get("variable_name")
        if not variable_name:
            continue

        unit = report_cfg.get("unit")
        if unit is None:
            unit = "mV" if variable_name == "v" else "unknown"

        out[report_name] = {
            "variable_name": variable_name,
            "unit": unit,
        }

    out["__default_voltage__"] = {
        "variable_name": "v",
        "unit": "mV",
    }

    return out


def _build_nwb_results_from_cells(
    cells: Dict[Any, Any],
    simulation_config_data: Dict[str, Any],
) -> Dict[str, Any]:
    report_meta = _get_report_metadata(simulation_config_data)
    results: Dict[str, Any] = {}

    for cell_id, cell in cells.items():
        population = cell_id.population_name
        gid = cell_id.id
        out_key = f"{population}_{gid}"

        try:
            time_ms = np.asarray(cell.get_recording("neuron.h._ref_t"), dtype=float)
        except Exception as exc:
            logger.warning(f"Skipping {out_key}: no time recording found: {exc}")
            continue

        time_s = time_ms / 1000.0
        recordings: Dict[str, Any] = {}

        for report_name, sites in getattr(cell, "report_sites", {}).items():
            meta = report_meta.get(report_name)
            if meta is None:
                continue

            variable_name = meta["variable_name"]
            unit = meta["unit"]

            for site in sites:
                rec_name = site["rec_name"]
                section_name = site["section"]
                segment = float(site["segx"])

                try:
                    values = np.asarray(cell.get_recording(rec_name), dtype=float)
                except Exception as exc:
                    logger.warning(f"Skipping recording '{rec_name}' for {out_key}: {exc}")
                    continue

                recordings[rec_name] = {
                    "variable_name": variable_name,
                    "section": section_name,
                    "segment": segment,
                    "unit": unit,
                    "area_um2": site["area_um2"],
                    "values": values.tolist(),
                }

        results[out_key] = {
            "time": time_s.tolist(),
            "time_unit": "s",
            "recordings": recordings,
        }

    return results


def _has_seclamp_input(simulation_config_data: Dict[str, Any]) -> bool:
    inputs = simulation_config_data.get("inputs", {}) or {}
    return any(str(v.get("module", "")).lower() == "seclamp" for v in inputs.values())


def _get_seclamp_input_def(simulation_config_data: Dict[str, Any]) -> Dict[str, Any] | None:
    inputs = simulation_config_data.get("inputs", {}) or {}
    for _, stim in inputs.items():
        if str(stim.get("module", "")).lower() == "seclamp":
            return stim
    return None


def _reconstruct_seclamp_command(
    simulation_config_data: Dict[str, Any],
    time_s: np.ndarray,
) -> np.ndarray | None:
    """
    Reconstruct SEClamp command waveform in mV from SONATA input config.
    Returns None if no seclamp input exists.
    """
    stim = _get_seclamp_input_def(simulation_config_data)
    if stim is None:
        return None

    t_ms = np.asarray(time_s, dtype=float) * 1000.0

    base_voltage = float(stim["voltage"])
    duration_total = float(stim["duration"])

    durations = stim.get("duration_levels")
    voltages = stim.get("voltage_levels")

    cmd = np.full_like(t_ms, fill_value=base_voltage, dtype=float)

    if durations and voltages:
        durations = [float(x) for x in durations]
        voltages = [float(x) for x in voltages]

        if len(voltages) != len(durations):
            raise ValueError(
                "Invalid SEClamp config: len(voltage_levels) must equal len(duration_levels)"
            )

        cumulative = np.cumsum(durations)

        if durations[0] == 0 and voltages:
            cmd[t_ms >= 0.0] = voltages[0]

        for idx, level in enumerate(voltages):
            start = cumulative[idx]
            stop = cumulative[idx + 1] if idx + 1 < len(cumulative) else duration_total
            cmd[(t_ms >= start) & (t_ms < stop)] = level

        # ensure last level holds until duration_total
        if voltages:
            cmd[t_ms >= cumulative[len(voltages) - 1]] = voltages[-1]

    return cmd


def save_voltage_results_to_nwb(
    results: Dict[str, Any],
    output_path: Union[str, Path],
):
    """Save voltage report results to NWB format."""
    nwbfile = NWBFile(
        session_description="Small Microcircuit Simulation voltage results",
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.now(timezone.utc),
        experimenter="OBI User",
        lab="Virtual Lab",
        institution="OBI",
        experiment_description="Voltage report results",
        session_id=str(uuid.uuid4()),
        was_generated_by=[DEFAULT_AGENT_NAME],
    )

    # Add device and electrode
    device = nwbfile.create_device(
        name="SimulatedElectrode",
        description="Virtual electrode for simulation recording",
    )

    wrote_any = False

    # Add voltage traces
    for cell_id, cell_result in results.items():
        time = np.asarray(cell_result.get("time", []), dtype=float)
        voltage_rec = None
        for _, rec in cell_result.get("recordings", {}).items():
            if rec.get("variable_name") == "v":
                voltage_rec = rec
                break

        if voltage_rec is None:
            logger.warning(f"Skipping {cell_id}: no voltage recording found")
            continue

        voltage = np.asarray(voltage_rec.get("values", []), dtype=float)
        n = min(len(time), len(voltage))
        if n < 2:
            logger.warning(f"Skipping {cell_id}: voltage/time length mismatch or too short")
            continue

        dt = time[1] - time[0]

        electrode = IntracellularElectrode(
            name=f"electrode_{cell_id}",
            description=f"Simulated electrode for {cell_id}",
            device=device,
            location="soma",
            filtering="none",
        )
        nwbfile.add_icephys_electrode(electrode)

        voltage_data = voltage[:n] / 1000.0  # Convert mV to V
        time_rate = 1.0 / dt

        # Create current clamp series
        ics = CurrentClampSeries(
            name=cell_id,
            data=H5DataIO(data=voltage_data, compression="gzip"),
            electrode=electrode,
            rate=time_rate,
            gain=1.0,
            unit="volts",
            description=f"Voltage trace for {cell_id}",
        )
        nwbfile.add_acquisition(ics)
        wrote_any = True

    if not wrote_any:
        logger.warning(f"No voltage traces found for NWB export: {output_path}")
        return

    # Save to file
    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwbfile)

    logger.info(f"Saved voltage results to {output_path}")


def save_current_results_to_nwb(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    simulation_config_data: Dict[str, Any],
):
    nwbfile = NWBFile(
        session_description="Current recordings",
        identifier=str(uuid.uuid4()),
        session_start_time=datetime.now(timezone.utc),
        experimenter="OBI User",
        lab="Virtual Lab",
        institution="OBI",
        experiment_description="Current recordings from simulation",
        session_id=str(uuid.uuid4()),
        was_generated_by=[CURRENT_REPORT_WRITER_AGENT_NAME],
    )

    device = nwbfile.create_device(
        name="SimulatedElectrode",
        description="Virtual electrode for simulation recording",
    )

    has_seclamp = _has_seclamp_input(simulation_config_data)
    wrote_any = False
    sweep_no = 0

    for cell_id, cell_result in results.items():
        time_s = np.asarray(cell_result["time"], dtype=float)
        if len(time_s) < 2:
            logger.warning(f"Skipping {cell_id}: not enough time points")
            continue

        dt_s = time_s[1] - time_s[0]
        if dt_s <= 0:
            logger.warning(f"Skipping {cell_id}: non-positive dt")
            continue

        rate_hz = 1.0 / dt_s

        electrode = IntracellularElectrode(
            name=f"electrode_{cell_id}",
            description=f"Simulated electrode for {cell_id}",
            device=device,
            location="soma",
            filtering="none",
        )
        nwbfile.add_icephys_electrode(electrode)

        if has_seclamp:
            cmd_mv = _reconstruct_seclamp_command(simulation_config_data, time_s)
            if cmd_mv is not None:
                stim_ts = VoltageClampStimulusSeries(
                    name=f"{cell_id}__vcss__sweep__{sweep_no:03d}",
                    data=H5DataIO(data=(cmd_mv / 1000.0), compression="gzip"),
                    electrode=electrode,
                    rate=rate_hz,
                    gain=1.0,
                    unit="volts",
                    description="SEClamp",
                    stimulus_description="SEClamp",
                    sweep_number=sweep_no,
                )
                nwbfile.add_stimulus(stim_ts)

        for rec_key, rec in cell_result.get("recordings", {}).items():
            variable_name = rec["variable_name"]
            if variable_name == "v":
                continue

            values = np.asarray(rec["values"], dtype=float)
            if values.size == 0:
                logger.warning(f"Skipping empty recording '{rec_key}' for {cell_id}")
                continue

            section_name = rec["section"]
            segment = rec["segment"]
            area_um2 = rec["area_um2"]

            if area_um2 is None:
                logger.warning(f"Skipping '{rec_key}' for {cell_id}: missing area_um2")
                continue
            area_um2 = float(area_um2)

            # convert mA/cm2 -> nA
            values_nA = values * area_um2 * 0.01

            if "." in variable_name:
                mech, var = variable_name.split(".", 1)
                nwb_var_name = f"{var}_{mech}"
            else:
                nwb_var_name = variable_name

            seg = f"{segment:.3f}".rstrip("0").rstrip(".")
            location = f"{section_name}({seg})"

            ts = VoltageClampSeries(
                name=f"{cell_id}__vcs__{nwb_var_name}__{location}__sweep__{sweep_no:03d}",
                data=H5DataIO(data=values_nA * 1e-9, compression="gzip"),
                electrode=electrode,
                rate=rate_hz,
                gain=1.0,
                unit="amperes",
                description=nwb_var_name,
                stimulus_description="SEClamp" if has_seclamp else "unknown",
                sweep_number=sweep_no,
            )

            nwbfile.add_acquisition(ts)
            wrote_any = True

    if not wrote_any:
        logger.warning(f"No current traces found for NWB export: {output_path}")
        return

    with NWBHDF5IO(str(output_path), "w") as io:
        io.write(nwbfile)

    logger.info(f"Saved current NWB to {output_path}")


def plot_voltage_traces(results: Dict[str, Any], output_path: Union[str, Path], max_cols: int = 3):
    """Plot voltage traces for all cells in a grid of subplots and save to file.

    Args:
        results: Dictionary containing simulation results for each cell
        output_path: Path where to save the plot (should include .png extension)
        max_cols: Maximum number of columns in the subplot grid
    """
    plotted = []
    for cell_id, cell_result in results.items():
        voltage_key = None
        for rec_key, rec in cell_result.get("recordings", {}).items():
            if rec.get("variable_name") == "v":
                voltage_key = rec_key
                break

        if voltage_key is not None:
            plotted.append((cell_id, cell_result, voltage_key))

    n_cells = len(plotted)
    if n_cells == 0:
        logger.warning("No voltage traces to plot")
        return

    # Calculate grid size
    n_cols = min(max_cols, n_cells)
    n_rows = (n_cells + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 3 * n_rows), squeeze=False, constrained_layout=True
    )

    # Flatten axes for easier iteration
    axes = axes.ravel()

    # Plot each cell's voltage trace in its own subplot
    for idx, (cell_id, cell_result, voltage_key) in enumerate(plotted):
        ax = axes[idx]
        time_s = np.asarray(cell_result["time"], dtype=float)
        time_ms = time_s * 1000.0
        voltage_mv = np.asarray(cell_result["recordings"][voltage_key]["values"], dtype=float)

        ax.plot(time_ms, voltage_mv, linewidth=1)
        ax.set_title(f"Cell {cell_id}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Only label bottom row x-axes
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (ms)", fontsize=8)

        # Only label leftmost column y-axes
        if idx % n_cols == 0:
            ax.set_ylabel("mV", fontsize=8)

    # Turn off unused subplots
    for idx in range(n_cells, len(axes)):
        axes[idx].axis("off")

    # Add a main title
    fig.suptitle(f"Voltage Traces for {n_cells} Cells", fontsize=12)

    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved voltage traces plot to {output_path}")


def get_instantiate_gids_params(
    simulation_config_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine instantiate_gids parameters from simulation config.

    This function gives parameters for sim.instantiate_gids() based on the
    simulation config. See the package BlueCellulab/bluecellulab/circuit_simulation.py
    for more details.

    Args:
        simulation_config_data: Loaded simulation configuration
    Returns:
        Dictionary of parameters for instantiate_gids
    """
    params = {
        # Core parameters - these are the main ones we need to set
        "add_stimuli": False,
        "add_synapses": False,
        "add_minis": False,
        "add_replay": False,
        "add_projections": False,
        "interconnect_cells": True,
        # These will be handled automatically by add_stimuli=True
        "add_noise_stimuli": False,
        "add_hyperpolarizing_stimuli": False,
        "add_relativelinear_stimuli": False,
        "add_pulse_stimuli": False,
        "add_shotnoise_stimuli": False,
        "add_ornstein_uhlenbeck_stimuli": False,
        "add_sinusoidal_stimuli": False,
        "add_linear_stimuli": False,
    }

    # Check for any inputs in the config
    if "inputs" in simulation_config_data and simulation_config_data["inputs"]:
        params["add_stimuli"] = True

        # Log any unsupported input types
        supported_types = {
            "noise",
            "hyperpolarizing",
            "relativelinear",
            "pulse",
            "sinusoidal",
            "linear",
            "shotnoise",
            "ornstein_uhlenbeck",
            "seclamp",
        }

        for input_def in simulation_config_data["inputs"].values():
            module = input_def.get("module", "").lower()
            if module not in supported_types:
                logger.warning(
                    f"Input type '{module}' may not be fully supported by instantiate_gids"
                )

    # Check for synapses and minis in conditions
    if "conditions" in simulation_config_data:
        conditions = simulation_config_data["conditions"]
        if "mechanisms" in conditions and conditions["mechanisms"]:
            params["add_synapses"] = True
            # Check if any mechanism has minis enabled
            for mech in conditions["mechanisms"].values():
                if mech.get("minis_single_vesicle", False):
                    params["add_minis"] = True
                    break

    # Check for spike replay in inputs
    if "inputs" in simulation_config_data:
        for input_def in simulation_config_data["inputs"].values():
            if input_def.get("module") == "synapse_replay":
                params["add_replay"] = True
                params["add_synapses"] = True
                break

    params["add_projections"] = params["add_synapses"] or params["add_replay"]
    return params


def run_bluecellulab(
    simulation_config: Union[str, Path],
    save_nwb: bool = False,
) -> None:
    """Run a simulation using BlueCelluLab backend.

    Args:
        simulation_config: Path to the simulation configuration file
        save_nwb: Whether to save results in NWB format.
    """

    # Get MPI info using NEURON's ParallelContext
    h.nrnmpi_init()
    pc = h.ParallelContext()
    rank = int(pc.id())
    size = int(pc.nhost())

    if rank == 0:
        logger.info("Initializing BlueCelluLab simulation")

    # Load configuration using json
    with open(simulation_config) as f:
        simulation_config_data = json.load(f)

    # Get simulation parameters from config
    t_stop = simulation_config_data["run"]["tstop"]
    dt = simulation_config_data["run"]["dt"]
    v_init = simulation_config_data["conditions"]["v_init"]

    # Get the directory of the simulation config
    sim_config_base_dir = Path(simulation_config).parent
    logger.info(f"sim_config_base_dir: {sim_config_base_dir}")

    # Get manifest path
    OUTPUT_DIR = simulation_config_data.get("manifest", {}).get("$OUTPUT_DIR", "./")
    logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")

    # Get the node_set
    node_set_name = simulation_config_data.get("node_set", "All")

    node_sets_file = sim_config_base_dir / simulation_config_data["node_sets_file"]
    logger.info(f"node_sets_file: {node_sets_file}")

    with open(node_sets_file) as f:
        node_set_data = json.load(f)

    # Get population and node IDs
    if node_set_name not in node_set_data:
        raise KeyError(f"Node set '{node_set_name}' not found in node sets file")

    population = node_set_data[node_set_name]["population"]
    all_node_ids = node_set_data[node_set_name]["node_id"]
    logger.info(f"Population: {population}")
    logger.info(f"All node IDs: {all_node_ids}")

    # Distribute nodes across ranks
    num_nodes = len(all_node_ids)
    nodes_per_rank = num_nodes // size
    remainder = num_nodes % size
    logger.info(
        f"Total nodes: {num_nodes}, Nodes per rank: {nodes_per_rank}, Remainder: {remainder}"
    )

    # Calculate start and end indices for this rank
    start_idx = rank * nodes_per_rank + min(rank, remainder)
    if rank < remainder:
        nodes_per_rank += 1
    end_idx = start_idx + nodes_per_rank
    logger.info(f"Rank {rank}: start_idx={start_idx}, end_idx={end_idx}")

    # Get node IDs for this rank
    rank_node_ids = all_node_ids[start_idx:end_idx]
    logger.info(f"Rank {rank} node IDs: {rank_node_ids}")
    # create cell_ids_for_this_rank
    cell_ids_for_this_rank = [(population, i) for i in rank_node_ids]
    logger.info(f"Rank {rank}: Handling {len(cell_ids_for_this_rank)} cells")

    if not cell_ids_for_this_rank:
        logger.warning(f"Rank {rank}: No cells to process")

    if rank == 0:
        logger.info(f"Running BlueCelluLab simulation with {size} MPI processes")
        logger.info(f"Total cells: {num_nodes}, Cells per rank: ~{num_nodes // size}")
        logger.info(f"Starting simulation: t_stop={t_stop}ms, dt={dt}ms")

    logger.info(
        f"Rank {rank}: Processing {len(rank_node_ids)} cells "
        f"(IDs: {rank_node_ids[0] if rank_node_ids else 'None'}..."
        f"{rank_node_ids[-1] if rank_node_ids else 'None'})"
    )

    # Create simulation
    sim = CircuitSimulation(simulation_config)

    # Get instantiate_gids arguments from config
    instantiate_params = get_instantiate_gids_params(simulation_config_data)

    if rank == 0:
        logger.info("Instantiate arguments from config:")
        for param, value in instantiate_params.items():
            if value:  # Only log parameters that are True
                logger.info(f"  {param}: {value}")

    try:
        logger.info(f"Rank {rank}: Instantiating cells...")
        # Instantiate cells on this rank with arguments from config
        sim.instantiate_gids(cell_ids_for_this_rank, **instantiate_params)

        # Run simulation
        logger.info(f"Rank {rank}: Setting up recordings...")
        recording_index, local_sites_index = prepare_recordings_for_reports(
            sim.cells,
            sim.circuit_access.config,
        )

        logger.info(f"Rank {rank}: Running simulation...")
        sim.run(t_stop, v_init, cvode=False)

        gathered_sites = pc.py_gather(local_sites_index, 0)

        local_payload = collect_local_payload(
            sim.cells,
            cell_ids_for_this_rank,
            recording_index,
        )
        local_spikes = collect_local_spikes(sim, cell_ids_for_this_rank)

        all_payload, all_spikes = gather_payload_to_rank0(pc, local_payload, local_spikes)
        if rank == 0:
            all_sites_index = gather_recording_sites(gathered_sites)
            cells_for_writer = payload_to_cells(all_payload, all_sites_index)

            all_cell_results = _build_nwb_results_from_cells(
                cells_for_writer,
                simulation_config_data,
            )

            report_mgr = ReportManager(sim.circuit_access.config, sim.dt)
            report_mgr.write_all(cells=cells_for_writer, spikes_by_pop=all_spikes)

            if save_nwb:
                output_dir_str = simulation_config_data["output"]["output_dir"]
                output_dir = Path(output_dir_str).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)

                voltage_nwb_path = output_dir / "voltage_report.nwb"
                current_nwb_path = output_dir / "current_report.nwb"

                save_voltage_results_to_nwb(
                    all_cell_results,
                    voltage_nwb_path,
                )

                save_current_results_to_nwb(
                    all_cell_results,
                    current_nwb_path,
                    simulation_config_data,
                )

                # Save voltage traces plot
                plot_path = output_dir / "voltage_traces.png"
                plot_voltage_traces(all_cell_results, plot_path)

    except Exception as e:
        logger.error(f"Rank {rank} failed: {str(e)}", exc_info=True)
        raise
    finally:
        try:
            logger.info(f"Rank {rank}: Cleaning up...")
            pc.barrier()
            h.quit()
            if rank == 0:
                logger.info("All ranks completed. Simulation finished.")
        except Exception as e:
            logger.error(f"Error during cleanup in rank {rank}: {str(e)}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a BlueCelluLab simulation.")
    parser.add_argument(
        "--simulation_config",
        "--config",
        type=str,
        required=True,
        help="Path to the simulation configuration file",
    )
    parser.add_argument("--save-nwb", action="store_true", help="Save results in NWB format")

    args = parser.parse_args()

    # Validate simulation config exists
    config_path = Path(args.simulation_config)
    if not config_path.exists():
        raise RuntimeError(f"Simulation config file not found: {config_path}")

    # Set up comprehensive logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'simulation_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logger.info(f"Logging to {log_file}")

    # Run the simulation
    run_bluecellulab(
        simulation_config=args.simulation_config,
        save_nwb=args.save_nwb,
    )


if __name__ == "__main__":
    main()
