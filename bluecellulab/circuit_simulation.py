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
"""Ssim class of bluecellulab that loads a circuit simulation to do cell
simulations."""


from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional
import logging
import warnings

from bluecellulab.reports.utils import prepare_recordings_for_reports
import neuron
import numpy as np
import pandas as pd
from pydantic.types import NonNegativeInt
from typing_extensions import deprecated
from typing import Optional

import bluecellulab
from bluecellulab.cell import CellDict
from bluecellulab.cell.sonata_proxy import SonataProxy
from bluecellulab.circuit import CellId, SimulationValidator, SynapseProperty
from bluecellulab.circuit.circuit_access import (
    CircuitAccess,
    BluepyCircuitAccess,
    SonataCircuitAccess,
    get_synapse_connection_parameters
)
from bluecellulab.circuit.config import SimulationConfig
from bluecellulab.circuit.format import determine_circuit_format, CircuitFormat
from bluecellulab.circuit.node_id import create_cell_id, create_cell_ids
from bluecellulab.circuit.simulation_access import BluepySimulationAccess, SimulationAccess, SonataSimulationAccess, _sample_array
from bluecellulab.importer import load_mod_files
from bluecellulab.rngsettings import RNGSettings
from bluecellulab.simulation.neuron_globals import NeuronGlobals
from bluecellulab.stimulus.circuit_stimulus_definitions import Noise, OrnsteinUhlenbeck, RelativeOrnsteinUhlenbeck, RelativeShotNoise, ShotNoise
import bluecellulab.stimulus.circuit_stimulus_definitions as circuit_stimulus_definitions
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.simulation import (
    set_global_condition_parameters,
)
from bluecellulab.synapse.synapse_types import SynapseID

logger = logging.getLogger(__name__)


@deprecated("SSim will be removed, use CircuitSimulation instead.")
class SSim:
    """Class that loads a circuit simulation to do cell simulations."""


class CircuitSimulation:
    """Class that loads a circuit simulation to do cell simulations."""

    @load_mod_files
    def __init__(
        self,
        simulation_config: str | Path | SimulationConfig,
        dt: Optional[float] = None,
        record_dt: Optional[float] = None,
        base_seed: Optional[NonNegativeInt] = None,
        rng_mode: Optional[str] = None,
        print_cellstate: bool = False,
        parallel_context=None,
        save_time: Optional[float] = None,
    ):
        """

        Parameters
        ----------
        simulation_config : Absolute filename of the simulation config file.
        dt : Timestep of the simulation
        record_dt : Sampling interval of the recordings
        base_seed : Base seed used for this simulation. Setting this
                    will override the value set in the simulation config.
        rng_mode : String with rng mode, if not specified mode is taken from
                    simulation config. Possible values are Compatibility, Random123
                    and UpdatedMCell.
        print_cellstate:
                    Flag to use NEURON prcellstate for simulation GIDs
        parallel_context:
                    Optional NEURON ParallelContext to use for MPI runs.
                    If provided, CircuitSimulation will reuse this object
                    instead of creating a new ParallelContext internally.
                    This is useful when the caller already initialized MPI
                    and manages rank/nhost externally.
        save_time:
            Time (ms) when `prcellstate` is dumped. If None (default), dump at
            the end of the simulation (tstop). Use 0 to dump immediately after
            initialization.
        """
        self.record_dt = record_dt

        self.circuit_format = determine_circuit_format(simulation_config)
        if self.circuit_format == CircuitFormat.SONATA:
            self.circuit_access: CircuitAccess = SonataCircuitAccess(simulation_config)
            self.simulation_access: SimulationAccess = SonataSimulationAccess(simulation_config)
        else:
            self.circuit_access = BluepyCircuitAccess(simulation_config)
            self.simulation_access = BluepySimulationAccess(simulation_config)
            SimulationValidator(self.circuit_access).validate()

        self.dt = dt if dt is not None else (self.circuit_access.config.dt or 0.025)
        pc = parallel_context if parallel_context is not None else neuron.h.ParallelContext()
        self.pc = pc if int(pc.nhost()) > 1 or print_cellstate else None
        self.print_cellstate = print_cellstate
        self.save_time = save_time

        self.rng_settings = RNGSettings.get_instance()
        self.rng_settings.set_seeds(
            rng_mode,
            self.circuit_access.config,
            base_seed=base_seed
        )

        self.cells: CellDict = CellDict()

        self.gids_instantiated = False

        self.spike_threshold = self.circuit_access.config.spike_threshold
        self.spike_location = self.circuit_access.config.spike_location

        self.projections: list[str] | str | None = None

        condition_parameters = self.circuit_access.config.condition_parameters()
        set_global_condition_parameters(condition_parameters)

        self._gid_stride = 1_000_000  # must exceed the max node_id in any population
        self._pop_index: dict[str, int] = {"": 0}

    def instantiate_gids(
        self,
        cells: int | tuple[str, int] | list[int | tuple[str, int]],
        add_replay: bool = False,
        add_stimuli: bool = False,
        add_synapses: bool = False,
        add_minis: bool = False,
        add_noise_stimuli: bool = False,
        add_hyperpolarizing_stimuli: bool = False,
        add_relativelinear_stimuli: bool = False,
        add_pulse_stimuli: bool = False,
        add_projections: bool | list[str] | str = False,
        intersect_pre_gids: Optional[list] = None,
        interconnect_cells: bool = True,
        pre_spike_trains: None | dict[tuple[str, int], Iterable] | dict[int, Iterable] = None,
        add_shotnoise_stimuli: bool = False,
        add_ornstein_uhlenbeck_stimuli: bool = False,
        add_sinusoidal_stimuli: bool = False,
        add_linear_stimuli: bool = False,
    ):
        """Instantiate a list of cells.

        Parameters
        ----------
        cells :
               List of cell ids. When a single element, it will be converted to a list
        add_replay : Add presynaptic spiketrains from the large simulation
                     If pre_spike_trains is combined with this option the
                     spiketrains will be merged
        add_stimuli : Add the same stimuli as in the large simulation
        add_synapses :
                       Add the touch-detected synapses, as described by the
                       circuit to the cell
                       (This option only influence the 'creation' of synapses,
                       it doesn't add any connections)
                       Default value is False
        add_minis : Add synaptic minis to the synapses
                    (this requires add_synapses=True)
                    Default value is False
        add_noise_stimuli :
                            Process the 'noise' stimuli blocks of the
                            simulation config,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        add_hyperpolarizing_stimuli : Process the 'hyperpolarizing' stimuli
                                      blocks of the simulation config.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_relativelinear_stimuli : Process the 'relativelinear' stimuli
                                     blocks of the simulation config.
                                     Setting add_stimuli=True,
                                     will automatically set this option to
                                     True.
        add_pulse_stimuli : Process the 'pulse' stimuli
                            blocks of the simulation config.
                            Setting add_stimuli=True,
                            will automatically set this option to
                            True.
        add_projections: Control whether projection edge populations are considered when adding synapses.
                            * ``False`` (default): intrinsic connectivity only (no projection edge populations)
                            * ``True``: intrinsic connectivity + all projection edge populations
                            * ``list[str] | str``: Intrinsic connectivity plus the specified projection edge population name(s).

                            Note:
                                Names refer to SONATA edge population names (``SnapCircuit.edges`` keys).
        intersect_pre_gids : list of gids
                             Only add synapses to the cells if their
                             presynaptic gid is in this list
        interconnect_cells : When multiple gids are instantiated,
                             interconnect the cells with real (non-replay)
                             synapses. When this option is combined with
                             add_replay, replay spiketrains will only be added
                             for those presynaptic cells that are not in the
                             network that's instantiated.
                             This option requires add_synapses=True
        pre_spike_trains : A dictionary with keys the presynaptic gids, and
                           values the list of spike timings of the
                           presynaptic cells with the given gids.
                           If this option is used in combination with
                           add_replay=True, the spike trains for the same
                           gids will be automatically merged
        add_shotnoise_stimuli :
                            Process the 'shotnoise' stimuli blocks of the
                            simulation config,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        add_ornstein_uhlenbeck_stimuli :
                            Process the 'ornstein_uhlenbeck' stimuli blocks
                            of the simulation config,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        add_sinusoidal_stimuli : Process the 'sinusoidal' stimuli
                            blocks of the simulation config.
                            Setting add_stimuli=True,
                            will automatically set this option to
                            True.
        add_linear_stimuli : Process the 'linear' stimuli
                                blocks of the simulation config.
                                Setting add_stimuli=True,
                                will automatically set this option to
                                True.
        """
        if not isinstance(cells, list):
            cells = [cells]

        # convert to CellId objects
        cell_ids: list[CellId] = create_cell_ids(cells)
        if intersect_pre_gids is not None:
            pre_gids: Optional[list[CellId]] = create_cell_ids(intersect_pre_gids)
        else:
            pre_gids = None

        # if pre_spike_trains take int as key then convert to CellId
        if pre_spike_trains is not None:
            if not isinstance(next(iter(pre_spike_trains.keys())), tuple):
                pre_spike_trains = {
                    create_cell_id(gid): pre_spike_trains[gid]  # type: ignore
                    for gid in pre_spike_trains
                }

        if self.gids_instantiated:
            raise BluecellulabError(
                "instantiate_gids() is called twice on the "
                "same CircuitSimumation, this is not supported")
        else:
            self.gids_instantiated = True

        if pre_spike_trains or add_replay:
            if add_synapses is False:
                raise BluecellulabError("You need to set add_synapses to True "
                                        "if you want to specify use add_replay or "
                                        "pre_spike_trains")

        # legacy for backward compatibility
        if add_projections is None:
            add_projections = False

        if add_projections is True:
            self.projections = self.circuit_access.config.get_all_projection_names()
        elif add_projections is False:
            self.projections = None
        else:
            self.projections = add_projections

        self._add_cells(cell_ids)
        if add_synapses:
            self._add_synapses(
                pre_gids=pre_gids,
                add_minis=add_minis)
        if add_replay or interconnect_cells or pre_spike_trains:
            if add_replay and not add_synapses:
                raise BluecellulabError("add_replay option can not be used if "
                                        "add_synapses is False")
            if self.pc is not None:
                self._init_pop_index_mpi()
                self._register_gids_for_mpi()
                self.pc.barrier()
                self.pc.setup_transfer()
                self.pc.set_maxstep(1.0)
            self._add_connections(add_replay=add_replay,
                                  interconnect_cells=interconnect_cells,
                                  user_pre_spike_trains=pre_spike_trains)  # type: ignore
        if add_stimuli:
            add_noise_stimuli = True
            add_hyperpolarizing_stimuli = True
            add_relativelinear_stimuli = True
            add_pulse_stimuli = True
            add_sinusoidal_stimuli = True
            add_shotnoise_stimuli = True
            add_ornstein_uhlenbeck_stimuli = True
            add_linear_stimuli = True

        if add_noise_stimuli or \
                add_hyperpolarizing_stimuli or \
                add_pulse_stimuli or \
                add_relativelinear_stimuli or \
                add_shotnoise_stimuli or \
                add_ornstein_uhlenbeck_stimuli or \
                add_sinusoidal_stimuli or \
                add_linear_stimuli:
            self._add_stimuli(
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli,
                add_shotnoise_stimuli=add_shotnoise_stimuli,
                add_ornstein_uhlenbeck_stimuli=add_ornstein_uhlenbeck_stimuli,
                add_sinusoidal_stimuli=add_sinusoidal_stimuli,
                add_linear_stimuli=add_linear_stimuli
            )

        self.recording_index, self.sites_index = prepare_recordings_for_reports(
            cells=self.cells,
            simulation_config=self.circuit_access.config
        )

        # add spike recordings
        for cell in self.cells.values():
            if not cell.is_recording_spikes(self.spike_location, threshold=self.spike_threshold):
                cell.start_recording_spikes(None, location=self.spike_location, threshold=self.spike_threshold)

    def _add_stimuli(self, add_noise_stimuli=False,
                     add_hyperpolarizing_stimuli=False,
                     add_relativelinear_stimuli=False,
                     add_pulse_stimuli=False,
                     add_shotnoise_stimuli=False,
                     add_ornstein_uhlenbeck_stimuli=False,
                     add_sinusoidal_stimuli=False,
                     add_linear_stimuli=False
                     ) -> None:
        """Instantiate all the stimuli."""
        stimuli_entries = self.circuit_access.config.get_all_stimuli_entries()
        # Also add the injections / stimulations as in the cortical model
        # check in which StimulusInjects the gid is a target
        # Every noise or shot noise stimulus gets a new seed
        noisestim_count = 0
        shotnoise_stim_count = 0
        ornstein_uhlenbeck_stim_count = 0

        # Pre-fetch compartment sets (if available) to detect when a stimulus
        # target refers to a named compartment_set rather than a node_set.
        compartment_sets: dict[str, dict[str, Any]] | None = None
        try:
            compartment_sets = self.circuit_access.config.get_compartment_sets()
        except ValueError:
            pass

        for stimulus in stimuli_entries:
            # Build a unified list of (cell_id, section, segx, section_name) targets
            targets: list[tuple] = []

            if stimulus.compartment_set is not None:
                targets = self._targets_from_compartment_set(stimulus, compartment_sets)
            elif stimulus.node_set is not None:
                gids_of_target = self.circuit_access.get_target_cell_ids(stimulus.node_set)
                for cell_id in self.cells:
                    if cell_id not in gids_of_target:
                        continue
                    sec = self.cells[cell_id].soma
                    sec_name = sec.name().split(".")[-1]
                    targets.append((cell_id, sec, 0.5, sec_name))
            else:
                raise ValueError(
                    f"Stimulus '{stimulus}' has neither node_set nor compartment_set; "
                    "cannot resolve target location."
                )

            for cell_id, sec, segx, sec_name in targets:
                if isinstance(stimulus, circuit_stimulus_definitions.Noise):
                    if add_noise_stimuli:
                        self.cells[cell_id].add_replay_noise(stimulus, noise_seed=None, noisestim_count=noisestim_count, section=sec, segx=segx)
                elif isinstance(stimulus, circuit_stimulus_definitions.Hyperpolarizing):
                    if add_hyperpolarizing_stimuli:
                        self.cells[cell_id].add_replay_hypamp(stimulus, section=sec, segx=segx)
                elif isinstance(stimulus, circuit_stimulus_definitions.Pulse):
                    if add_pulse_stimuli:
                        self.cells[cell_id].add_pulse(stimulus, section=sec, segx=segx)
                elif isinstance(stimulus, circuit_stimulus_definitions.Linear):
                    if add_linear_stimuli:
                        self.cells[cell_id].add_replay_linear(stimulus, section=sec, segx=segx)
                elif isinstance(stimulus, circuit_stimulus_definitions.RelativeLinear):
                    if add_relativelinear_stimuli:
                        self.cells[cell_id].add_replay_relativelinear(stimulus, section=sec, segx=segx)
                elif isinstance(stimulus, circuit_stimulus_definitions.ShotNoise):
                    if add_shotnoise_stimuli:
                        self.cells[cell_id].add_replay_shotnoise(sec, segx, stimulus, shotnoise_stim_count=shotnoise_stim_count)
                elif isinstance(stimulus, circuit_stimulus_definitions.RelativeShotNoise):
                    if add_shotnoise_stimuli:
                        self.cells[cell_id].add_replay_relative_shotnoise(sec, segx, stimulus, shotnoise_stim_count=shotnoise_stim_count)
                elif isinstance(stimulus, circuit_stimulus_definitions.OrnsteinUhlenbeck):
                    if add_ornstein_uhlenbeck_stimuli:
                        self.cells[cell_id].add_ornstein_uhlenbeck(sec, segx, stimulus, stim_count=ornstein_uhlenbeck_stim_count)
                elif isinstance(stimulus, circuit_stimulus_definitions.RelativeOrnsteinUhlenbeck):
                    if add_ornstein_uhlenbeck_stimuli:
                        self.cells[cell_id].add_relative_ornstein_uhlenbeck(sec, segx, stimulus, stim_count=ornstein_uhlenbeck_stim_count)
                elif isinstance(stimulus, circuit_stimulus_definitions.Sinusoidal):
                    if add_sinusoidal_stimuli:
                        self.cells[cell_id].add_sinusoidal(stimulus)
                elif isinstance(stimulus, circuit_stimulus_definitions.SynapseReplay):  # sonata only
                    if self.circuit_access.target_contains_cell(
                        stimulus.target, cell_id
                    ):
                        self.cells[cell_id].add_synapse_replay(
                            stimulus, self.spike_threshold, self.spike_location
                        )
                else:
                    raise ValueError("Found stimulus with pattern %s, not supported" % stimulus)
                logger.debug(f"Added {stimulus} to cell_id {cell_id} at {sec_name}({segx})")

            if isinstance(stimulus, Noise):
                noisestim_count += 1
            elif isinstance(stimulus, (ShotNoise, RelativeShotNoise)):
                shotnoise_stim_count += 1
            elif isinstance(stimulus, (OrnsteinUhlenbeck, RelativeOrnsteinUhlenbeck)):
                ornstein_uhlenbeck_stim_count += 1

    def _add_synapses(
            self, pre_gids=None, add_minis=False):
        """Instantiate all the synapses."""
        for cell_id in self.cells:
            self._add_cell_synapses(
                cell_id, pre_gids=pre_gids,
                add_minis=add_minis)

    def _add_cell_synapses(
        self, cell_id: CellId, pre_gids=None, add_minis=False
    ) -> None:
        syn_descriptions = self.get_syn_descriptions(cell_id)

        if pre_gids is not None:
            if self.circuit_format == CircuitFormat.SONATA:
                syn_descriptions = self._intersect_pre_gids_cell_ids_multipopulation(
                    syn_descriptions, pre_gids
                )
            else:
                syn_descriptions = self._intersect_pre_gids(syn_descriptions, pre_gids)

        # Check if there are any presynaptic cells, otherwise skip adding
        # synapses
        if syn_descriptions.empty:
            logger.warning(
                f"No presynaptic cells found for gid {cell_id}, no synapses added"
            )

        else:
            for idx, syn_description in syn_descriptions.iterrows():
                popids = (
                    syn_description["source_popid"],
                    syn_description["target_popid"],
                )
                self._instantiate_synapse(
                    cell_id=cell_id,
                    syn_id=idx,  # type: ignore
                    syn_description=syn_description,
                    add_minis=add_minis,
                    popids=popids
                )
            logger.info(f"Added {syn_descriptions} synapses for gid {cell_id}")
            if add_minis:
                logger.info(f"Added minis for {cell_id=}")

    def _targets_from_compartment_set(
        self,
        stimulus: circuit_stimulus_definitions.Stimulus,
        compartment_sets: dict[str, dict[str, Any]] | None,
    ) -> list[tuple]:
        """Resolve a compartment_set stimulus into (cell_id, section, segx,
        sec_name) targets."""
        if compartment_sets is None:
            raise ValueError(
                "Simulation config provides compartment_set stimuli but "
                "no 'compartment_sets_file' is configured."
            )

        comp_name = stimulus.compartment_set
        if comp_name not in compartment_sets:
            raise ValueError(f"Compartment set '{comp_name}' not found in compartment_sets file.")

        comp_entry = compartment_sets[comp_name]
        comp_nodes = comp_entry.get("compartment_set", [])

        population_name = comp_entry.get("population")

        targets: list[tuple] = []
        for cell_id in self.cells:
            if population_name is not None and getattr(cell_id, "population_name", None) != population_name:
                continue
            try:
                resolved = self.cells[cell_id].resolve_segments_from_compartment_set(
                    cell_id.id if hasattr(cell_id, "id") else cell_id,
                    comp_nodes,
                )
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to resolve compartment_set for cell {cell_id}: {e}")
                continue

            for sec, sec_name, segx in resolved:
                targets.append((cell_id, sec, segx, sec_name))

        return targets

    @staticmethod
    def _intersect_pre_gids(syn_descriptions, pre_gids: list[CellId]) -> pd.DataFrame:
        """Return the synapse descriptions with pre_gids intersected."""
        _pre_gids = {x.id for x in pre_gids}
        return syn_descriptions[syn_descriptions[SynapseProperty.PRE_GID].isin(_pre_gids)]

    @staticmethod
    def _intersect_pre_gids_cell_ids_multipopulation(syn_descriptions, pre_cell_ids: list[CellId]) -> pd.DataFrame:
        """Return the synapse descriptions with pre_cell_ids intersected.

        Supports multipopulations.
        """
        filtered_rows = syn_descriptions.apply(
            lambda row: any(
                cell.population_name == row["source_population_name"] and row[SynapseProperty.PRE_GID] == cell.id
                for cell in pre_cell_ids
            ),
            axis=1,
        )
        return syn_descriptions[filtered_rows]

    def get_syn_descriptions(self, cell_id: int | tuple[str, int]) -> pd.DataFrame:
        """Get synapse descriptions dataframe."""
        cell_id = create_cell_id(cell_id)
        return self.circuit_access.extract_synapses(cell_id, projections=self.projections)

    @staticmethod
    def merge_pre_spike_trains(*train_dicts) -> dict[CellId, np.ndarray]:
        """Merge presynaptic spike train dicts."""
        filtered_dicts = [d for d in train_dicts if isinstance(d, dict) and d]

        if not filtered_dicts:
            return {}

        all_keys = set().union(*[d.keys() for d in filtered_dicts])
        result = {}

        for k in all_keys:
            valid_arrays = []
            for d in filtered_dicts:
                if k in d:
                    val = d[k]
                    if isinstance(val, (np.ndarray, list)) and len(val) > 0:
                        valid_arrays.append(np.asarray(val))
            if valid_arrays:
                result[k] = np.sort(np.concatenate(valid_arrays))

        return result

    def _find_matching_override(self, overrides, pre: CellId, post: CellId):
        matched = None
        for ov in overrides:
            # ov.source and ov.target are nodeset names
            if (
                self.circuit_access.target_contains_cell(ov.source, pre)
                and self.circuit_access.target_contains_cell(ov.target, post)
            ):
                matched = ov   # "last match wins" like Neurodamus ordering
        return matched

    def _add_connections(
            self,
            add_replay=None,
            interconnect_cells=None,
            user_pre_spike_trains: None | dict[CellId, Iterable] = None) -> None:
        """Instantiate the (replay and real) connections in the network."""
        pre_spike_trains = self.simulation_access.get_spikes() if add_replay else {}
        pre_spike_trains = self.merge_pre_spike_trains(
            pre_spike_trains,
            user_pre_spike_trains)

        connections_overrides = (
            self.circuit_access.config.connection_entries()
            if hasattr(self, "circuit_access")
            else []
        )

        for post_gid in self.cells:
            for syn_id in self.cells[post_gid].synapses:
                synapse = self.cells[post_gid].synapses[syn_id]
                syn_description: pd.Series = synapse.syn_description
                delay_weights = synapse.delay_weights
                source_population = syn_description["source_population_name"]
                pre_gid = CellId(source_population, int(syn_description[SynapseProperty.PRE_GID]))

                ov = self._find_matching_override(connections_overrides, pre_gid, post_gid)

                if ov is not None and ov.weight == 0.0:
                    logger.debug(
                        "Skipping connection due to zero weight override: %s -> %s | syn_id=%s",
                        pre_gid, post_gid, syn_id
                    )
                    continue

                if self.pc is None:
                    real_synapse_connection = bool(interconnect_cells) and (pre_gid in self.cells)
                else:
                    real_synapse_connection = bool(interconnect_cells)

                if real_synapse_connection:
                    if (
                            user_pre_spike_trains is not None
                            and pre_gid in user_pre_spike_trains
                    ):
                        raise BluecellulabError(
                            """Specifying prespike trains of real connections"""
                            """ is not allowed."""
                        )
                    if self.pc is None:  # serial only
                        connection = bluecellulab.Connection(
                            self.cells[post_gid].synapses[syn_id],
                            pre_spiketrain=None,
                            pre_cell=self.cells[pre_gid],
                            stim_dt=self.dt,
                            parallel_context=None,
                            spike_threshold=self.spike_threshold,
                            spike_location=self.spike_location,
                        )
                    else:  # MPI cross-rank
                        pre_g = self.global_gid(pre_gid.population_name, pre_gid.id)
                        connection = bluecellulab.Connection(
                            self.cells[post_gid].synapses[syn_id],
                            pre_spiketrain=None,
                            pre_gid=pre_g,
                            pre_cell=None,
                            stim_dt=self.dt,
                            parallel_context=self.pc,
                            spike_threshold=self.spike_threshold,
                            spike_location=self.spike_location,
                        )

                    logger.debug(f"Added real connection between {pre_gid} and {post_gid}, {syn_id}")
                else:  # replay connection
                    pre_spiketrain = pre_spike_trains.get(pre_gid, None)
                    connection = bluecellulab.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=pre_spiketrain,
                        pre_cell=None,
                        stim_dt=self.dt,
                        parallel_context=None,
                        spike_threshold=self.spike_threshold,
                        spike_location=self.spike_location
                    )

                    logger.debug(f"Added replay connection from {pre_gid} to {post_gid}, {syn_id}")

                if ov is not None:
                    logger.debug(
                        "Override matched: %s -> %s | syn_id=%s | weight=%s delay=%s",
                        pre_gid, post_gid, syn_id, ov.weight, ov.delay
                    )

                    syn_delay = getattr(ov, "synapse_delay_override", None)
                    if syn_delay is not None:
                        connection.set_netcon_delay(float(syn_delay))
                        logger.debug(
                            "Applied synapse_delay_override %.4g ms to %s -> %s | syn_id=%s",
                            syn_delay, pre_gid, post_gid, syn_id
                        )

                    if ov.delay is not None:
                        logger.warning(
                            "SONATA override 'delay' (delayed weight activation) is not supported yet; "
                            "applying weight immediately. %s -> %s | syn_id=%s | delay=%s",
                            pre_gid, post_gid, syn_id, ov.delay
                        )

                    if ov.weight is not None:
                        connection.set_weight_scalar(float(ov.weight))
                        logger.debug(
                            "Applied weight override factor %.4g to %s -> %s | syn_id=%s | final_weight=%.4g",
                            ov.weight, pre_gid, post_gid, syn_id, connection.post_netcon_weight
                        )

                self.cells[post_gid].connections[syn_id] = connection
                for delay, weight_scale in delay_weights:
                    self.cells[post_gid].add_replay_delayed_weight(
                        syn_id, delay,
                        weight_scale * connection.weight)

            if len(self.cells[post_gid].connections) > 0:
                logger.debug(f"Added synaptic connections for target {post_gid}")

    def _add_cells(self, cell_ids: list[CellId]) -> None:
        """Instantiate cells from a gid list."""
        self.cells = CellDict()

        for cell_id in cell_ids:
            self.cells[cell_id] = cell = self.create_cell_from_circuit(cell_id)
            if self.circuit_access.node_properties_available:
                cell.connect_to_circuit(SonataProxy(cell_id, self.circuit_access))

    def _instantiate_synapse(self, cell_id: CellId, syn_id: SynapseID, syn_description,
                             add_minis=False, popids=(0, 0)) -> None:
        """Instantiate one synapse for a given gid, syn_id and
        syn_description."""
        pre_cell_id = CellId(syn_description["source_population_name"], int(syn_description[SynapseProperty.PRE_GID]))
        syn_connection_parameters = get_synapse_connection_parameters(
            circuit_access=self.circuit_access,
            pre_cell=pre_cell_id,
            post_cell=cell_id)
        if syn_connection_parameters["add_synapse"]:
            condition_parameters = self.circuit_access.config.condition_parameters()

            self.cells[cell_id].add_replay_synapse(
                syn_id, syn_description, syn_connection_parameters, condition_parameters,
                popids=popids, extracellular_calcium=self.circuit_access.config.extracellular_calcium)
            if add_minis:
                mini_frequencies = self.circuit_access.fetch_mini_frequencies(cell_id)
                logger.debug(f"Adding minis for synapse {syn_id}: syn_description={syn_description}, connection={syn_connection_parameters}, frequency={mini_frequencies}")

                self.cells[cell_id].add_replay_minis(
                    syn_id,
                    syn_description,
                    syn_connection_parameters,
                    popids=popids,
                    mini_frequencies=mini_frequencies,
                )

    def run(
        self,
        t_stop: Optional[float] = None,
        v_init: Optional[float] = None,
        celsius: Optional[float] = None,
        dt: Optional[float] = None,
        forward_skip: bool = True,
        forward_skip_value: Optional[float] = None,
        cvode: bool = False,
        show_progress: bool = False,
    ):
        """Simulate the Circuit.

        Parameters
        ----------
        t_stop :
            This function will run the simulation until t_stop
        v_init :
            Voltage initial value when the simulation starts
        celsius :
            Temperature at which the simulation runs
        dt :
            Timestep (delta-t) for the simulation
        forward_skip :
                       [compatibility/non-sonata] Enable/disable ForwardSkip, when
                       forward_skip_value is None, forward skip will only be
                       enabled if the simulation config has a ForwardSkip value)
        forward_skip_value :
                       [compatibility/non-sonata] Overwrite the ForwardSkip value
                       in the simulation config. If this is set to None, the value
                       in the simulation config is used.
        cvode :
                Force the simulation to run in variable timestep. Not possible
                when there are stochastic channels in the neuron model. When
                enabled results from a large network simulation will not be
                exactly reproduced.
        show_progress:
                       Show a progress bar during simulations. When
                       enabled results from a large network simulation
                       will not be exactly reproduced.

        Note
        ----
        Passing ``dt`` to ``run()`` is deprecated and will be removed in a
        future release. The simulation timestep used is the one resolved at
        :class:`CircuitSimulation` construction (either the explicit
        ``dt`` passed to the constructor or the value from the
        ``simulation_config``).
        """
        if t_stop is None:
            t_stop = self.circuit_access.config.tstop
            if t_stop is None:  # type narrowing
                t_stop = 0.0
        if dt is not None and dt != self.dt:
            warnings.warn(
                "Passing `dt` to `run()` to change the simulation timestep is deprecated; "
                "pass `dt` to the CircuitSimulation constructor instead. "
                "This behavior will be removed in a future release.",
                DeprecationWarning,
            )
        dt = self.dt

        config_forward_skip_value = self.circuit_access.config.forward_skip  # legacy
        config_tstart = self.circuit_access.config.tstart or 0.0             # SONATA
        # Determine effective skip value and flag
        if forward_skip_value is not None:
            # User explicitly provided value → use it
            effective_skip_value = forward_skip_value
            effective_skip = forward_skip
        elif config_forward_skip_value is not None:
            # Use legacy config if available
            effective_skip_value = config_forward_skip_value
            effective_skip = forward_skip
        elif config_tstart > 0.0:
            # Use SONATA tstart *only* if no other skip value was provided
            effective_skip_value = config_tstart
            effective_skip = True
        else:
            # No skip
            effective_skip_value = None
            effective_skip = False

        if celsius is None:
            celsius = self.circuit_access.config.celsius
        NeuronGlobals.get_instance().temperature = celsius
        if v_init is None:
            v_init = self.circuit_access.config.v_init
        NeuronGlobals.get_instance().v_init = v_init

        sim = bluecellulab.Simulation(self.pc)
        for cell_id in self.cells:
            sim.add_cell(self.cells[cell_id])

        self.fih_prcellstate = None
        if self.pc is not None and self.print_cellstate:
            def dump():
                for cell in self.cells:
                    pop = cell.population_name
                    gid = cell.id
                    g = self.global_gid(pop, gid)
                    self.pc.prcellstate(g, f"bluecellulab_t={neuron.h.t}")

            def schedule_dump():
                t_dump = self.save_time if self.save_time is not None else self.circuit_access.config.tstop
                neuron.h.cvode.event(t_dump, dump)

            self.fih_prcellstate = neuron.h.FInitializeHandler(1, schedule_dump)

        if show_progress:
            logger.warning("show_progress enabled, this will very likely"
                           "break the exact reproducibility of large network"
                           "simulations")

        sim.run(
            tstop=t_stop,
            cvode=cvode,
            dt=dt,
            forward_skip=effective_skip,
            forward_skip_value=effective_skip_value,
            show_progress=show_progress)

    def get_mainsim_voltage_trace(
            self, cell_id: int | tuple[str, int], t_start=None, t_stop=None, t_step=None
    ) -> np.ndarray:
        """Get the voltage trace from a cell from the main simulation.

        Parameters
        -----------
        cell_id: cell id of interest.
        t_start, t_stop: time range of interest,
        report time range is used by default.
        t_step: time step (should be a multiple of report time step T;
        equals T by default)

        Returns:
            One dimentional np.ndarray to represent the voltages.
        """
        cell_id = create_cell_id(cell_id)
        return self.simulation_access.get_soma_voltage(cell_id, t_start, t_stop, t_step)

    def get_mainsim_time_trace(self, t_step=None) -> np.ndarray:
        """Get the time trace from the main simulation.

        Parameters
        -----------
        t_step: time step (should be a multiple of report time step T;
        equals T by default)

        Returns:
            One dimentional np.ndarray to represent the times.
        """
        return self.simulation_access.get_soma_time_trace(t_step)

    def get_time(self) -> np.ndarray:
        """Get the time vector for the recordings, contains negative times.

        The negative times occur as a result of ForwardSkip.
        """
        first_key = next(iter(self.cells))
        return self.cells[first_key].get_time()

    def get_time_trace(self, t_start=None, t_stop=None, t_step=None) -> np.ndarray:
        """Get the time vector for the recordings, negative times removed.

        Parameters
        -----------
        t_start, t_stop: time range of interest.
        t_step: time step (multiple of report dt; equals dt by default)

        Returns:
            1D np.ndarray representing time points.
        """
        time = self.get_time()
        time = time[time >= 0.0]

        if t_start is None or t_start < 0:
            t_start = 0
        if t_stop is None:
            t_stop = np.inf

        time = time[(time >= t_start) & (time <= t_stop)]

        if t_step is not None:
            ratio = t_step / self.dt
            time = _sample_array(time, ratio)

        return time

    def get_voltage_trace(
            self, cell_id: int | tuple[str, int], t_start=None, t_stop=None, t_step=None
    ) -> np.ndarray:
        """Get the voltage vector for the cell_id, negative times removed.

        Parameters
        -----------
        cell_id: cell id of interest.
        t_start, t_stop: time range of interest,
        report time range is used by default.
        t_step: time step (should be a multiple of report time step T;
        equals T by default)

        Returns:
            One dimentional np.ndarray to represent the voltages.
        """
        cell_id = create_cell_id(cell_id)
        time = self.get_time()
        voltage = self.cells[cell_id].get_soma_voltage()

        if t_start is None or t_start < 0:
            t_start = 0
        if t_stop is None:
            t_stop = np.inf

        voltage = voltage[np.where((time >= t_start) & (time <= t_stop))]

        if t_step is not None:
            ratio = t_step / self.dt
            voltage = _sample_array(voltage, ratio)
        return voltage

    def delete(self):
        """Delete CircuitSimulation and all of its attributes.

        NEURON objects are explicitly needed to be deleted.
        """
        if hasattr(self, 'cells'):
            for _, cell in self.cells.items():
                cell.delete()
            cell_ids = list(self.cells.keys())
            for cell_id in cell_ids:
                del self.cells[cell_id]

    def __del__(self):
        """Destructor.

        Deletes all allocated NEURON objects.
        """
        self.delete()

    def fetch_cell_kwargs(self, cell_id: CellId) -> dict:
        """Get the kwargs to instantiate a Cell object."""
        emodel_properties = self.circuit_access.get_emodel_properties(cell_id)
        cell_kwargs = {
            'template_path': self.circuit_access.emodel_path(cell_id),
            'morphology_path': self.circuit_access.morph_filepath(cell_id),
            'cell_id': cell_id,
            'record_dt': self.record_dt,
            'template_format': self.circuit_access.get_template_format(),
            'emodel_properties': emodel_properties,
        }

        return cell_kwargs

    def create_cell_from_circuit(self, cell_id: CellId) -> bluecellulab.Cell:
        """Create a Cell object from the circuit."""
        cell_kwargs = self.fetch_cell_kwargs(cell_id)
        return bluecellulab.Cell(template_path=cell_kwargs['template_path'],
                                 morphology_path=cell_kwargs['morphology_path'],
                                 cell_id=cell_kwargs['cell_id'],
                                 record_dt=cell_kwargs['record_dt'],
                                 template_format=cell_kwargs['template_format'],
                                 emodel_properties=cell_kwargs['emodel_properties'])

    def global_gid(self, pop: str, gid: int) -> int:
        """Return a globally unique NEURON GID for a (population, node_id)
        pair.

        NEURON's ParallelContext requires presynaptic sources to be identified by a
        single integer GID across all ranks. In SONATA circuits, node ids are only
        unique *within* a population, so we combine (population_name, node_id) into a
        single integer:

            global_gid = pop_index[population] * STRIDE + node_id

        Notes
        -----
        STRIDE must be larger than the maximum node_id in any population to avoid GID
        collisions.

        Parameters
        ----------
        pop : str
            SONATA population name (e.g. "S1nonbarrel_neurons", "POm", ...).
        gid : int
            Node id within that population.

        Returns
        -------
        int
            Globally unique integer GID used with ParallelContext.
        """
        if pop not in self._pop_index:
            raise KeyError(
                f"Population '{pop}' missing from pop index. "
                f"Known pops: {sorted(self._pop_index.keys())}"
            )
        return self._pop_index[pop] * self._gid_stride + int(gid)

    def _init_pop_index_mpi(self) -> None:
        """Build a consistent pop->index mapping across ranks."""
        assert self.pc is not None

        local_pops = set()
        for post_gid in self.cells:
            local_pops.add(post_gid.population_name)
            synapses = getattr(self.cells[post_gid], "synapses", None)
            if synapses:
                for syn in synapses.values():
                    sd = syn.syn_description
                    local_pops.add(sd["source_population_name"])

        gathered = self.pc.py_gather(sorted(local_pops), 0)

        if int(self.pc.id()) == 0:
            all_pops = set([""])
            for lst in gathered:
                all_pops.update(lst)
            pops_sorted = sorted(all_pops)
            pop_index = {p: i for i, p in enumerate(pops_sorted)}
        else:
            pop_index = None

        pop_index = self.pc.py_broadcast(pop_index, 0)
        self._pop_index = pop_index

    def _register_gids_for_mpi(self) -> None:
        assert self.pc is not None

        for cell_id, cell in self.cells.items():
            g = self.global_gid(cell_id.population_name, cell_id.id)

            self.pc.set_gid2node(g, int(self.pc.id()))
            nc = cell.create_netcon_spikedetector(
                None,
                location=self.spike_location,
                threshold=self.spike_threshold
            )
            self.pc.cell(g, nc)
