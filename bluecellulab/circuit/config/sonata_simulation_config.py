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
from __future__ import annotations
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Optional
import warnings

from bluecellulab.circuit.config.sections import Conditions, ConnectionOverrides
from bluecellulab.stimulus.circuit_stimulus_definitions import Stimulus

from bluepysnap import Simulation as SnapSimulation

logger = logging.getLogger(__name__)


class SonataSimulationConfig:
    """Sonata implementation of SimulationConfig protocol."""

    def __init__(self, config: str | Path | SnapSimulation) -> None:
        if isinstance(config, (str, Path)):
            if not Path(config).exists():
                raise FileNotFoundError(f"Config file {config} not found.")
            else:
                self.impl = SnapSimulation(config)
        elif isinstance(config, SnapSimulation):
            self.impl = config
        else:
            raise TypeError("Invalid config type.")

        self._connection_overrides: list[ConnectionOverrides] = []

    def get_all_projection_names(self) -> list[str]:
        return [
            edge_name
            for edge_name, edge_pop in self.impl.circuit.edges.items()
            if getattr(edge_pop.source, "type", None) == "virtual"
        ]

    def get_all_stimuli_entries(self) -> list[Stimulus]:
        result: list[Stimulus] = []
        inputs = self.impl.config.get("inputs")
        if inputs is None:
            return result
        config_dir = self._get_config_dir()

        compartment_sets = None
        try:
            compartment_sets = self.get_compartment_sets()
        except ValueError:
            pass

        for value in inputs.values():
            # Validate mutual exclusivity and existence of compartment_set
            if "compartment_set" in value and "node_set" in value:
                raise ValueError("Stimulus entry must not include both 'node_set' and 'compartment_set'.")

            if "compartment_set" in value:
                if compartment_sets is None:
                    raise ValueError("SONATA simulation config references 'compartment_set' in inputs but no 'compartment_sets_file' is configured.")
                comp_name = value["compartment_set"]
                if comp_name not in compartment_sets:
                    raise ValueError(f"Compartment set '{comp_name}' not found in compartment_sets file.")
                # Validate the list: must be list of triples, sorted and unique by (node_id, sec_ref, seg)
                comp_entry = compartment_sets[comp_name]
                comp_nodes = comp_entry.get("compartment_set")
                if comp_nodes is None:
                    raise ValueError(f"Compartment set '{comp_name}' does not contain 'compartment_set' key.")
                # Validate duplicates and sorted order
                try:
                    last = None
                    for trip in comp_nodes:
                        if not (isinstance(trip, list) and len(trip) >= 3):
                            raise ValueError(f"Invalid compartment_set entry '{trip}' in '{comp_name}'; expected [node_id, section, seg].")
                        key = (trip[0], trip[1], trip[2])
                        if last is not None and key < last:
                            raise ValueError(f"Compartment list for '{comp_name}' must be sorted ascending.")
                        if last == key:
                            raise ValueError(f"Compartment list for '{comp_name}' contains duplicate entry {key}.")
                        last = key
                except TypeError:
                    raise ValueError(f"Compartment list for '{comp_name}' contains non-comparable entries.")

            stimulus = Stimulus.from_sonata(value, config_dir=config_dir)
            if stimulus:
                result.append(stimulus)
        return result

    @lru_cache(maxsize=1)
    def condition_parameters(self) -> Conditions:
        """Returns parameters of global condition block of sonataconfig."""
        condition_object = self.impl.conditions
        return Conditions.from_sonata(condition_object)

    @lru_cache(maxsize=1)
    def _connection_entries(self) -> list[ConnectionOverrides]:
        result: list[ConnectionOverrides] = []
        if "connection_overrides" not in self.impl.config:
            return result
        conn_overrides: list = self.impl.config["connection_overrides"]
        if conn_overrides is None:
            return result
        for conn_entry in conn_overrides:
            result.append(ConnectionOverrides.from_sonata(conn_entry))
        return result

    @lru_cache(maxsize=1)
    def get_compartment_sets(self) -> dict[str, dict]:
        filepath = self.impl.config.get("compartment_sets_file")
        if not filepath:
            raise ValueError("No 'compartment_sets_file' entry found in SONATA config.")
        config_dir = self._get_config_dir()
        full_path = Path(filepath)
        if config_dir is not None and not full_path.is_absolute():
            full_path = Path(config_dir) / filepath
        with open(full_path, 'r') as f:
            return json.load(f)

    @lru_cache(maxsize=1)
    def get_node_sets(self) -> dict[str, dict]:
        circuit_filepath = self.impl.circuit.config.get("node_sets_file")
        base_node_sets = {}
        if circuit_filepath:
            with open(circuit_filepath, "r") as f:
                base_node_sets = json.load(f)

        sim_filepath = self.impl.config.get("node_sets_file")
        if sim_filepath:
            with open(sim_filepath, "r") as f:
                sim_node_sets = json.load(f)
            # Overwrite/add entries
            base_node_sets.update(sim_node_sets)

        if not base_node_sets:
            raise ValueError("No 'node_sets_file' found in simulation or circuit config.")

        return base_node_sets

    @lru_cache(maxsize=1)
    def get_report_entries(self) -> dict[str, dict]:
        """Returns the 'reports' dictionary from the SONATA simulation config.

        Each key is a report name, and the value is its configuration.
        """
        reports = self.impl.config.get("reports", {})
        if not isinstance(reports, dict):
            raise ValueError("Invalid format for 'reports' in SONATA config.")
        return reports

    def connection_entries(self) -> list[ConnectionOverrides]:
        return self._connection_entries() + self._connection_overrides

    def report_file_path(self, report_cfg: dict, report_key: str) -> Path:
        """Resolve the full path for the report output file."""
        output_dir = Path(self.output_root_path)
        file_name = report_cfg.get("file_name", f"{report_key}.h5")
        if not file_name.endswith(".h5"):
            file_name += ".h5"
        return output_dir / file_name

    @property
    def base_seed(self) -> int:
        return self.impl.run.random_seed

    @property
    def synapse_seed(self) -> int:
        return self.impl.run.synapse_seed

    @property
    def ionchannel_seed(self) -> int:
        return self.impl.run.ionchannel_seed

    @property
    def stimulus_seed(self) -> int:
        return self.impl.run.stimulus_seed

    @property
    def minis_seed(self) -> int:
        return self.impl.run.minis_seed

    @property
    def rng_mode(self) -> str:
        """Only Random123 is supported in SONATA."""
        return "Random123"

    @property
    def spike_threshold(self) -> float:
        return self.impl.run.spike_threshold

    @property
    def spike_location(self) -> str:
        return self.impl.conditions.spike_location.name

    @property
    def tstart(self) -> Optional[float]:
        return self.impl.config.get("run", {}).get("tstart", 0.0)

    @property
    def tstop(self) -> float:
        return self.impl.run.tstop

    @property
    def duration(self) -> Optional[float]:
        warnings.warn(
            "`duration` is deprecated. Use `tstop` instead.",
            DeprecationWarning
        )
        return self.tstop

    @property
    def dt(self) -> float:
        return self.impl.run.dt

    @property
    def forward_skip(self) -> Optional[float]:
        """forward_skip is removed from SONATA."""
        return None

    @property
    def celsius(self) -> float:
        value = self.condition_parameters().celsius
        return value if value is not None else 34.0

    @property
    def v_init(self) -> float:
        value = self.condition_parameters().v_init
        return value if value is not None else -65.0

    @property
    def output_root_path(self) -> str:
        return self.impl.config.get("output", {}).get("output_dir", "output")

    @property
    def spikes_file_path(self) -> Path:
        output_dir = Path(self.output_root_path)
        spikes_file = self.impl.config.get("output", {}).get("spikes_file", "out.h5")
        return output_dir / spikes_file

    @property
    def extracellular_calcium(self) -> Optional[float]:
        return self.condition_parameters().extracellular_calcium

    def add_connection_override(
        self,
        connection_override: ConnectionOverrides
    ) -> None:
        self._connection_overrides.append(connection_override)

    def _get_config_dir(self):
        # Prefer config_path, fallback to _simulation_config_path
        config_path = getattr(self.impl, "config_path", None)
        if config_path is None:
            sim_config_path = getattr(self.impl, "_simulation_config_path", None)
            if sim_config_path is not None:
                config_path = Path(sim_config_path)
        return str(config_path.parent) if config_path is not None else None
