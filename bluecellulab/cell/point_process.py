from __future__ import annotations

import logging
from pathlib import Path
import queue
from typing import Optional

import bluecellulab
from bluecellulab.cell import Cell
from bluecellulab.circuit.simulation_access import get_synapse_replay_spikes
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.circuit import SynapseProperty
from bluecellulab.psection import PSection
from bluecellulab.type_aliases import HocObjectType

from neuron import h
import numpy as np

from bluecellulab.circuit.node_id import CellId
from bluecellulab.synapse.synapse_types import SynapseID
from bluecellulab.point.point_connection import PointProcessConnection
from bluecellulab.point.connection_params import PointProcessConnParameters

logger = logging.getLogger(__name__)


class BasePointProcessCell(Cell):
    """Base class for NEURON artificial point processes (IntFire1/2/...)."""

    def __init__(self, cell_id: Optional[CellId]) -> None:

        if cell_id is None:
            raise ValueError("PointProcessCell requires valid cell_id")
        self.cell_id = cell_id

        self._spike_times = h.Vector()
        self._spike_detector: Optional[h.NetCon] = None
        self.pointcell = None  # type: ignore[assignment]
        self.synapses = {}
        self.connections: dict[SynapseID, bluecellulab.Connection] = {}

        self._replay_vecs: list[h.Vector] = []
        self._replay_vecstims: list[h.VecStim] = []
        self._replay_netcons: list[h.NetCon] = []

        # TODO: some members used in base class Cell are init to None, empty; to refactor
        self.soma = None

        self.recordings = {}
        self.report_sites: dict[str, list[dict]] = {}

        self.post_gid = None
        self.ips = {}
        self.syn_mini_netcons = {}
        self.hocname = None
        self.record_dt = None
        self.delayed_weights = queue.PriorityQueue()
        self.psections: dict[int, PSection] = {}
        self.secname_to_psection: dict[str, PSection] = {}
        self.is_made_passive = False
        self.sonata_proxy = None
        self.persistent: list[HocObjectType] = []
        self.hypamp = 0.0
        self.threshold = 0.0

    @property
    def hoc_cell(self):
        return self.pointcell

    def init_callbacks(self):
        pass

    def connect_to_circuit(self, proxy) -> None:
        self._circuit_proxy = proxy

    def delete(self) -> None:
        # Stop recording
        if self._spike_detector is not None:
            # NetCon will be GC'd when no Python refs remain
            self._spike_detector = None
        if self._spike_times is not None:
            self._spike_times = None

        # Drop pointer to underlying NEURON object
        self.pointcell = None

    def get_spike_times(self) -> list[float]:
        return list(self._spike_times)

    def create_netcon_spikedetector(
        self,
        sec,              # ignored for artificial cells
        location=None,    # ignored for artificial cells
        threshold: float = 0.0,
    ) -> h.NetCon:
        if self.pointcell is None:
            raise ValueError("attempting to create netcon without valid pointprocess")
        nc = h.NetCon(self.pointcell.pointcell, None)
        nc.threshold = threshold  # harmless for artificial cells
        return nc

    def is_recording_spikes(self, location=None, threshold: float | None = None) -> bool:
        return self._spike_detector is not None

    def start_recording_spikes(self, sec, location=None, threshold: float = 0.0) -> None:
        if self._spike_detector is not None:
            return
        if self.pointcell is None:
            raise ValueError("attempting to record spikes without valid pointprocess")
        self._spike_times = h.Vector()
        self._spike_detector = h.NetCon(self.pointcell.pointcell, None)
        self._spike_detector.threshold = threshold
        self._spike_detector.record(self._spike_times)

    def get_recorded_spikes(self, location="pointcell", threshold=-20):
        return self._spike_times


class HocPointProcessCell(BasePointProcessCell):
    """Point process that wraps an arbitrary HOC/mod artificial mechanism."""

    def __init__(
        self,
        cell_id: Optional[CellId],
        mechanism_name: str,
        spike_threshold: float = 1.0,
    ) -> None:
        super().__init__(cell_id)

        try:
            mech_cls = getattr(h, mechanism_name)
        except AttributeError as exc:
            raise BluecellulabError(
                f"Point mechanism '{mechanism_name}' not found in NEURON. "
                "Make sure the mod/hoc files are compiled and loaded."
            ) from exc

        if cell_id is None:
            raise ValueError("call to create pointprocess mechanism without valid cell_id")
        point = mech_cls(cell_id.id)

        self.pointcell = point
        self.start_recording_spikes(None, None, threshold=spike_threshold)

    def add_synapse_replay(self, stimulus, spike_threshold: float, spike_location: str) -> None:
        """SONATA-style spike replay for point processes.

        This is a simplified analogue of Cell.add_synapse_replay, but
        instead of mapping spikes to individual synapses, we directly
        connect each presynaptic node_id's spike train to this
        artificial cell via VecStim → NetCon.
        """
        file_path = Path(stimulus.spike_file).expanduser()

        if not file_path.is_absolute():
            config_dir = stimulus.config_dir
            if config_dir is not None:
                file_path = Path(config_dir) / file_path

        file_path = file_path.resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Spike file not found: {str(file_path)}")

        synapse_spikes = get_synapse_replay_spikes(str(file_path))

        for synapse_id, synapse in self.synapses.items():
            pre_cell_id = CellId(
                str(synapse.syn_description["source_population_name"]),
                int(synapse.syn_description[SynapseProperty.PRE_GID]),
            )

            if pre_cell_id not in synapse_spikes:
                continue

            spikes_of_interest = synapse_spikes[pre_cell_id]
            delay = getattr(stimulus, "delay", 0.0) or 0.0
            duration = getattr(stimulus, "duration", np.inf)

            spikes_of_interest = spikes_of_interest[
                (spikes_of_interest >= delay) & (spikes_of_interest <= duration)
            ]

            if spikes_of_interest.size == 0:
                continue

            vec = h.Vector(spikes_of_interest)
            vs = h.VecStim()
            vs.play(vec)

            if self.pointcell is None:
                raise ValueError("attempting to add replay spikes without valid pointprocess")
            nc = h.NetCon(vs, self.pointcell.pointcell)
            # Use stimulus weight if available, otherwise default to 1.0
            weight = getattr(stimulus, "weight", 1.0)
            nc.weight[0] = weight
            nc.delay = 0.0  # delay already baked into spike times

            self._replay_vecs.append(vec)
            self._replay_vecstims.append(vs)
            self._replay_netcons.append(nc)

            logger.debug(
                f"Added replay connection from pre_node_id={pre_cell_id} "
                f"to point neuron {self.cell_id}"
            )

    def add_replay_synapse(self, syn_id, syn_description, syn_connection_parameters, condition_parameters,
                           popids, extracellular_calcium):
        """For Point Neurons, the replay simply queues events directly to the
        point obj."""

        # syn_connection_parameters should only have 1 element, PointProcessConnection will confirm
        point_params = PointProcessConnParameters(sgid=syn_description[SynapseProperty.PRE_GID], delay=syn_description[SynapseProperty.AXONAL_DELAY],
                                                  weight=syn_description[SynapseProperty.G_SYNX])

        pointConn = PointProcessConnection([point_params], syn_connection_parameters.get("Weight", 1.0))
        pointConn.syn_description = syn_description
        pointConn.hsynapse = self.pointcell.pointcell
        pointConn.syn_id = SynapseID(*syn_id)
        pointConn.post_cell_id = self.cell_id

        self.synapses[pointConn.syn_id] = pointConn


def mechanism_name_from_model_template(template_path: str, model_template: str) -> str:
    """Translate SONATA model_template into a NEURON mechanism name.

    Examples:
        'hoc:AllenPointCell' -> 'AllenPointCell'
        'nrn:IntFire1'       -> 'IntFire1'
        'AllenPointCell'     -> 'AllenPointCell'
    """
    mt = str(model_template).strip()
    if ":" in mt:
        prefix, name = mt.split(":", 1)
        prefix = prefix.lower()
        if prefix in ("hoc", "nrn"):
            h.load_file(template_path)
            return name
    return mt
