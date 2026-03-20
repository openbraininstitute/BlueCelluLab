from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from bluecellulab.circuit.simulation_access import get_synapse_replay_spikes
from bluecellulab.exceptions import BluecellulabError
from neuron import h
import numpy as np

from bluecellulab.circuit.node_id import CellId

logger = logging.getLogger(__name__)

class BasePointProcessCell:
    """Base class for NEURON artificial point processes (IntFire1/2/...)."""

    def __init__(self, cell_id: Optional[CellId]) -> None:
        self.cell_id = cell_id

        self._spike_times = h.Vector()
        self._spike_detector: Optional[h.NetCon] = None
        self.pointcell = None  # type: ignore[assignment]
        self.synapses: dict = {}
        self.connections: dict = {}

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
        nc = h.NetCon(self.pointcell, None)
        nc.threshold = threshold  # harmless for artificial cells
        return nc

    def is_recording_spikes(self, location=None, threshold: float | None = None) -> bool:
        return self._spike_detector is not None

    def start_recording_spikes(self, sec, location=None, threshold: float = 0.0) -> None:
        if self._spike_detector is not None:
            return
        self._spike_times = h.Vector()
        self._spike_detector = h.NetCon(self.pointcell, None)
        self._spike_detector.threshold = threshold
        self._spike_detector.record(self._spike_times)


    def connect2target(self, target_pp=None) -> h.NetCon:
        """Neurodamus-like helper: NetCon from this cell to a target point process."""
        return h.NetCon(self.pointcell, target_pp)


class HocPointProcessCell(BasePointProcessCell):
    """Point process that wraps an arbitrary HOC/mod artificial mechanism.
    """

    def __init__(
        self,
        cell_id: Optional[CellId],
        mechanism_name: str,
        param_overrides: Optional[Mapping[str, Any]] = None,
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

        point = mech_cls()
        if param_overrides:
            for name, value in param_overrides.items():
                if hasattr(point, name):
                    setattr(point, name, value)

        self.pointcell = point
        self.start_recording_spikes(None, None, threshold=spike_threshold)

    def add_synapse_replay(self, stimulus, spike_threshold: float, spike_location: str) -> None:
        """SONATA-style spike replay for point processes.

        This is a simplified analogue of Cell.add_synapse_replay, but instead of
        mapping spikes to individual synapses, we directly connect each presynaptic
        node_id's spike train to this artificial cell via VecStim → NetCon.
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

        if not hasattr(self, "_replay_vecs"):
            self._replay_vecs: list[h.Vector] = []
        if not hasattr(self, "_replay_vecstims"):
            self._replay_vecstims: list[h.VecStim] = []
        if not hasattr(self, "_replay_netcons"):
            self._replay_netcons: list[h.NetCon] = []

        for pre_node_id, spikes in synapse_spikes.items():
            delay = getattr(stimulus, "delay", 0.0) or 0.0
            duration = getattr(stimulus, "duration", np.inf)

            spikes_of_interest = spikes[
                (spikes >= delay) & (spikes <= duration)
            ]
            if spikes_of_interest.size == 0:
                continue

            vec = h.Vector(spikes_of_interest)
            vs = h.VecStim()
            vs.play(vec)

            nc = h.NetCon(vs, self.pointcell)
            # Use stimulus weight if available, otherwise default to 1.0
            weight = getattr(stimulus, "weight", 1.0)
            nc.weight[0] = weight
            nc.delay = 0.0  # delay already baked into spike times

            self._replay_vecs.append(vec)
            self._replay_vecstims.append(vs)
            self._replay_netcons.append(nc)

            logger.debug(
                f"Added replay connection from pre_node_id={pre_node_id} "
                f"to point neuron {self.cell_id}"
            )

def mechanism_name_from_model_template(model_template: str) -> str:
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
            return name
    return mt

@dataclass
class IntFire1Params:
    tau: float = 10.0
    refrac: float = 2.0


class IntFire1Cell(BasePointProcessCell):
    def __init__(
        self,
        cell_id: Optional[CellId] = None,
        tau: float = 10.0,
        refrac: float = 2.0,
    ) -> None:
        super().__init__(cell_id)
        point = h.IntFire1()
        point.tau = tau
        point.refrac = refrac
        self.pointcell = point

        self.start_recording_spikes(None, None, threshold=1.0)


@dataclass
class IntFire2Params:
    taum: float = 10.0
    taus: float = 20.0
    ib: float = 0.0


class IntFire2Cell(BasePointProcessCell):
    def __init__(
        self,
        cell_id: Optional[CellId] = None,
        taum: float = 10.0,
        taus: float = 20.0,
        ib: float = 0.0,
    ) -> None:
        super().__init__(cell_id)
        point = h.IntFire2()
        point.taum = taum
        point.taus = taus
        point.ib = ib
        self.pointcell = point

        self.start_recording_spikes(None, None, threshold=1.0)


def create_intfire1_cell(
    tau: float = 10.0,
    refrac: float = 2.0,
    cell_id: Optional[CellId] = None,
) -> IntFire1Cell:
    return IntFire1Cell(cell_id=cell_id, tau=tau, refrac=refrac)


def create_intfire2_cell(
    taum: float = 10.0,
    taus: float = 20.0,
    ib: float = 0.0,
    cell_id: Optional[CellId] = None,
) -> IntFire2Cell:
    return IntFire2Cell(cell_id=cell_id, taum=taum, taus=taus, ib=ib)
