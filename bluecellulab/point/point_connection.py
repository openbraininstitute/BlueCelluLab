from __future__ import annotations

from typing import Any, Iterable, List, Optional

from neuron import h

from bluecellulab.point.connection_params import PointProcessConnParameters

pc = h.ParallelContext()


DEFAULT_SPIKE_THRESHOLD = 0.0


class PointProcessConnection:
    """Allen-style point connection: sgid -> PointNeuronCell.pointcell.

    Mirrors Neurodamus PointConnection:
      - at most one synapse per connection
      - uses pc.gid_connect(sgid, cell.pointcell)
      - can later be extended with replay (VecStim) if needed.
    """

    def __init__(
        self,
        synapse_params: Iterable[PointProcessConnParameters],
        weight_factor: float = 1.0,
        syndelay_override: Optional[float] = None,
        attach_src_cell: bool = True,
        replay=None,  # placeholder for future replay object
    ) -> None:
        self.synapse_params = list(synapse_params)
        assert len(self.synapse_params) <= 1, (
            "PointProcessConnection supports max. one synapse per connection"
        )

        self.weight = weight_factor
        self.syndelay_override = syndelay_override
        self.attach_src_cell = attach_src_cell
        self._replay = replay

        self.syn_description = None
        self.delay_weights: list[tuple[float, float]] = []
        self.hsynapse = None
        self.syn_id = None
        self.pre_gid = self.synapse_params[0].sgid
        self.post_cell_id = None

        self._netcons: List[h.NetCon] = []

    @property
    def netcons(self) -> list[h.NetCon]:
        return self._netcons

    @property
    def info_dict(self):
        synapse_dict: dict[str, Any] = {}
        synapse_dict['syn_description'] = self.syn_description.to_dict()
        # if keys are enum make them str
        synapse_dict['syn_description'] = {
            str(k): v for k, v in synapse_dict['syn_description'].items()}
        return synapse_dict
