from __future__ import annotations

from typing import Iterable, List, Optional

from neuron import h

from bluecellulab.point.connection_params import PointProcessConnParameters
from bluecellulab.cell.point_process import BasePointProcessCell

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

        self.weight_factor = weight_factor
        self.syndelay_override = syndelay_override
        self.attach_src_cell = attach_src_cell
        self._replay = replay

        self._netcons: List[h.NetCon] = []

    @property
    def netcons(self) -> list[h.NetCon]:
        return self._netcons

    def finalize(self, cell: BasePointProcessCell) -> int:
        """Create NetCon(s) onto the given point neuron cell.

        Returns
        -------
        int
            Number of synapses (0 or 1).
        """
        n_syns = 0

        for params in self.synapse_params:
            n_syns += 1

            if self.attach_src_cell:
                # --- main path: presyn cell with sgid ---
                nc = pc.gid_connect(params.sgid, cell.pointcell)
                nc.delay = self.syndelay_override or float(params.delay)
                nc.weight[0] = float(params.weight) * self.weight_factor
                nc.threshold = DEFAULT_SPIKE_THRESHOLD
                self._netcons.append(nc)

            # --- replay path (optional, stubbed) ---
            if self._replay is not None and getattr(self._replay, "has_data", lambda: False)():
                vecstim = h.VecStim()
                vecstim.play(self._replay.time_vec)
                nc = h.NetCon(
                    vecstim,
                    cell.pointcell,
                    10.0,
                    self.syndelay_override or float(params.delay),
                    float(params.weight),
                )
                nc.weight[0] = float(params.weight) * self.weight_factor
                self._replay._store(vecstim, nc)

        return n_syns