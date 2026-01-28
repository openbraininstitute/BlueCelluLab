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
"""Class that represents a connection between two cells in bluecellulab."""

from typing import Optional

import neuron
import numpy as np

from bluecellulab.cell.core import Cell
from bluecellulab.circuit import SynapseProperty


class Connection:
    """Class that represents a connection between two cells in bluecellulab."""

    def __init__(
            self,
            post_synapse,
            pre_spiketrain: Optional[np.ndarray] = None,
            pre_gid: Optional[int] = None,
            pre_cell: Optional[Cell] = None,
            stim_dt=None,
            parallel_context=None,
            spike_threshold: float = -30.0,
            spike_location="soma"):
        self.persistent = []
        self.delay = post_synapse.syn_description[SynapseProperty.AXONAL_DELAY]
        self.weight = post_synapse.syn_description[SynapseProperty.G_SYNX]
        self.pre_cell = pre_cell
        self.pre_spiketrain = pre_spiketrain
        self.post_synapse = post_synapse
        self.pc = parallel_context

        if post_synapse.weight is not None:
            self.weight_scalar = post_synapse.weight
        else:
            self.weight_scalar = 1.0

        self.post_netcon = None
        self.post_netcon_delay = self.delay
        self.post_netcon_weight = self.weight * self.weight_scalar

        if self.pre_spiketrain is not None:
            if np.any(self.pre_spiketrain < 0):
                raise ValueError("bluecellulab Connection: a spiketrain contains "
                                 "a negative time, this is not supported by "
                                 "NEURON's Vecstim: %s" %
                                 str(self.pre_spiketrain))
            t_vec = neuron.h.Vector(self.pre_spiketrain)
            vecstim = neuron.h.VecStim()
            vecstim.play(t_vec, stim_dt)
            self.post_netcon = neuron.h.NetCon(
                vecstim, self.post_synapse.hsynapse,
                spike_threshold,
                self.post_netcon_delay,
                self.post_netcon_weight)
            # set netcon type
            nc_param_name = 'nc_type_param_{}'.format(
                self.post_synapse.hsynapse).split('[')[0]
            if hasattr(neuron.h, nc_param_name):
                nc_type_param = int(getattr(neuron.h, nc_param_name))
                self.post_netcon.weight[nc_type_param] = 2  # NC_REPLAY
            self.persistent.append(t_vec)
            self.persistent.append(vecstim)
        elif self.pre_cell is not None or (self.pc is not None and pre_gid is not None):
            if self.pre_cell is not None and self.pc is None:
                # serial/local
                self.post_netcon = self.pre_cell.create_netcon_spikedetector(
                    self.post_synapse.hsynapse, location=spike_location, threshold=spike_threshold
                )
            else:
                # MPI gid-based (works across ranks)
                if pre_gid is None:
                    raise ValueError("pre_gid must be provided when using ParallelContext")
                self.post_netcon = self.pc.gid_connect(int(pre_gid), self.post_synapse.hsynapse)

            self.set_netcon_weight(self.post_netcon_weight)
            self.set_netcon_delay(self.post_netcon_delay)
            self.post_netcon.threshold = spike_threshold
            # set netcon type
            nc_param_name = 'nc_type_param_{}'.format(
                self.post_synapse.hsynapse).split('[')[0]
            if hasattr(neuron.h, nc_param_name):
                nc_type_param = int(getattr(neuron.h, nc_param_name))
                self.post_netcon.weight[nc_type_param] = 0  # NC_PRESYN

    def set_netcon_weight(self, w: float) -> None:
        self.post_netcon_weight = float(w)
        if self.post_netcon is not None:
            self.post_netcon.weight[0] = self.post_netcon_weight

    def set_weight_scalar(self, scalar: float) -> None:
        self.weight_scalar = float(scalar)
        self.set_netcon_weight(self.weight * self.weight_scalar)

    def set_netcon_delay(self, d: float) -> None:
        self.post_netcon_delay = float(d)
        if self.post_netcon is not None:
            self.post_netcon.delay = self.post_netcon_delay


    @property
    def info_dict(self):
        """Return dict that contains information that can restore this conn."""

        connection_dict = {}

        connection_dict['pre_cell_id'] = self.post_synapse.pre_gid
        connection_dict['post_cell_id'] = self.post_synapse.post_cell_id.id
        connection_dict['post_synapse_id'] = self.post_synapse.syn_id.sid

        connection_dict['post_netcon'] = {}
        connection_dict['post_netcon']['weight'] = self.post_netcon_weight
        connection_dict['post_netcon']['delay'] = self.post_netcon_delay

        return connection_dict

    def delete(self):
        """Delete the connection."""
        for persistent_object in self.persistent:
            del persistent_object

    def __del__(self):
        self.delete()
