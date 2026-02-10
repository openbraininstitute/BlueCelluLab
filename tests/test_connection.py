# Copyright 2025 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from bluecellulab.connection import Connection
from bluecellulab.circuit import SynapseProperty


class FakeNetcon:
    def __init__(self):
        self.weight = [None, None, None]
        self.delay = None
        self.threshold = None


class FakePC:
    def __init__(self):
        self.connected = []

    def gid_connect(self, gid, target):
        self.connected.append((gid, target))
        return FakeNetcon()


class FakeSynapse:
    def __init__(self, delay=1.0, weight=0.5, weight_scalar=None):
        self.syn_description = {
            SynapseProperty.AXONAL_DELAY: delay,
            SynapseProperty.G_SYNX: weight,
        }
        self.hsynapse = "syn_obj[0]"
        self.weight = weight_scalar


def test_mpi_connection_requires_pre_gid_when_parallel_context_present():
    pc = FakePC()
    syn = FakeSynapse()

    with pytest.raises(ValueError, match="pre_gid must be provided"):
        Connection(syn, pre_cell=object(), parallel_context=pc)


def test_mpi_connection_uses_gid_connect_and_sets_weights():
    pc = FakePC()
    syn = FakeSynapse(delay=2.5, weight=0.5, weight_scalar=3.0)

    conn = Connection(
        syn,
        pre_gid=7,
        pre_cell=None,
        parallel_context=pc,
        spike_threshold=-20.0,
    )

    assert pc.connected == [(7, syn.hsynapse)]
    assert conn.post_netcon.weight[0] == pytest.approx(1.5)  # 0.5 * weight_scalar
    assert conn.post_netcon.delay == pytest.approx(2.5)
    assert conn.post_netcon.threshold == pytest.approx(-20.0)


def test_setters_update_netcon_weight_and_delay():
    pc = FakePC()
    syn = FakeSynapse(delay=1.0, weight=0.5, weight_scalar=2.0)

    conn = Connection(
        syn,
        pre_gid=3,
        pre_cell=None,
        parallel_context=pc,
        spike_threshold=-20.0,
    )

    conn.set_netcon_weight(0.7)
    assert conn.post_netcon.weight[0] == pytest.approx(0.7)
    assert conn.post_netcon_weight == pytest.approx(0.7)

    conn.set_weight_scalar(4.0)
    assert conn.weight_scalar == pytest.approx(4.0)
    assert conn.post_netcon.weight[0] == pytest.approx(syn.syn_description[SynapseProperty.G_SYNX] * 4.0)
    assert conn.post_netcon_weight == pytest.approx(syn.syn_description[SynapseProperty.G_SYNX] * 4.0)

    conn.set_netcon_delay(3.3)
    assert conn.post_netcon.delay == pytest.approx(3.3)
    assert conn.post_netcon_delay == pytest.approx(3.3)


def test_negative_spiketrain_raises_with_numpy_array():
    syn = FakeSynapse()
    negative_train = np.array([-0.1, 0.2])
    with pytest.raises(ValueError, match="negative time"):
        Connection(syn, pre_spiketrain=negative_train, parallel_context=None, spike_threshold=-20.0)
