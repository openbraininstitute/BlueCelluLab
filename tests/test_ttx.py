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
"""Unit tests for TTX in mod files"""

import os
import types
from unittest.mock import MagicMock, patch

import pytest

import bluecellulab
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.mod_compilation import internal_mods_path

script_dir = os.path.dirname(__file__)


@pytest.mark.v5
def test_allNaChannels():
    """TTX: Testing ttx enabling"""

    na_channelnames = ['NaTs2_t']

    cell = bluecellulab.Cell(
        "%s/examples/cell_example_empty/test_cell.hoc" %
        script_dir,
        "%s/examples/cell_example_empty" %
        script_dir)

    for na_channelname in na_channelnames:
        cell.soma.insert(na_channelname)

        cell.add_step(0, 1000, .1)
        sim = bluecellulab.Simulation()
        sim.add_cell(cell)

        sim.run(10)
        voltage_nottx1 = cell.get_soma_voltage()

        cell.enable_ttx()
        sim.run(10)
        voltage_ttx = cell.get_soma_voltage()

        cell.disable_ttx()
        sim.run(10)
        voltage_nottx2 = cell.get_soma_voltage()

        # Check if voltage changed due to enable_ttx
        assert voltage_nottx1[-1] != voltage_ttx[-1]

        assert voltage_nottx1[-1] == voltage_nottx2[-1]


@pytest.mark.v6
def test_allNaChannels_v6a():
    """TTX: Testing ttx enabling in v6a cell"""

    cell = bluecellulab.Cell(
        "%s/examples/cell_example_empty/test_cell_v6a.hoc" %
        script_dir,
        "%s/examples/cell_example_empty" %
        script_dir)

    cell.add_step(0, 1000, .1)
    sim = bluecellulab.Simulation()
    sim.add_cell(cell)

    sim.run(10)
    # time_nottx1 = cell.get_time()
    voltage_nottx1 = cell.get_soma_voltage()

    cell.enable_ttx()
    sim.run(10)
    # time_ttx = cell.get_time()
    voltage_ttx = cell.get_soma_voltage()

    cell.disable_ttx()
    sim.run(10)
    voltage_nottx2 = cell.get_soma_voltage()

    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(time_nottx1, voltage_nottx1, label='no ttx')
    plt.plot(time_ttx, voltage_ttx, label='ttx')
    plt.legend()
    plt.savefig('ttx.png')
    '''

    # Check if voltage changed due to enable_ttx

    assert voltage_nottx1[-1] != voltage_ttx[-1]
    assert voltage_nottx1[-1] == voltage_nottx2[-1]


class TestTtxAvailabilityIsChecked:
    """`Cell.enable_ttx` must refuse to run when TTXDynamicsSwitch is missing."""

    def _cell(self):
        return bluecellulab.Cell(
            "%s/examples/cell_example_empty/test_cell.hoc" % script_dir,
            "%s/examples/cell_example_empty" % script_dir,
        )

    @pytest.mark.v5
    def test_raises_when_mechanism_absent(self):
        cell = self._cell()
        with patch("bluecellulab.cell.core.registered_mechanisms", return_value={"pas"}):
            for method in (cell.enable_ttx, cell.disable_ttx):
                with pytest.raises(BluecellulabError, match="not.*available in NEURON"):
                    method()

    @pytest.mark.v5
    def test_error_points_at_the_bundled_copy(self):
        """The message has to say where to get the file from."""
        cell = self._cell()
        with patch("bluecellulab.cell.core.registered_mechanisms", return_value={"pas"}):
            with pytest.raises(BluecellulabError) as excinfo:
                cell.enable_ttx()
        assert str(internal_mods_path()) in str(excinfo.value)

    @pytest.mark.v5
    def test_does_not_interfere_when_ttx_is_usable(self):
        """The normal case must be untouched: no raise, and ttx still works."""
        cell = self._cell()
        cell.soma.insert("NaTs2_t")
        cell.add_step(0, 1000, 0.1)
        sim = bluecellulab.Simulation()
        sim.add_cell(cell)

        sim.run(10)
        before = cell.get_soma_voltage()[-1]
        cell.enable_ttx()
        sim.run(10)
        after = cell.get_soma_voltage()[-1]

        assert before != after


def test_default_ttx_checks_the_right_section():
    """`ismembrane` must get `sec=section`, or it tests the currently accessed
    section for every loop iteration."""
    sections = {"soma": MagicMock(), "dend": MagicMock()}

    cell = types.SimpleNamespace(sections=sections)
    with patch(
        "bluecellulab.cell.core.neuron"
    ) as mock_neuron:
        mock_neuron.h.ismembrane.return_value = False
        bluecellulab.Cell._default_enable_ttx(cell)

    calls = mock_neuron.h.ismembrane.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["sec"] is sections["soma"]
    assert calls[1].kwargs["sec"] is sections["dend"]
