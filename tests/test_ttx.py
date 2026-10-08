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
from unittest.mock import patch

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
    """`Cell.enable_ttx` must refuse to run when TTX cannot take effect.

    TTXDynamicsSwitch blocks the sodium channels by writing the ``ttx`` ion that
    they read. NEURON does not share an ion between separately compiled
    mechanism libraries, so a switch compiled apart from the sodium channels is
    present but inert: inserting it changes nothing, and without this check the
    simulation would finish reporting success with the channels never blocked.
    """

    def _cell(self):
        return bluecellulab.Cell(
            "%s/examples/cell_example_empty/test_cell.hoc" % script_dir,
            "%s/examples/cell_example_empty" % script_dir,
        )

    @pytest.mark.v5
    def test_raises_when_compiled_separately(self):
        cell = self._cell()
        with patch(
            "bluecellulab.cell.core.mechanisms_with_split_ion_coupling",
            return_value={"TTXDynamicsSwitch"},
        ):
            for method in (cell.enable_ttx, cell.disable_ttx):
                with pytest.raises(BluecellulabError, match="compiled separately"):
                    method()

    @pytest.mark.v5
    def test_raises_when_mechanism_absent(self):
        cell = self._cell()
        with patch(
            "bluecellulab.cell.core.mechanisms_with_split_ion_coupling",
            return_value=set(),
        ):
            with patch(
                "bluecellulab.cell.core.registered_mechanisms", return_value={"pas"}
            ):
                for method in (cell.enable_ttx, cell.disable_ttx):
                    with pytest.raises(BluecellulabError, match="not.*available in NEURON"):
                        method()

    @pytest.mark.v5
    def test_error_points_at_the_bundled_copy(self):
        """The message has to say where to get the file from."""
        cell = self._cell()
        with patch(
            "bluecellulab.cell.core.mechanisms_with_split_ion_coupling",
            return_value={"TTXDynamicsSwitch"},
        ):
            with pytest.raises(BluecellulabError) as excinfo:
                cell.enable_ttx()
        assert str(internal_mods_path()) in str(excinfo.value)
        assert "nrnivmodl" in str(excinfo.value)

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
