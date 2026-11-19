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
"""Unit tests for tools.py"""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import bluecellulab
from bluecellulab.cell import Cell
from bluecellulab import CircuitSimulation
from bluecellulab.cell.ballstick import create_ball_stick
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.node_id import CellId, create_cell_id
from bluecellulab.exceptions import UnsteadyCellError
from bluecellulab.simulation.neuron_globals import set_neuron_globals
from bluecellulab.simulation.parallel import IsolatedProcess
from bluecellulab.tools import (
    _probe_rheobase_and_rin,
    calculate_SS_voltage,
    calculate_SS_voltage_subprocess,
    calculate_input_resistance,
    detect_hyp_current,
    detect_spike,
    detect_spike_step,
    detect_spike_step_subprocess,
    holding_current,
    holding_current_subprocess,
    search_threshold_current,
    template_accepts_cvode,
    check_empty_topology,
    calculate_max_thresh_current,
    calculate_rheobase,
    validate_section_and_segment,
    compute_memodel_properties,
    compute_memodel_properties_from_template,
    compute_memodel_properties_from_template_subprocess,
    compute_memodel_properties_subprocess,
    compute_memodel_properties_v2,
)
from bluecellulab.cell.section_tools import currents_vars, mechs_vars


script_dir = Path(__file__).parent


@pytest.fixture
def mock_cell():
    emodel_properties = EmodelProperties(
        threshold_current=1.1433533430099487,
        holding_current=1.4146618843078613,
        AIS_scaler=1.4561502933502197,
        soma_scaler=1.0
    )
    cell = Cell(
        f"{script_dir}/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
        f"{script_dir}/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
        template_format="v6",
        emodel_properties=emodel_properties
    )
    return cell


@pytest.fixture
def mock_cell_sections():
    cell = MagicMock(spec=Cell)
    cell.sections = {
        'soma': MagicMock(),
        'axon[0]': MagicMock(),
        'axon[1]': MagicMock(),
    }
    return cell


@pytest.fixture
def mock_calculate_SS_voltage():
    return MagicMock(return_value=-65.0)


@pytest.fixture
def mock_calculate_input_resistance():
    return MagicMock(return_value=100.0)


@pytest.fixture
def mock_search_threshold_current():
    return MagicMock(return_value=0.1)


@pytest.mark.v5
def test_calculate_SS_voltage_subprocess():
    """Tools: Test calculate_SS_voltage"""
    SS_voltage = calculate_SS_voltage_subprocess(
        template_path=script_dir / "examples/cell_example1/test_cell.hoc",
        morphology_path=script_dir / "examples/cell_example1",
        template_format="v5",
        emodel_properties=None,
        step_level=0,
        spike_threshold=-20,
        check_for_spiking=False,
    )
    assert abs(SS_voltage - -73.9235504304) < 0.001

    SS_voltage_stoch = calculate_SS_voltage_subprocess(
        template_path=script_dir / "examples/cell_example2/test_cell.hoc",
        morphology_path=script_dir / "examples/cell_example2",
        template_format="v5",
        emodel_properties=None,
        step_level=0,
        check_for_spiking=False,
        spike_threshold=-20,
    )
    assert abs(SS_voltage_stoch - -73.9235504304) < 0.001


def test_detect_spike():
    """Unit test for detect_spike."""

    # Case where there is a spike
    voltage_with_spike = np.array([-80, -70, -50, -10, -60, -80])
    assert detect_spike(voltage_with_spike) is True

    # Case where there is no spike
    voltage_without_spike = np.array([-80, -70, -60, -50, -60, -80])
    assert detect_spike(voltage_without_spike) is False

    # Edge case where the voltage reaches exactly -20 mV but does not surpass it
    voltage_at_edge = np.array([-80, -70, -60, -20, -60, -80])
    assert detect_spike(voltage_at_edge) is False

    # Test with an empty array
    voltage_empty = np.array([])
    assert detect_spike(voltage_empty) is False


@pytest.mark.v6
class TestOnSonataCell:
    def setup_method(self):
        self.template_name = (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc"
        )
        self.morphology_path = (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc"
        )
        self.template_format = "v6"
        self.emodel_properties = EmodelProperties(
            threshold_current=0.03203125, holding_current=-0.11, AIS_scaler=1.11
        )

    def test_detect_hyp_current(self):
        """Unit test detect_hyp_current."""
        target_voltage = -85
        hyp_current = detect_hyp_current(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            target_voltage=target_voltage,
        )
        assert hyp_current == pytest.approx(-0.0009765625)

    def test_calculate_input_resistance(self):
        """Unit test calculate_input_resistance."""
        input_resistance = calculate_input_resistance(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
        )

        assert input_resistance == pytest.approx(334.11, abs=0.07)

    def test_calculate_SS_voltage(self):
        """Unit test calculate_SS_voltage."""
        step_level = 0
        SS_voltage = calculate_SS_voltage(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            step_level=step_level,
        )
        assert SS_voltage == pytest.approx(-85.13, abs=0.02)

    def test_calculate_SS_voltage_subprocess_exception(self):
        """Unit test calculate_SS_voltage_subprocess."""
        step_level = 2
        with pytest.raises(UnsteadyCellError):
            _ = calculate_SS_voltage_subprocess(
                template_path=self.template_name,
                morphology_path=self.morphology_path,
                template_format=self.template_format,
                emodel_properties=self.emodel_properties,
                step_level=step_level,
                check_for_spiking=True,
                spike_threshold=-20,
            )

    def test_template_accepts_cvode(self):
        """Unit test for template_accepts_cvode."""
        assert template_accepts_cvode(self.template_name) is True

        mock_template_with_stochkv = script_dir / "examples/cell_example2/test_cell.hoc"
        assert template_accepts_cvode(mock_template_with_stochkv) is False

    def test_detect_spike_step(self):
        """Unit test for detect_spike_step."""
        hyp_level = -2
        inj_start = 100  # some start time for the current injection
        inj_stop = 200  # some stop time for the current injection
        step_level = 4  # some current level for the step

        spike_occurred = detect_spike_step(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            hyp_level=hyp_level,
            inj_start=inj_start,
            inj_stop=inj_stop,
            step_level=step_level,
        )
        assert spike_occurred is True

    def test_search_threshold_current(self):
        """Unit test for search_threshold_current."""
        hyp_level = -2
        inj_start, inj_stop = 100, 200
        min_current, max_current = 0, 10

        threshold_current = search_threshold_current(
            template_name=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            hyp_level=hyp_level,
            inj_start=inj_start,
            inj_stop=inj_stop,
            min_current=min_current,
            max_current=max_current,
        )
        assert threshold_current == pytest.approx(2.00195, abs=1e-5)

    def test_detect_spike_step_subprocess(self):
        """Unit test for detect_spike_step_subprocess."""
        hyp_level = -2
        inj_start = 100
        inj_stop = 200
        step_level = 4

        spike_occurred = detect_spike_step_subprocess(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
            hyp_level=hyp_level,
            inj_start=inj_start,
            inj_stop=inj_stop,
            step_level=step_level,
        )
        assert spike_occurred is True

    def test_compute_memodel_properties(self):
        """Unit test compute_memodel_properties.

        Expected values from origin/main on this cADpyr_L2TPC example cell.
        """
        cell = Cell(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
        )
        resp_dict = compute_memodel_properties(cell)

        assert resp_dict["holding_current"] == pytest.approx(-0.11)
        assert resp_dict["rheobase"] == pytest.approx(0.13921147797816324)
        assert resp_dict["rin"] == pytest.approx(334.1378411934045)

    def test_compute_memodel_properties_subprocess(self):
        """Cover compute_memodel_properties_subprocess in-process (IsolatedProcess is not measured)."""
        cell_kwargs = {
            "template_path": self.template_name,
            "morphology_path": self.morphology_path,
            "template_format": self.template_format,
            "emodel_properties": self.emodel_properties,
        }
        rheobase, rin = compute_memodel_properties_subprocess(
            cell_kwargs, holding_current=-0.11, spike_threshold_voltage=-30
        )
        assert rheobase == pytest.approx(0.1392, abs=1e-4)
        assert rin == pytest.approx(334.14, abs=0.1)

    def test_compute_memodel_properties_from_template(self):
        """from_template alone in worker (no parent Cell / gid 0).

        Expected to differ from ``test_compute_memodel_properties`` on this cvode
        cell: the cell API forks after building a parent Cell, so the worker still
        has that unused ghost cell and cvode's step sizes change. from_template has
        no ghost cell and is the more accurate path.
        """
        resp_dict = compute_memodel_properties_from_template(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            emodel_properties=self.emodel_properties,
        )

        assert resp_dict["holding_current"] == pytest.approx(-0.11)
        assert resp_dict["rheobase"] == pytest.approx(0.13918982460425472)
        assert resp_dict["rin"] == pytest.approx(334.1862365049614)
        # Ghost-cell path golden from test_compute_memodel_properties (same template).
        assert resp_dict["rheobase"] != pytest.approx(0.13921147797816324)
        assert resp_dict["rin"] != pytest.approx(334.1378411934045)

    def test_compute_memodel_properties_v2(self):
        """Unit test compute_memodel_properties_v2."""
        resp_dict = compute_memodel_properties_v2(
            template_path=self.template_name,
            morphology_path=self.morphology_path,
            template_format=self.template_format,
            holding_voltage=-85.0,
            emodel_properties=self.emodel_properties,
        )

        assert resp_dict["holding_current"] == pytest.approx(0.0003600167531203624)
        assert resp_dict["resting_potential"] == pytest.approx(-85.13006257460853)
        assert resp_dict["input_resistance"] == pytest.approx(334.1862365049614)
        assert resp_dict["threshold_current"] == pytest.approx(0.03380088341395323)


@pytest.mark.v6
class TestOnSonataCircuit:
    def setup_method(self):
        self.circuit_path = (
            script_dir
            / "examples/sim_quick_scx_sonata_multicircuit/simulation_config_noinput.json"
        )
        self.cell_id = ("NodeA", 0)

    def test_holding_current(self):
        """Unit test for holding_current on a SONATA circuit."""
        v_hold = -70  # arbitrary holding voltage for the test
        i_hold, v_control = holding_current(
            v_hold, cell_id=self.cell_id, circuit_path=self.circuit_path
        )

        assert i_hold == pytest.approx(0.0020779234)
        assert v_control == pytest.approx(v_hold)

    def test_holding_current_subprocess(self):
        """Unit test for holding_current_subprocess on a SONATA circuit."""
        v_hold = -80
        circuit_sim = CircuitSimulation(self.circuit_path)
        cell_id = create_cell_id(self.cell_id)
        cell_kwargs = circuit_sim.fetch_cell_kwargs(cell_id)
        i_hold, v_control = holding_current_subprocess(
            v_hold, enable_ttx=True, cell_kwargs=cell_kwargs
        )
        assert i_hold == pytest.approx(-0.03160848349, rel=1e-5)
        assert v_control == pytest.approx(v_hold)


def test_check_empty_topology():
    """Unit test for the check_empty_topology function."""
    assert check_empty_topology() is True
    cell = create_ball_stick()
    assert check_empty_topology() is False


def test_compute_memodel_properties_requires_emodel_properties():
    """compute_memodel_properties rejects cells without emodel_properties."""
    cell = MagicMock()
    cell.hypamp = None
    cell.template_params.emodel_properties = None
    with pytest.raises(ValueError, match="emodel_properties cannot be None"):
        compute_memodel_properties(cell)


def test_compute_memodel_properties_from_template_requires_emodel_properties():
    """compute_memodel_properties_from_template rejects missing emodel_properties."""
    with pytest.raises(ValueError, match="emodel_properties cannot be None"):
        compute_memodel_properties_from_template(
            template_path="template.hoc",
            morphology_path="morph.asc",
            template_format="v5",
            emodel_properties=None,
        )


def _memodel_cell_mock(hypamp, holding_current, template_path="template.hoc"):
    cell = MagicMock()
    cell.hypamp = hypamp
    cell.template_params.template_filepath = template_path
    cell.template_params.morph_filepath = "morph.asc"
    cell.template_params.template_format = "v5"
    cell.template_params.emodel_properties = EmodelProperties(
        threshold_current=0.0, holding_current=holding_current
    )
    return cell


def test_compute_memodel_properties_uses_emodel_holding_for_search():
    """Search holding comes from emodel_properties; returned holding from hypamp."""
    cell = _memodel_cell_mock(hypamp=None, holding_current=-0.05)
    runner = MagicMock()
    runner.apply.return_value = (0.1, 200.0)
    runner.__enter__.return_value = runner
    runner.__exit__.return_value = False

    with patch("bluecellulab.tools.IsolatedProcess", return_value=runner), patch(
        "bluecellulab.tools.set_neuron_globals"
    ):
        result = compute_memodel_properties(cell)

    assert runner.apply.call_args.args[0] is compute_memodel_properties_subprocess
    assert runner.apply.call_args.args[1][1] == -0.05
    assert result == {"holding_current": 0.0, "rheobase": 0.1, "rin": 200.0}


def test_compute_memodel_properties_always_uses_single_cell_path():
    """StochKv and deterministic templates share the single-Cell worker."""
    cell = _memodel_cell_mock(hypamp=-0.11, holding_current=-0.11)
    runner = MagicMock()
    runner.apply.return_value = (0.14, 334.0)
    runner.__enter__.return_value = runner
    runner.__exit__.return_value = False

    with patch("bluecellulab.tools.IsolatedProcess", return_value=runner), patch(
        "bluecellulab.tools.calculate_rheobase"
    ) as rheobase, patch(
        "bluecellulab.tools.calculate_input_resistance"
    ) as rin, patch("bluecellulab.tools.set_neuron_globals"):
        result = compute_memodel_properties(cell)

    rheobase.assert_not_called()
    rin.assert_not_called()
    assert runner.apply.call_args.args[0] is compute_memodel_properties_subprocess
    assert result == {"holding_current": -0.11, "rheobase": 0.14, "rin": 334.0}


def test_compute_memodel_properties_from_template_uses_emodel_holding_for_search():
    """Path-based API: search holding from emodel; returned holding from worker Cell."""
    emodel_properties = EmodelProperties(
        threshold_current=0.0, holding_current=-0.05
    )
    runner = MagicMock()
    runner.apply.return_value = (0.0, 0.1, 200.0)
    runner.__enter__.return_value = runner
    runner.__exit__.return_value = False

    with patch("bluecellulab.tools.IsolatedProcess", return_value=runner), patch(
        "bluecellulab.tools.set_neuron_globals"
    ):
        result = compute_memodel_properties_from_template(
            template_path="template.hoc",
            morphology_path="morph.asc",
            template_format="v5",
            emodel_properties=emodel_properties,
        )

    assert (
        runner.apply.call_args.args[0]
        is compute_memodel_properties_from_template_subprocess
    )
    assert runner.apply.call_args.args[1][1] == -0.05
    assert result == {"holding_current": 0.0, "rheobase": 0.1, "rin": 200.0}


@pytest.mark.v5
def test_compute_memodel_properties_v5_cstut():
    """Golden values from origin/main on cSTUT_7 with emodel holding -0.05 nA."""
    emodel_properties = EmodelProperties(
        threshold_current=0.0, holding_current=-0.05
    )
    cell = Cell(
        template_path=script_dir / "examples/cell_example_cstut/cSTUT_7.hoc",
        morphology_path=script_dir / "examples/cell_example_cstut",
        template_format="v5",
        emodel_properties=emodel_properties,
    )
    resp_dict = compute_memodel_properties(cell)

    assert resp_dict["holding_current"] == pytest.approx(0.0)
    assert resp_dict["rheobase"] == pytest.approx(0.0968098129791797)
    assert resp_dict["rin"] == pytest.approx(241.439121257406)


@pytest.mark.v5
def test_compute_memodel_properties_from_template_v5_cstut():
    """Path-based API on cSTUT_7; identical to cell-based (fixed step, unseeded StochKv)."""
    emodel_properties = EmodelProperties(
        threshold_current=0.0, holding_current=-0.05
    )
    resp_dict = compute_memodel_properties_from_template(
        template_path=script_dir / "examples/cell_example_cstut/cSTUT_7.hoc",
        morphology_path=script_dir / "examples/cell_example_cstut",
        template_format="v5",
        emodel_properties=emodel_properties,
    )

    assert resp_dict["holding_current"] == pytest.approx(0.0)
    assert resp_dict["rheobase"] == pytest.approx(0.0968098129791797)
    assert resp_dict["rin"] == pytest.approx(241.439121257406)


def test_compute_memodel_properties_from_template_pins_cell_id_zero():
    """Worker constructs Cell with fixed CellId(\"\", 0), not Cell.last_id."""
    emodel_properties = EmodelProperties(
        threshold_current=0.0, holding_current=-0.05
    )
    cell_kwargs = {
        "template_path": "template.hoc",
        "morphology_path": "morph.asc",
        "template_format": "v5",
        "emodel_properties": emodel_properties,
    }
    cell = MagicMock()
    cell.hypamp = None
    with patch("bluecellulab.tools.bluecellulab.Cell", return_value=cell) as cell_cls, patch(
        "bluecellulab.tools._probe_rheobase_and_rin", return_value=(0.1, 200.0)
    ):
        holding, rheobase, rin = compute_memodel_properties_from_template_subprocess(
            cell_kwargs, holding_current=-0.05, spike_threshold_voltage=-30
        )

    assert cell_cls.call_args.kwargs["cell_id"].id == 0
    assert cell_cls.call_args.kwargs["cell_id"].population_name == ""
    assert (holding, rheobase, rin) == (0.0, 0.1, 200.0)
    cell.delete.assert_called_once()


@pytest.mark.v6
def test_compute_memodel_properties_from_template_bluepyopt_hippocampus():
    """Hippocampus: from_template and cell API differ; both locked.

    Expected: different rheobase/Rin. Reason: ``compute_memodel_properties(cell)``
    forks after the parent Cell exists, so the worker integrates an unused ghost
    cell and cvode picks different steps. ``from_template`` builds only the measured
    cell in the worker (more accurate for cvode). Callers switching to from_template
    will see changed calibration values; that is intentional.
    """
    emodel_properties = EmodelProperties(
        threshold_current=0.1, holding_current=-0.05, AIS_scaler=1.0
    )
    kw = dict(
        template_path=(
            script_dir
            / "examples/hippocampus_opt_cell_template/electrophysiology/cell.hoc"
        ),
        morphology_path=(
            script_dir / "examples/hippocampus_opt_cell_template/morphology/cell.asc"
        ),
        template_format="bluepyopt",
        emodel_properties=emodel_properties,
    )
    # from_template first so the parent process has no Cell before that fork.
    from_template = compute_memodel_properties_from_template(**kw)
    assert from_template["holding_current"] == pytest.approx(0.0)
    assert from_template["rheobase"] == pytest.approx(0.034891986455173184)
    assert from_template["rin"] == pytest.approx(1432.9937925617862)

    existing = compute_memodel_properties(Cell(**kw))
    assert existing["holding_current"] == pytest.approx(0.0)
    assert existing["rheobase"] == pytest.approx(0.03396649209097065)
    assert existing["rin"] == pytest.approx(1472.0389690669485)
    assert from_template["rheobase"] != pytest.approx(existing["rheobase"])
    assert from_template["rin"] != pytest.approx(existing["rin"])


def _cadpyr_memodel_kwargs():
    return {
        "template_path": (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc"
        ),
        "morphology_path": (
            script_dir
            / "examples/circuit_sonata_quick_scx/components/morphologies/asc/"
            "rr110330_C3_idA.asc"
        ),
        "template_format": "v6",
        "emodel_properties": EmodelProperties(
            threshold_current=0.1, holding_current=-0.11, AIS_scaler=1.11
        ),
    }


def _probe_worker_setup(extra_cell: bool, gid: int):
    """Review repro: alone ID 0/1 vs unused ghost cell + ID 1 in the worker."""
    kw = _cadpyr_memodel_kwargs()
    # Keep a reference so the ghost is not GC'd during the probe (cvode sees it).
    ghost = None
    if extra_cell:
        ghost = bluecellulab.Cell(**kw, cell_id=CellId("", 0))
    cell = bluecellulab.Cell(**kw, cell_id=CellId("", gid))
    return _probe_rheobase_and_rin(cell, -0.11, -30, kw["template_path"])


def test_probe_worker_setup_covered_in_process():
    """Cover _probe_worker_setup in-process (IsolatedProcess is not measured)."""
    measured = MagicMock()
    with patch("bluecellulab.Cell", side_effect=[MagicMock(), measured]) as cell_cls, patch(
        "tests.test_tools._probe_rheobase_and_rin", return_value=(0.1, 300.0)
    ) as probe:
        assert _probe_worker_setup(True, 1) == (0.1, 300.0)

    assert cell_cls.call_count == 2
    assert cell_cls.call_args_list[0].kwargs["cell_id"].id == 0
    assert cell_cls.call_args_list[1].kwargs["cell_id"].id == 1
    probe.assert_called_once()
    assert probe.call_args.args[0] is measured


@pytest.mark.v6
def test_probe_worker_setups_match_from_template_and_cell_api():
    """Alone-in-worker matches from_template; ghost+ID1 matches compute_memodel_properties.

    Expected: alone != ghost. Reason: an unused sibling Cell in the NEURON topology
    changes cvode step sizes; from_template avoids that and is more accurate.
    Must not create a parent Cell before these forks (inherited state would pollute
    every worker). Public APIs run after the matrix.
    """
    set_neuron_globals(temperature=34.0, v_init=-80.0)
    alone0 = alone1 = ghost1 = None
    with IsolatedProcess() as runner:
        alone0 = runner.apply(_probe_worker_setup, [False, 0])
    with IsolatedProcess() as runner:
        alone1 = runner.apply(_probe_worker_setup, [False, 1])
    with IsolatedProcess() as runner:
        ghost1 = runner.apply(_probe_worker_setup, [True, 1])

    # Deterministic cvode: gid alone does not change results.
    assert alone0 == alone1
    assert alone0[0] == pytest.approx(0.13918982460553736)
    assert alone0[1] == pytest.approx(334.1862365025271)
    # Unused ghost cell changes the variable-step solution (expected divergence).
    assert ghost1[0] == pytest.approx(0.13921147797816324)
    assert ghost1[1] == pytest.approx(334.1378411934045)
    assert alone0 != ghost1

    kw = _cadpyr_memodel_kwargs()
    from_template = compute_memodel_properties_from_template(**kw)
    assert from_template["rheobase"] == pytest.approx(alone0[0])
    assert from_template["rin"] == pytest.approx(alone0[1])

    cell = Cell(**kw)
    existing = compute_memodel_properties(cell)
    assert existing["rheobase"] == pytest.approx(ghost1[0])
    assert existing["rin"] == pytest.approx(ghost1[1])


def _stochkv_voltage_for_gid(gid: int):
    """Short fixed-step run after re_init_rng; StochKv MCellRan4 seed uses CellRef.gid."""
    set_neuron_globals(temperature=34.0, v_init=-80.0)
    cell = bluecellulab.Cell(
        template_path=script_dir / "examples/cell_example_cstut/cSTUT_7.hoc",
        morphology_path=script_dir / "examples/cell_example_cstut",
        template_format="v5",
        emodel_properties=EmodelProperties(
            threshold_current=0.0, holding_current=-0.05
        ),
        cell_id=CellId("", gid),
    )
    try:
        cell.re_init_rng()
        sec = cell.get_section("soma[0]")
        cell.add_voltage_recording(cell.sections["soma[0]"], 0.5)
        cell.add_ramp(0, 500, 0.15, 0.15, section=sec, segx=0.5)
        bluecellulab.Simulation().run(200, cvode=False)
        return cell.get_voltage_recording(section=sec, segx=0.5)
    finally:
        cell.delete()


def test_stochkv_voltage_for_gid_covered_in_process():
    """Cover _stochkv_voltage_for_gid in-process (IsolatedProcess is not measured)."""
    cell = MagicMock()
    cell.sections = {"soma[0]": MagicMock()}
    cell.get_section.return_value = cell.sections["soma[0]"]
    cell.get_voltage_recording.return_value = np.array([0.0, 1.0])
    with patch("tests.test_tools.set_neuron_globals"), patch(
        "bluecellulab.Cell", return_value=cell
    ) as cell_cls, patch("bluecellulab.Simulation") as sim_cls:
        assert np.array_equal(_stochkv_voltage_for_gid(0), np.array([0.0, 1.0]))

    assert cell_cls.call_args.kwargs["cell_id"].id == 0
    cell.re_init_rng.assert_called_once()
    sim_cls.return_value.run.assert_called_once_with(200, cvode=False)
    cell.delete.assert_called_once()


@pytest.mark.v5
def test_stochkv_noise_depends_on_cell_gid():
    """When StochKv RNG is seeded, gid 0 and gid 1 produce different voltage traces.

    Explains why from_template pins CellId(\"\", 0): StochKv3/bIR-style models that seed
    from gid in INITIAL (or via re_init_rng) would otherwise depend on Cell.last_id.
    """
    with IsolatedProcess() as runner:
        v0 = runner.apply(_stochkv_voltage_for_gid, [0])
    with IsolatedProcess() as runner:
        v1 = runner.apply(_stochkv_voltage_for_gid, [1])
    assert not np.allclose(v0, v1)


def test_calculate_max_thresh_current(mock_cell, mock_calculate_SS_voltage, mock_calculate_input_resistance):
    """Test the calculate_max_thresh_current function."""
    with patch('bluecellulab.tools.calculate_SS_voltage', mock_calculate_SS_voltage), \
         patch('bluecellulab.tools.calculate_input_resistance', mock_calculate_input_resistance):

        threshold_voltage = -30.0
        upperbound_threshold_current = calculate_max_thresh_current(mock_cell, threshold_voltage)

        assert upperbound_threshold_current == (threshold_voltage - mock_calculate_SS_voltage.return_value) / mock_calculate_input_resistance.return_value
        assert upperbound_threshold_current == 0.35


def test_calculate_rheobase(mock_cell, mock_calculate_SS_voltage, mock_calculate_input_resistance, mock_search_threshold_current):
    """Test the calculate_rheobase function."""
    with patch('bluecellulab.tools.calculate_SS_voltage', mock_calculate_SS_voltage), \
         patch('bluecellulab.tools.calculate_input_resistance', mock_calculate_input_resistance), \
         patch('bluecellulab.tools.search_threshold_current', mock_search_threshold_current):

        threshold_voltage = -30.0
        threshold_search_stim_start = 300.0
        threshold_search_stim_stop = 1000.0

        rheobase = calculate_rheobase(mock_cell, threshold_voltage, threshold_search_stim_start, threshold_search_stim_stop)

        assert rheobase == mock_search_threshold_current.return_value


def test_validate_section_and_segment_valid(mock_cell_sections):
    """Test validate_section_and_segment with valid inputs."""
    # Valid section and segment position
    validate_section_and_segment(mock_cell_sections, 'soma', 0.5)
    validate_section_and_segment(mock_cell_sections, 'axon[0]', 0.0)
    validate_section_and_segment(mock_cell_sections, 'axon[1]', 1.0)


def test_validate_section_and_segment_invalid_section(mock_cell_sections):
    """Test validate_section_and_segment with an invalid section name."""
    with pytest.raises(ValueError, match="Section 'dend' not found in the cell model."):
        validate_section_and_segment(mock_cell_sections, 'dend', 0.5)


def test_validate_section_and_segment_invalid_segment_position(mock_cell_sections):
    """Test validate_section_and_segment with an invalid segment position."""
    with pytest.raises(ValueError, match="Segment position must be between 0.0 and 1.0, got -0.1."):
        validate_section_and_segment(mock_cell_sections, 'soma', -0.1)
    with pytest.raises(ValueError, match="Segment position must be between 0.0 and 1.0, got 1.1."):
        validate_section_and_segment(mock_cell_sections, 'axon[0]', 1.1)


def test_currents_vars():
    """Unit test for currents_vars."""
    # Mock section and segment
    class MockSeg:
        pass

    class MockSection:
        def __call__(self):
            return MockSeg()

        def psection(self):
            return {
                "ions": {
                    "na": {"ina": 1, "ena": 2},
                    "k": {"ik": 1, "ek": 2},
                },
                "density_mechs": {
                    "pas": {"i": 1},
                    "hh": {"m": 1, "h": 2},  # no 'i', should not appear
                },
                "point_mechs": {
                    "ExpSyn": {"i": 1},
                    "AlphaSyn": {"g": 1},  # no 'i', should not appear
                }
            }

    section = MockSection()
    result = currents_vars(section)
    # Should include ionic currents
    assert result["ina"] == {"units": "mA/cm²", "kind": "ionic_current"}
    assert result["ik"] == {"units": "mA/cm²", "kind": "ionic_current"}
    # Should include nonspecific current from pas
    assert result["i_pas"] == {"units": "mA/cm²", "kind": "nonspecific_current"}
    # Should not include AlphaSyn.i or hh.i
    assert "i_AlphaSyn" not in result
    assert "i_hh" not in result
    assert "ittx" not in result  # ttx current should be excluded


def test_mechs_vars():
    """Unit test for mechs_vars."""
    class MockSection:
        def __call__(self, x):
            return None

        def psection(self):
            return {
                "density_mechs": {
                    "hh": {"m": 1, "h": 2},
                    "pas": {"i": 1}
                },
                "point_mechs": {
                    "ExpSyn": {"i": 1, "g": 2},
                    "AlphaSyn": {"g": 1}
                }
            }

    section = MockSection()
    # Test without point mechanisms
    result = mechs_vars(section, include_point_mechs=False)
    assert "mech" in result
    assert "hh" in result["mech"]
    assert "pas" in result["mech"]
    assert sorted(result["mech"]["hh"]) == ["h", "m"]
    assert sorted(result["mech"]["pas"]) == ["i"]
    assert "point" not in result

    # Test with point mechanisms
    result_with_point = mechs_vars(section, include_point_mechs=True)
    assert "point" in result_with_point
    assert "ExpSyn" in result_with_point["point"]
    assert sorted(result_with_point["point"]["ExpSyn"]) == ["g", "i"]
    assert "AlphaSyn" in result_with_point["point"]
    assert sorted(result_with_point["point"]["AlphaSyn"]) == ["g"]
