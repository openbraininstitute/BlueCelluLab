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

from bluecellulab.cell import Cell
from bluecellulab import CircuitSimulation
from bluecellulab.cell.ballstick import create_ball_stick
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.node_id import create_cell_id
from bluecellulab.exceptions import UnsteadyCellError
from bluecellulab.tools import calculate_SS_voltage, calculate_SS_voltage_subprocess, calculate_input_resistance, detect_hyp_current, detect_spike, detect_spike_step, detect_spike_step_subprocess, holding_current, holding_current_subprocess, search_threshold_current, template_accepts_cvode, check_empty_topology, calculate_max_thresh_current, calculate_rheobase, validate_section_and_segment


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

        assert input_resistance == pytest.approx(375.88, abs=0.02)

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
        assert i_hold == pytest.approx(-0.03160848349)
        assert v_control == pytest.approx(v_hold)


def test_check_empty_topology():
    """Unit test for the check_empty_topology function."""
    assert check_empty_topology() is True
    cell = create_ball_stick()
    assert check_empty_topology() is False


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
