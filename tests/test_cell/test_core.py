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
"""Unit tests for Cell.py"""

import math
import random
import warnings
from pathlib import Path
from unittest.mock import patch
import uuid

import neuron
import numpy as np
import pytest
import re

import bluecellulab
from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.cell.template import NeuronTemplate, public_hoc_cell, shorten_and_hash_string
from bluecellulab.exceptions import BluecellulabError
from bluecellulab import CircuitSimulation
from bluecellulab.cell.recording import section_to_voltage_recording_str


warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v5
def test_longname():
    """Cell: Test loading cell with long name"""

    cell = bluecellulab.Cell(
        "%s/examples/cell_example1/test_cell_longname1.hoc" % str(parent_dir),
        "%s/examples/cell_example1" % str(parent_dir))
    assert isinstance(cell, bluecellulab.Cell)

    del cell


@pytest.mark.v5
@patch('uuid.uuid4')
def test_load_template(mock_uuid):
    """Test the neuron template loading."""
    id = '12345678123456781234567812345678'
    mock_uuid.return_value = uuid.UUID(id)

    hoc_path = parent_dir / "examples/cell_example1/test_cell.hoc"
    morph_path = parent_dir / "examples/cell_example1/test_cell.asc"
    template = NeuronTemplate(hoc_path, morph_path, "v5", None)
    template_name = template.template_name
    assert template_name == f"test_cell_bluecellulab_{id}"


def test_shorten_and_hash_string():
    """Unit test for the shorten_and_hash_string function."""
    with pytest.raises(ValueError):
        shorten_and_hash_string(label="1", hash_length=21)

    short_label = "short-label"
    assert shorten_and_hash_string(short_label) == short_label

    long_label = "test-cell" * 10
    assert len(shorten_and_hash_string(long_label)) < len(long_label)


@pytest.mark.v5
class TestCellBaseClass1:
    """First Cell test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
            "%s/examples/cell_example1" % str(parent_dir))
        assert isinstance(self.cell, bluecellulab.Cell)

    def teardown_method(self):
        """Teardown"""
        del self.cell

    def test_fields(self):
        """Cell: Test the fields of a Cell object"""
        assert isinstance(self.cell.soma, neuron.nrn.Section)
        assert isinstance(self.cell.axonal[0], neuron.nrn.Section)
        assert math.fabs(self.cell.threshold - 0.184062) < 0.00001
        assert math.fabs(self.cell.hypamp - -0.070557) < 0.00001
        # Lowered precision because of commit
        # 81a7a398214f2f5fba199ac3672c3dc3ccb6b103
        # in nrn simulator repo
        assert math.fabs(self.cell.soma.diam - 13.78082) < 0.0001
        assert math.fabs(self.cell.soma.L - 19.21902) < 0.00001
        assert math.fabs(self.cell.basal[2].diam - 0.595686) < 0.00001
        assert math.fabs(self.cell.basal[2].L - 178.96164) < 0.00001
        assert math.fabs(self.cell.apical[10].diam - 0.95999) < 0.00001
        assert math.fabs(self.cell.apical[10].L - 23.73195) < 0.00001

    def test_get_psection(self):
        """Cell: Test cell.get_psection"""
        idx = 0
        name = "Cell[0].soma[0]"
        assert isinstance(
            self.cell.get_psection(idx).hsection, neuron.nrn.Section)
        assert isinstance(
            self.cell.get_psection(name).hsection, neuron.nrn.Section)
        assert self.cell.get_psection(idx) == self.cell.get_psection(name)
        with pytest.raises(BluecellulabError):
            self.cell.get_psection(None)
        with pytest.raises(BluecellulabError):
            self.cell.get_psection(5.8673453123)

    def test_add_recording(self):
        """Cell: Test cell.add_recording"""
        varname = 'self.apical[1](0.5)._ref_v'
        self.cell.add_recording(varname)
        assert varname in self.cell.recordings

        second_varname = 'self.apical[1](0.6)._ref_v'
        self.cell.add_recording(second_varname, dt=0.025)
        assert second_varname in self.cell.recordings

    def test_add_recordings(self):
        """Cell: Test cell.add_recordings"""
        varnames = [
            'self.axonal[0](0.25)._ref_v',
            'self.soma(0.5)._ref_v',
            'self.apical[1](0.5)._ref_v']
        self.cell.add_recordings(varnames)
        for varname in varnames:
            assert varname in self.cell.recordings

    def test_add_ais_recording(self):
        """Cell Test add_ais_recording."""
        self.cell.add_ais_recording()
        ais_key = "self.axonal[1](0.5)._ref_v"
        assert ais_key in self.cell.recordings

    def test_add_allsections_voltagerecordings(self):
        """Cell: Test cell.add_allsections_voltagerecordings"""
        self.cell.add_allsections_voltagerecordings()

        all_sections = self.cell.cell.getCell().all
        for section in all_sections:
            varname = 'neuron.h.%s(0.5)._ref_v' % section.name()
            assert varname in self.cell.recordings

    def test_manual_add_allsection_voltage_recordings(self):
        """Cell: Test cell.add_voltage_recording."""
        all_sections = self.cell.cell.getCell().all
        last_section = None
        for section in all_sections:
            self.cell.add_voltage_recording(section, 0.5)
            recording = self.cell.get_voltage_recording(section, 0.5)
            assert len(recording) == 0
            last_section = section
        with pytest.raises(BluecellulabError):
            self.cell.get_voltage_recording(last_section, 0.7)

    def test_get_allsections_voltagerecordings(self):
        """Cell: Test cell.get_allsections_voltagerecordings."""
        self.cell.recordings.clear()

        with pytest.raises(BluecellulabError):
            recordings = self.cell.get_allsections_voltagerecordings()

        self.cell.add_allsections_voltagerecordings()
        recordings = self.cell.get_allsections_voltagerecordings()
        assert len(recordings) == len(self.cell.recordings)
        for recording in recordings:
            assert any(recording in s for s in self.cell.recordings)

    def test_euclid_section_distance(self):
        """Cell: Test cell.euclid_section_distance"""

        random.seed(1)

        location1 = 0.0
        location2 = 1.0
        for _ in range(1000):
            hsection1 = random.choice(random.choice(
                [self.cell.apical, self.cell.somatic, self.cell.basal]))
            hsection2 = random.choice(random.choice(
                [self.cell.apical, self.cell.somatic, self.cell.basal]))
            distance_euclid = \
                self.cell.euclid_section_distance(hsection1=hsection1,
                                                  hsection2=hsection2,
                                                  location1=location1,
                                                  location2=location2,
                                                  projection='xyz')
            x1 = neuron.h.x3d(0, sec=hsection1)
            y1 = neuron.h.y3d(0, sec=hsection1)
            z1 = neuron.h.z3d(0, sec=hsection1)
            x2 = neuron.h.x3d(neuron.h.n3d(sec=hsection2) - 1, sec=hsection2)
            y2 = neuron.h.y3d(neuron.h.n3d(sec=hsection2) - 1, sec=hsection2)
            z2 = neuron.h.z3d(neuron.h.n3d(sec=hsection2) - 1, sec=hsection2)

            distance_hand = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            assert distance_euclid == distance_hand


@pytest.mark.v5
class TestCellBaseClassVClamp:

    """First Cell test class"""

    def setup_method(self):
        """Setup"""
        self.cell = bluecellulab.Cell(
            "%s/examples/cell_example1/test_cell.hoc" % str(parent_dir),
            "%s/examples/cell_example1" % str(parent_dir))
        assert (isinstance(self.cell, bluecellulab.Cell))

    def teardown_method(self):
        """Teardown"""
        del self.cell

    def test_add_voltage_clamp(self):
        """Cell: Test add_voltage_clamp"""

        level = -90
        stop_time = 50
        total_time = 200
        rs = .1
        vclamp = self.cell.add_voltage_clamp(
            stop_time=stop_time,
            level=level,
            current_record_name='test_vclamp',
            rs=rs)

        assert vclamp.amp1 == level
        assert vclamp.dur1 == stop_time
        assert vclamp.dur2 == 0
        assert vclamp.dur3 == 0
        assert vclamp.rs == rs

        sim = bluecellulab.Simulation()
        sim.add_cell(self.cell)
        sim.run(total_time, dt=.1, cvode=False)

        time = self.cell.get_time()
        current = self.cell.get_recording('test_vclamp')
        import numpy as np

        voltage = self.cell.get_soma_voltage()

        voltage_vc_end = np.mean(
            voltage[np.where((time < stop_time) & (time > .9 * stop_time))])

        assert (abs(voltage_vc_end - level) < .1)

        voltage_end = np.mean(
            voltage
            [np.where((time < total_time) & (time > .9 * total_time))])

        assert (abs(voltage_end - (-73)) < 1)

        current_vc_end = np.mean(
            current[np.where((time < stop_time) & (time > .9 * stop_time))])

        assert (abs(current_vc_end - (-.1)) < .01)

        current_after_vc_end = np.mean(
            current[
                np.where((time > stop_time) & (time < 1.1 * stop_time))])

        assert current_after_vc_end == 0.0


class TestCellSpikes:

    def setup_method(self):
        self.cell = bluecellulab.Cell(
            f"{parent_dir}/examples/cell_example1/test_cell.hoc",
            f"{parent_dir}/examples/cell_example1")
        self.sim = bluecellulab.Simulation()
        self.sim.add_cell(self.cell)

    @pytest.mark.v5
    def test_get_recorded_spikes(self):
        """Cell: Test get_recorded_spikes."""
        self.cell.start_recording_spikes(None, "soma", -30)
        self.cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
        self.sim.run(24, cvode=False)
        spikes = self.cell.get_recorded_spikes("soma")
        ground_truth = [3.350000000100014, 11.52500000009988, 19.9750000000994]
        assert np.allclose(spikes, ground_truth)

    @pytest.mark.v5
    def test_create_netcon_spikedetector(self):
        """Test creating a NetCon for spike detection."""
        threshold = -29.0
        netcon_ais = self.cell.create_netcon_spikedetector(None, "AIS", threshold)
        netcon_soma = self.cell.create_netcon_spikedetector(None, "soma", threshold)
        assert netcon_ais.threshold == threshold, "AIS NetCon threshold mismatch"
        assert netcon_soma.threshold == threshold, "Soma NetCon threshold mismatch"

        # Test invalid location
        with pytest.raises(ValueError):
            self.cell.create_netcon_spikedetector(None, "Dendrite", threshold)

    @pytest.mark.v5
    def test_create_netcon_spikedetector_custom_location(self):
        """Test creating a NetCon with a custom location."""
        threshold = -30.0
        # Test valid custom locations
        valid_locations = [
            ("soma[0](0.5)", 0.5),
            ("soma[0](1.0)", 1.0),
            ("axon[1](0.3)", 0.3),
        ]
        for location, pos in valid_locations:
            netcon_custom = self.cell.create_netcon_spikedetector(None, location, threshold)
            assert netcon_custom.threshold == threshold, f"Threshold mismatch for location {location}"
            # Additional checks for source voltage if accessible (not included in this context)

    @pytest.mark.v5
    def test_invalid_location_format(self):
        """Test handling of invalid location formats."""
        threshold = -30.0
        invalid_locations = [
            "soma[abc](0.5)",  # Non-integer in square brackets
            "soma[0](abc)",    # Non-decimal in parentheses
            "soma[0]abc",      # Missing parentheses
            "soma()",          # Empty parentheses
            "soma[0](1.5.3)",  # Invalid decimal format
            "soma[0](1.5",     # Unmatched parentheses
        ]
        for location in invalid_locations:
            with pytest.raises(ValueError, match=re.escape(f"Invalid location format: {location}")):
                self.cell.create_netcon_spikedetector(None, location, threshold)

    @pytest.mark.v5
    def test_invalid_segment_or_section(self):
        """Test creating a NetCon with an invalid segment index or section."""
        threshold = -30.0

        # Non-existent section name
        with pytest.raises(ValueError, match="Invalid spike detection location:"):
            self.cell.create_netcon_spikedetector(None, "invalid_section[0](0.5)", threshold)

        # Out-of-bounds segment index
        with pytest.raises(ValueError, match=re.escape("Invalid spike detection location: soma[999](0.5)")):
            self.cell.create_netcon_spikedetector(None, "soma[999](0.5)", threshold)

        # Invalid position (greater than 1.0 or negative)
        with pytest.raises(ValueError, match=re.escape("Invalid spike detection location: soma[0](1.5)")):
            self.cell.create_netcon_spikedetector(None, "soma[0](1.5)", threshold)
        with pytest.raises(ValueError, match=re.escape("Invalid spike detection location: soma[0](-0.3)")):
            self.cell.create_netcon_spikedetector(None, "soma[0](-0.3)", threshold)

    @pytest.mark.v5
    def test_default_position_in_custom_location(self):
        """Test default position when parentheses are omitted in a valid custom location."""
        threshold = -30.0
        location = "soma[0]"  # No position specified; should default to 0.5
        netcon = self.cell.create_netcon_spikedetector(None, location, threshold)
        assert netcon.threshold == threshold, "Threshold mismatch for default position"


@pytest.mark.v6
def test_add_dendrogram():
    """Cell: Test get_recorded_spikes."""
    emodel_properties = EmodelProperties(threshold_current=1.1433533430099487,
                                         holding_current=1.4146618843078613,
                                         AIS_scaler=1.4561502933502197,
                                         soma_scaler=1.0)
    cell = bluecellulab.Cell(
        "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
        "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
        template_format="v6",
        emodel_properties=emodel_properties)
    cell.add_plot_window(['self.soma(0.5)._ref_v'])
    output_path = "cADpyr_L2TPC_dendrogram.png"
    cell.add_dendrogram(save_fig_path=output_path)
    sim = bluecellulab.Simulation()
    sim.add_cell(cell)
    cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
    sim.run(24, cvode=False)
    assert Path(output_path).is_file()


@pytest.mark.v6
class TestCellV6:
    """Test class for testing Cell object functionalities with v6 template."""

    def setup_method(self):
        """Setup."""
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0
        )
        self.cell = bluecellulab.Cell(
            "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
            "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
            template_format="v6",
            emodel_properties=emodel_properties
        )

    def test_repr_and_str(self):
        """Test the repr and str representations of a cell object."""
        # >>> print(cell)
        # Cell Object: <bluecellulab.cell.core.Cell object at 0x7f73b3fb2550>.
        # NEURON ID: cADpyr_L2TPC_bluecellulab_0x7f73b48e2510.
        # make sure NEURON template name is in the string representation
        assert self.cell.cell.hname().split('[')[0] in str(self.cell)

    def test_area(self):
        """Test the cell's area computation."""
        assert self.cell.area() == 5812.493415302344

    def test_cell_id(self):
        """Test for checking if cell_id is different btw. 2 cells when unspecified."""
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0
        )
        cell2 = bluecellulab.Cell(
            "%s/examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc" % str(parent_dir),
            "%s/examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc" % str(parent_dir),
            template_format="v6",
            emodel_properties=emodel_properties
        )
        assert self.cell.cell_id != cell2.cell_id

    def test_get_childrensections(self):
        """Test the get_childrensections method."""
        res = self.cell.get_childrensections(self.cell.soma)
        assert len(res) == 3

    def test_get_parentsection(self):
        """Test the get_parentsection method."""
        section = self.cell.get_childrensections(self.cell.soma)[0]
        res = self.cell.get_parentsection(section)
        assert res == self.cell.soma

    def test_somatic_sections(self):
        """Test that somatic property returns a non-empty list sections."""
        assert isinstance(self.cell.somatic, list)
        assert len(self.cell.somatic) > 0

    def test_basal_sections(self):
        """Test that basal property returns a non-empty list of sections."""
        assert isinstance(self.cell.basal, list)
        assert len(self.cell.basal) > 0

    def test_apical_sections(self):
        """Test that apical property returns a non-empty list sections."""
        assert isinstance(self.cell.apical, list)
        assert len(self.cell.apical) > 0

    def test_axonal_sections(self):
        """Test that axonal property returns a non-empty list of sections."""
        assert isinstance(self.cell.axonal, list)
        assert len(self.cell.axonal) > 0

    def test_all_sections(self):
        """Test that the sections property returns a non-empty dict of sections."""
        assert isinstance(self.cell.sections, dict)
        assert "axon[0]" in self.cell.sections
        assert "dend[0]" in self.cell.sections
        assert len(self.cell.sections) > 0

    def test_extract_sections(self):
        """Unit test for _extract_sections."""
        sections = self.cell._extract_sections(public_hoc_cell(self.cell.cell).axonal)
        assert len(sections) == len(self.cell.axonal)

    def test_n_segments(self):
        """Unit test for the n_segments method/property."""
        assert self.cell.n_segments == 247

    def test_add_voltage_recording(self):
        """Test adding a voltage recording to a section."""
        self.cell.add_voltage_recording()
        assert f"neuron.h.{self.cell.soma.name()}(0.5)._ref_v" in self.cell.recordings
        self.cell.add_voltage_recording(self.cell.apical[1])
        assert f"neuron.h.{self.cell.apical[1].name()}(0.5)._ref_v" in self.cell.recordings

    def test_get_voltage_recording(self):
        """Test getting the voltage recording of a section."""
        self.cell.add_voltage_recording()
        self.cell.get_voltage_recording()  # get soma voltage recording
        recording = self.cell.get_voltage_recording(self.cell.soma)
        assert len(recording) == 0
        with pytest.raises(BluecellulabError):
            self.cell.get_voltage_recording(self.cell.basal[0])

    def test_from_template_parameters(self):
        """Unit test creating Cell from template_parameters."""
        new_cell = bluecellulab.Cell.from_template_parameters(self.cell.template_params)
        assert new_cell.template_params == self.cell.template_params

    def test_get_voltage_recording_soma(self):
        """Test get_voltage_recording for the soma section."""
        # Add a voltage recording at the soma
        self.cell.add_voltage_recording(self.cell.soma, segx=0.5)

        self.cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
        sim = bluecellulab.Simulation()
        sim.add_cell(self.cell)
        sim.run(24, cvode=False)

        # Retrieve the voltage recording
        voltage_soma_v1 = self.cell.get_voltage_recording(self.cell.soma, segx=0.5)
        voltage_soma_v2 = self.cell.get_soma_voltage()

        assert np.array_equal(voltage_soma_v1, voltage_soma_v2), "Arrays are not equal"

        # Check the reference name is correct and the recording exists
        reference_name = "self.soma(0.5)._ref_v"
        assert reference_name in self.cell.recordings
        assert isinstance(voltage_soma_v1, np.ndarray)

    def test_get_voltage_recording_apical(self):
        """Test get_voltage_recording for an apical section."""
        apical_section = self.cell.apical[0]

        # Add a voltage recording at an apical section
        self.cell.add_voltage_recording(apical_section, segx=0.5)

        self.cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
        sim = bluecellulab.Simulation()
        sim.add_cell(self.cell)
        sim.run(24, cvode=False)

        # Retrieve the voltage recording
        voltage = self.cell.get_voltage_recording(apical_section, segx=0.5)
        voltage_soma = self.cell.get_soma_voltage()

        # Check the reference name is correct and the recording exists
        reference_name = section_to_voltage_recording_str(apical_section, 0.5)

        assert reference_name in self.cell.recordings
        assert isinstance(voltage, np.ndarray)
        assert not np.array_equal(voltage, voltage_soma), "Arrays should not be equal"

    def test_get_voltage_recording_missing(self):
        """Test the behavior of the `get_voltage_recording` method when
        attempting to retrieve a voltage recording that was not previously added."""
        with pytest.raises(BluecellulabError, match="get_voltage_recording: Voltage recording .* was not added previously using add_voltage_recording"):
            self.cell.get_voltage_recording(self.cell.soma, segx=1.5)


@pytest.mark.v6
def test_add_synapse_replay():
    """Cell: test add_synapse_replay."""
    sonata_sim_path = (
        parent_dir
        / "examples"
        / "sonata_unit_test_sims"
        / "synapse_replay"
        / "simulation_config.json"
    )
    circuit_sim = bluecellulab.CircuitSimulation(sonata_sim_path)
    circuit_sim.spike_threshold = -900.0
    cell_id = ("hippocampus_neurons", 0)
    circuit_sim.instantiate_gids(cell_id,
                                 add_stimuli=True, add_synapses=True,
                                 interconnect_cells=False)
    cell = circuit_sim.cells[cell_id]
    assert len(cell.connections) == 3
    assert cell.connections[
        ("hippocampus_projections__hippocampus_neurons__chemical", 0)
    ].pre_spiketrain.tolist() == [16.0, 22.0, 48.0]


@pytest.mark.v6
class TestWithinCircuit:

    def setup_method(self):
        """Setup method called before each test method."""
        sonata_sim_path = (
            parent_dir
            / "examples"
            / "sim_quick_scx_sonata_multicircuit"
            / "simulation_config_noinput.json"
        )
        cell_id = ("NodeA", 0)
        circuit_sim = CircuitSimulation(sonata_sim_path)
        circuit_sim.instantiate_gids(cell_id, add_synapses=True, add_stimuli=False)
        self.cell = circuit_sim.cells[cell_id]
        self.circuit_sim = circuit_sim  # for persistance

    def test_pre_gids(self):
        """Test get_pre_gids within a circuit."""
        pre_gids = self.cell.pre_gids()
        assert pre_gids == [0, 1]

    def test_pre_gid_synapse_ids(self):
        """Test pre_gid_synapse_ids within a circuit."""
        pre_gids = self.cell.pre_gids()

        first_gid_synapse_ids = self.cell.pre_gid_synapse_ids(pre_gids[0])
        assert first_gid_synapse_ids == [('NodeB__NodeA__chemical', 0), ('NodeB__NodeA__chemical', 2)]

        second_gid_synapse_ids = self.cell.pre_gid_synapse_ids(pre_gids[1])
        assert len(second_gid_synapse_ids) == 4
        assert second_gid_synapse_ids[0] == ('NodeA__NodeA__chemical', 0)
        assert second_gid_synapse_ids[1] == ('NodeA__NodeA__chemical', 1)
        assert second_gid_synapse_ids[2] == ('NodeB__NodeA__chemical', 1)
        assert second_gid_synapse_ids[3] == ('NodeB__NodeA__chemical', 3)
