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
"""Tests for synapses that target sections absent from the built cell.

An emodel's ``replace_axon()`` collapses the axon into a couple of stubs, but
section indices are assigned (``indexSections()``) on the *full* morphology
beforehand. Afferent synapses placed on the removed axon sections therefore
reference section ids that no longer resolve, and looking them up used to
raise a bare ``KeyError``, aborting the whole simulation.

neurodamus skips such synapses instead - see ``Connection.add_synapses``:
"We may need to skip invalid synapses (e.g. on Axon)". These tests pin that
behaviour down.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bluecellulab.circuit import CellId
from bluecellulab.circuit.synapse_properties import SynapseProperty
from bluecellulab.exceptions import BluecellulabError, SectionDoesNotExistError
from bluecellulab.synapse.synapse_factory import SynapseFactory


class TestSectionDoesNotExistError:
    """The exception itself."""

    def test_carries_section_id(self):
        error = SectionDoesNotExistError(41)
        assert error.section_id == 41

    def test_is_a_bluecellulab_error(self):
        """So existing broad handlers keep working."""
        assert isinstance(SectionDoesNotExistError(1), BluecellulabError)

    def test_default_message_mentions_the_section(self):
        assert "41" in str(SectionDoesNotExistError(41))

    def test_custom_message_is_kept(self):
        error = SectionDoesNotExistError(7, "custom explanation")
        assert str(error) == "custom explanation"
        assert error.section_id == 7


class _FakeCell:
    """Minimal stand-in for a Cell, avoiding any NEURON setup.

    ``existing_section_ids`` mimics the psections mapping of a cell whose axon
    was replaced: only some indices resolve.
    """

    def __init__(self, existing_section_ids):
        self._existing = existing_section_ids
        self.cell_id = CellId("biophysical_neurons", 0)

    def get_psection(self, section_id):
        if section_id not in self._existing:
            raise KeyError(section_id)
        return SimpleNamespace(hsection=f"section_{section_id}")


def _syn_description(section_id, section_pos=0.5):
    return pd.Series(
        {
            SynapseProperty.POST_SECTION_ID: section_id,
            SynapseProperty.AFFERENT_SECTION_POS: section_pos,
        }
    )


class TestDetermineSynapseLocation:
    """``SynapseFactory.determine_synapse_location``."""

    def test_raises_for_missing_section(self):
        """A deleted (axon) section id must raise the typed error, not KeyError."""
        cell = _FakeCell(existing_section_ids={0, 1, 2, 3, 60})
        with pytest.raises(SectionDoesNotExistError) as excinfo:
            SynapseFactory.determine_synapse_location(_syn_description(41), cell)
        assert excinfo.value.section_id == 41

    def test_error_message_is_actionable(self):
        cell = _FakeCell(existing_section_ids={0})
        with pytest.raises(SectionDoesNotExistError) as excinfo:
            SynapseFactory.determine_synapse_location(_syn_description(41), cell)
        message = str(excinfo.value)
        assert "41" in message
        assert "axon" in message.lower()

    def test_existing_section_resolves_normally(self):
        """The happy path must be untouched."""
        cell = _FakeCell(existing_section_ids={60})
        hoc_args = SynapseFactory.determine_synapse_location(
            _syn_description(60, section_pos=0.25), cell
        )
        assert hoc_args.section == "section_60"
        assert hoc_args.location == 0.25

    @pytest.mark.parametrize(
        "section_pos, expected",
        [(0.0, 0.0000001), (1.0, 0.9999999), (1.5, 0.9999999)],
    )
    def test_section_position_is_clamped(self, section_pos, expected):
        """Guard the boundary handling next to the new error path."""
        cell = _FakeCell(existing_section_ids={60})
        hoc_args = SynapseFactory.determine_synapse_location(
            _syn_description(60, section_pos=section_pos), cell
        )
        assert hoc_args.location == expected

    def test_section_id_is_cast_to_int(self):
        """SONATA ids arrive as numpy integers."""
        cell = _FakeCell(existing_section_ids={60})
        description = _syn_description(np.int64(60))
        hoc_args = SynapseFactory.determine_synapse_location(description, cell)
        assert hoc_args.section == "section_60"

    def test_missing_numpy_section_id_reports_plain_int(self):
        cell = _FakeCell(existing_section_ids={60})
        with pytest.raises(SectionDoesNotExistError) as excinfo:
            SynapseFactory.determine_synapse_location(
                _syn_description(np.int64(41)), cell
            )
        assert excinfo.value.section_id == 41
        assert isinstance(excinfo.value.section_id, int)


class TestAddCellSynapsesSkipping:
    """``CircuitSimulation._add_cell_synapses`` skips unresolvable synapses."""

    @staticmethod
    def _simulation(syn_descriptions, failing_section_ids):
        from bluecellulab.circuit_simulation import CircuitSimulation

        simulation = CircuitSimulation.__new__(CircuitSimulation)
        simulation.circuit_format = None
        simulation.get_syn_descriptions = lambda _cell_id: syn_descriptions

        def fake_instantiate(cell_id, syn_id, syn_description, add_minis, popids):
            section_id = int(syn_description[SynapseProperty.POST_SECTION_ID])
            if section_id in failing_section_ids:
                raise SectionDoesNotExistError(section_id)
            simulation.instantiated.append(syn_id)

        simulation.instantiated = []
        simulation._instantiate_synapse = fake_instantiate
        return simulation

    @staticmethod
    def _descriptions(section_ids):
        return pd.DataFrame(
            {
                SynapseProperty.POST_SECTION_ID: section_ids,
                SynapseProperty.AFFERENT_SECTION_POS: [0.5] * len(section_ids),
                "source_popid": [0] * len(section_ids),
                "target_popid": [0] * len(section_ids),
            }
        )

    def test_unresolvable_synapses_are_skipped_not_raised(self, caplog):
        descriptions = self._descriptions([60, 41, 61, 42])
        simulation = self._simulation(descriptions, failing_section_ids={41, 42})
        cell_id = CellId("biophysical_neurons", 0)

        with caplog.at_level(logging.WARNING):
            simulation._add_cell_synapses(cell_id)

        # the resolvable ones were still added
        assert simulation.instantiated == [0, 2]
        assert "Skipped 2/4 synapse(s)" in caplog.text

    def test_warning_is_aggregated_and_lists_section_ids(self, caplog):
        """One warning per cell, not one per synapse (there can be thousands)."""
        descriptions = self._descriptions([41] * 500 + [60])
        simulation = self._simulation(descriptions, failing_section_ids={41})
        cell_id = CellId("biophysical_neurons", 0)

        with caplog.at_level(logging.WARNING):
            simulation._add_cell_synapses(cell_id)

        skipped_warnings = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "Skipped" in record.message
        ]
        assert len(skipped_warnings) == 1
        assert "Skipped 500/501 synapse(s)" in skipped_warnings[0].message
        assert "[41]" in skipped_warnings[0].message

    def test_no_warning_when_everything_resolves(self, caplog):
        descriptions = self._descriptions([60, 61])
        simulation = self._simulation(descriptions, failing_section_ids=set())
        cell_id = CellId("biophysical_neurons", 0)

        with caplog.at_level(logging.WARNING):
            simulation._add_cell_synapses(cell_id)

        assert simulation.instantiated == [0, 1]
        assert "Skipped" not in caplog.text

    def test_all_synapses_unresolvable_still_does_not_raise(self, caplog):
        descriptions = self._descriptions([41, 42])
        simulation = self._simulation(descriptions, failing_section_ids={41, 42})
        cell_id = CellId("biophysical_neurons", 0)

        with caplog.at_level(logging.WARNING):
            simulation._add_cell_synapses(cell_id)

        assert simulation.instantiated == []
        assert "Skipped 2/2 synapse(s)" in caplog.text

    def test_empty_descriptions_warns_about_no_presynaptic_cells(self, caplog):
        simulation = self._simulation(self._descriptions([]), failing_section_ids=set())
        cell_id = CellId("biophysical_neurons", 0)

        with caplog.at_level(logging.WARNING):
            simulation._add_cell_synapses(cell_id)

        assert "no synapses added" in caplog.text
