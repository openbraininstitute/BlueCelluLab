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
"""Tests for the handling of virtual SONATA node populations.

Virtual populations (spike sources for projections/replay) carry no
morphology, no emodel template and no ``dynamics_params``. Per the SONATA
spec, ``threshold_current``/``holding_current`` are mandatory only for the
*biophysical* node group.

Following neurodamus, such populations must never be instantiated as
biophysical cells: ``CircuitManager.new_node_manager`` returns a lightweight
``VirtualCellPopulation`` before any node data is read. Previously
bluecellulab tried to build a real ``Cell`` for them, which raised
``KeyError: '@dynamics:threshold_current'``.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from bluecellulab.circuit import CellId, SonataCircuitAccess
from bluecellulab.circuit.circuit_access import EmodelProperties

parent_dir = Path(__file__).resolve().parent.parent

# This circuit has both a biophysical population (hippocampus_neurons) and a
# virtual one (hippocampus_projections).
hipp_circuit_with_projections = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)

BIOPHYSICAL_POP = "hippocampus_neurons"
VIRTUAL_POP = "hippocampus_projections"


class TestIsVirtualPopulation:
    """``SonataCircuitAccess.is_virtual_population``."""

    def setup_method(self):
        self.circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)

    def test_virtual_population_is_detected(self):
        assert self.circuit_access.is_virtual_population(VIRTUAL_POP) is True

    def test_biophysical_population_is_not_virtual(self):
        assert self.circuit_access.is_virtual_population(BIOPHYSICAL_POP) is False

    def test_unknown_population_is_not_virtual(self):
        """An unknown population must not raise, it falls back to False."""
        assert self.circuit_access.is_virtual_population("does_not_exist") is False


class TestGetEmodelPropertiesVirtual:
    """``get_emodel_properties`` must not touch dynamics of virtual cells."""

    def setup_method(self):
        self.circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)

    def test_returns_none_for_virtual_cell(self):
        """Virtual cells have no emodel, so there is nothing to return.

        Before the fix this raised KeyError('@dynamics:threshold_current').
        """
        cell_id = CellId(VIRTUAL_POP, 0)
        assert self.circuit_access.get_emodel_properties(cell_id) is None

    def test_returns_real_values_for_biophysical_cell(self):
        """The biophysical path must keep working unchanged."""
        cell_id = CellId(BIOPHYSICAL_POP, 0)
        emodel_properties = self.circuit_access.get_emodel_properties(cell_id)
        assert isinstance(emodel_properties, EmodelProperties)
        assert emodel_properties.threshold_current is not None
        assert emodel_properties.holding_current is not None


class TestGetEmodelPropertiesDefaults:
    """Absent threshold/holding currents fall back to 0.0.

    neurodamus treats them as optional: ``io/cell_readers.py`` sets them to
    ``None`` when the population has no such dynamics attributes and
    ``metype.py`` substitutes 0.0. Mirrored here instead of raising KeyError.
    """

    def _circuit_access_returning(self, properties: pd.Series) -> SonataCircuitAccess:
        circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)
        # Replace the snap circuit so `nodes[pop].get(id)` yields our series.
        # `type` is set explicitly so the population is not treated as virtual.
        mocked_circuit = MagicMock()
        mocked_population = mocked_circuit.nodes.__getitem__.return_value
        mocked_population.get.return_value = properties
        mocked_population.type = "biophysical"
        circuit_access._circuit = mocked_circuit
        return circuit_access

    def test_missing_currents_default_to_zero(self, caplog):
        circuit_access = self._circuit_access_returning(pd.Series(dtype=object))
        with caplog.at_level(logging.WARNING):
            emodel_properties = circuit_access.get_emodel_properties(
                CellId(BIOPHYSICAL_POP, 0)
            )
        assert emodel_properties == EmodelProperties(
            threshold_current=0.0, holding_current=0.0, AIS_scaler=1.0, soma_scaler=1.0
        )
        assert "'@dynamics:threshold_current' not found" in caplog.text
        assert "'@dynamics:holding_current' not found" in caplog.text

    def test_present_currents_are_used(self):
        circuit_access = self._circuit_access_returning(
            pd.Series(
                {
                    "@dynamics:threshold_current": 0.4,
                    "@dynamics:holding_current": -0.1,
                    "@dynamics:AIS_scaler": 1.5,
                    "@dynamics:soma_scaler": 2.5,
                }
            )
        )
        emodel_properties = circuit_access.get_emodel_properties(
            CellId(BIOPHYSICAL_POP, 0)
        )
        assert emodel_properties == EmodelProperties(
            threshold_current=0.4, holding_current=-0.1, AIS_scaler=1.5, soma_scaler=2.5
        )


class TestFilterOutVirtualCells:
    """``CircuitSimulation._filter_out_virtual_cells``.

    Mirrors neurodamus, which never routes virtual populations through cell
    creation. The cells stay usable as presynaptic/spike-replay sources,
    which is handled via edge populations rather than ``Cell`` objects.
    """

    @staticmethod
    def _simulation():
        """A CircuitSimulation-like object with only what the filter needs."""
        from bluecellulab.circuit_simulation import CircuitSimulation

        simulation = CircuitSimulation.__new__(CircuitSimulation)
        simulation.circuit_access = SonataCircuitAccess(
            hipp_circuit_with_projections
        )
        return simulation

    def test_virtual_cells_are_dropped(self, caplog):
        simulation = self._simulation()
        cell_ids = [CellId(VIRTUAL_POP, 0), CellId(VIRTUAL_POP, 1)]
        with caplog.at_level(logging.WARNING):
            assert simulation._filter_out_virtual_cells(cell_ids) == []
        assert VIRTUAL_POP in caplog.text
        assert "Skipping 2 cell(s)" in caplog.text

    def test_biophysical_cells_are_kept(self, caplog):
        simulation = self._simulation()
        cell_ids = [CellId(BIOPHYSICAL_POP, 0), CellId(BIOPHYSICAL_POP, 1)]
        with caplog.at_level(logging.WARNING):
            assert simulation._filter_out_virtual_cells(cell_ids) == cell_ids
        # nothing was skipped, so no warning should be emitted
        assert "Skipping" not in caplog.text

    def test_mixed_list_keeps_only_biophysical(self):
        simulation = self._simulation()
        biophysical = CellId(BIOPHYSICAL_POP, 0)
        cell_ids = [CellId(VIRTUAL_POP, 0), biophysical, CellId(VIRTUAL_POP, 5)]
        assert simulation._filter_out_virtual_cells(cell_ids) == [biophysical]

    def test_order_is_preserved(self):
        simulation = self._simulation()
        cell_ids = [
            CellId(BIOPHYSICAL_POP, 2),
            CellId(VIRTUAL_POP, 0),
            CellId(BIOPHYSICAL_POP, 0),
            CellId(BIOPHYSICAL_POP, 1),
        ]
        assert simulation._filter_out_virtual_cells(cell_ids) == [
            CellId(BIOPHYSICAL_POP, 2),
            CellId(BIOPHYSICAL_POP, 0),
            CellId(BIOPHYSICAL_POP, 1),
        ]

    def test_empty_list(self):
        simulation = self._simulation()
        assert simulation._filter_out_virtual_cells([]) == []

    def test_circuit_access_missing_the_method_raises(self):
        """``is_virtual_population`` is part of the CircuitAccess protocol.

        Every implementation (Bluepy, SONATA) provides it, so a
        circuit_access object missing it is a programming error and should
        surface as such rather than being silently tolerated.
        """
        from bluecellulab.circuit_simulation import CircuitSimulation

        simulation = CircuitSimulation.__new__(CircuitSimulation)

        class MinimalCircuitAccess:
            """No is_virtual_population attribute at all."""

        simulation.circuit_access = MinimalCircuitAccess()
        cell_ids = [CellId(VIRTUAL_POP, 0), CellId(BIOPHYSICAL_POP, 0)]
        with pytest.raises(AttributeError):
            simulation._filter_out_virtual_cells(cell_ids)


def test_bluepy_circuit_access_has_no_virtual_populations():
    """Legacy (non-SONATA) circuits have no notion of virtual populations."""
    from bluecellulab.circuit.circuit_access.bluepy_circuit_access import (
        BluepyCircuitAccess,
    )

    circuit_access = BluepyCircuitAccess.__new__(BluepyCircuitAccess)
    assert circuit_access.is_virtual_population("anything") is False


def test_circuit_access_protocol_declares_is_virtual_population():
    """The protocol must advertise the method for implementers."""
    from bluecellulab.circuit.circuit_access.definition import CircuitAccess

    assert hasattr(CircuitAccess, "is_virtual_population")
    with pytest.raises(NotImplementedError):
        CircuitAccess.is_virtual_population(
            CircuitAccess.__new__(CircuitAccess), "some_population"
        )
