"""Tests for the excluded-projection warning and the holding current options
of CircuitSimulation.instantiate_gids."""
import logging
from pathlib import Path

import numpy as np
import pytest

from bluecellulab import CircuitSimulation
from bluecellulab.exceptions import BluecellulabError

examples_dir = Path(__file__).resolve().parent.parent / "examples"

PROJECTIONS_SIM = (
    examples_dir / "sonata_unit_test_sims" / "projections" / "simulation_config.json"
)
PROJECTION_EDGES = "hippocampus_projections__hippocampus_neurons__chemical"
HIPP_CELL = ("hippocampus_neurons", 1)

QUICK_SCX_DIR = examples_dir / "sim_quick_scx_sonata"
QUICK_SCX_CELL = ("NodeA", 2)


def _warnings(caplog, text):
    return [r for r in caplog.records if r.levelno == logging.WARNING and text in r.message]


@pytest.mark.v6
class TestExcludedProjectionsWarning:

    @pytest.mark.parametrize("add_projections", [False, None])
    def test_warns_when_projections_excluded(self, caplog, add_projections):
        sim = CircuitSimulation(PROJECTIONS_SIM)
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(
                HIPP_CELL, add_synapses=True, add_projections=add_projections
            )
        records = _warnings(caplog, "Projection edge population(s)")
        assert len(records) == 1
        assert PROJECTION_EDGES in records[0].message
        assert "add_projections=True" in records[0].message

    @pytest.mark.parametrize("add_projections", [True, [PROJECTION_EDGES], PROJECTION_EDGES])
    def test_no_warning_when_projections_selected(self, caplog, add_projections):
        sim = CircuitSimulation(PROJECTIONS_SIM)
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(
                HIPP_CELL, add_synapses=True, add_projections=add_projections
            )
        assert not _warnings(caplog, "Projection edge population(s)")

    def test_no_warning_without_synapses(self, caplog):
        sim = CircuitSimulation(PROJECTIONS_SIM)
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(HIPP_CELL, add_synapses=False)
        assert not _warnings(caplog, "Projection edge population(s)")

    def test_excluded_projection_names(self):
        sim = CircuitSimulation(PROJECTIONS_SIM)
        access = sim.circuit_access
        assert access.excluded_projection_names({"hippocampus_neurons"}, False) == [
            PROJECTION_EDGES
        ]
        assert access.excluded_projection_names({"hippocampus_neurons"}, True) == []
        assert access.excluded_projection_names({"other_population"}, False) == []


@pytest.mark.v6
class TestHoldingCurrent:

    def test_warns_when_holding_current_not_applied(self, caplog):
        sim = CircuitSimulation(QUICK_SCX_DIR / "simulation_config_hypamp.json")
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(QUICK_SCX_CELL, add_stimuli=False)
        records = _warnings(caplog, "non-zero holding_current")
        assert len(records) == 1
        assert "add_holding_current=True" in records[0].message

    def test_no_warning_when_hyperpolarizing_input_applied(self, caplog):
        sim = CircuitSimulation(QUICK_SCX_DIR / "simulation_config_hypamp.json")
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(QUICK_SCX_CELL, add_stimuli=True)
        assert not _warnings(caplog, "non-zero holding_current")

    def test_add_holding_current_matches_hyperpolarizing_input(self, caplog):
        """A constant holding clamp reproduces the reference simulation that
        uses a whole-run 'hyperpolarizing' input, without any config input."""
        sim = CircuitSimulation(QUICK_SCX_DIR / "simulation_config_noinput.json")
        with caplog.at_level(logging.WARNING):
            sim.instantiate_gids(QUICK_SCX_CELL, add_stimuli=False, add_holding_current=True)
        assert not _warnings(caplog, "non-zero holding_current")
        t_stop = 10.0
        sim.run(t_stop)
        voltage = sim.get_voltage_trace(QUICK_SCX_CELL, 0, t_stop, 0.025)[:-1]

        ref_sim = CircuitSimulation(QUICK_SCX_DIR / "simulation_config_hypamp.json")
        ref_voltage = ref_sim.get_mainsim_voltage_trace(QUICK_SCX_CELL)
        assert np.sqrt(np.mean((voltage - ref_voltage) ** 2)) < 1e-4

        noinput_voltage = sim.get_mainsim_voltage_trace(QUICK_SCX_CELL)
        assert np.sqrt(np.mean((voltage - noinput_voltage) ** 2)) > 1e-2

    def test_add_holding_current_with_hyperpolarizing_input_raises(self):
        sim = CircuitSimulation(QUICK_SCX_DIR / "simulation_config_hypamp.json")
        with pytest.raises(BluecellulabError, match="injected twice"):
            sim.instantiate_gids(QUICK_SCX_CELL, add_stimuli=True, add_holding_current=True)
