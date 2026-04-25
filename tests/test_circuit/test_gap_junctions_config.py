"""Unit tests for the gap-junction / continuous-connection config models."""
from __future__ import annotations

import pytest

from bluecellulab.circuit.config.gap_junctions import (
    ContinuousConnectionConfig,
    GapJunctionConfig,
)


def test_gap_junction_config_defaults():
    cfg = GapJunctionConfig(edge_population="elec")
    assert cfg.edge_population == "elec"
    assert cfg.node_set == "All"
    assert cfg.weight_factor == 1.0
    assert cfg.mod == "Gap"
    assert cfg.pointer_name == "vgap"
    assert cfg.symmetric is True


def test_gap_junction_config_overrides():
    cfg = GapJunctionConfig(
        edge_population="elec",
        node_set="my_nodes",
        weight_factor=0.5,
        mod="gapjunction",
        pointer_name="vpre",
        symmetric=False,
    )
    assert cfg.mod == "gapjunction"
    assert cfg.pointer_name == "vpre"
    assert cfg.symmetric is False


def test_gap_junction_config_extra_forbidden():
    with pytest.raises(Exception):
        GapJunctionConfig(edge_population="elec", unknown_field=1)


def test_continuous_connection_config_defaults():
    cfg = ContinuousConnectionConfig(name="exc", edge_population="chem")
    assert cfg.name == "exc"
    assert cfg.source == "All"
    assert cfg.target == "All"
    assert cfg.symmetric is False


def test_continuous_connection_rejects_symmetric():
    with pytest.raises(Exception):
        ContinuousConnectionConfig(
            name="x", edge_population="chem", symmetric=True
        )
