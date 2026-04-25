"""Tests for parsing of gap_junctions / continuous_connections in
SonataSimulationConfig and the relaxed `mod_override` field on
ConnectionOverrides."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from bluecellulab.circuit.config import SonataSimulationConfig
from bluecellulab.circuit.config.sections import ConnectionOverrides


parent_dir = Path(__file__).resolve().parent.parent
base_conf_path = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "condition_parameters"
    / "simulation_config.json"
)


def _make_conf_with_blocks(tmp_path: Path, extra: dict) -> Path:
    src_dir = base_conf_path.parent
    dst_dir = tmp_path / "sim"
    shutil.copytree(src_dir, dst_dir)
    cfg_path = dst_dir / "simulation_config.json"
    with cfg_path.open() as fh:
        data = json.load(fh)
    data.update(extra)
    with cfg_path.open("w") as fh:
        json.dump(data, fh)
    return cfg_path


def test_gap_junctions_parsed(tmp_path):
    """gap_junctions entries are per-population overrides for auto-detected
    electrical edge populations."""
    cfg_path = _make_conf_with_blocks(
        tmp_path,
        {
            "gap_junctions": [
                {
                    "edge_population": "elec",
                    "mod": "gapjunction",
                    "pointer_name": "vpre",
                    "weight_factor": 0.5,
                }
            ]
        },
    )
    sim = SonataSimulationConfig(cfg_path)
    blocks = sim.gap_junctions()
    assert len(blocks) == 1
    assert blocks[0].edge_population == "elec"
    assert blocks[0].mod == "gapjunction"
    assert blocks[0].pointer_name == "vpre"
    assert blocks[0].weight_factor == 0.5


def test_continuous_connections_parsed(tmp_path):
    cfg_path = _make_conf_with_blocks(
        tmp_path,
        {
            "continuous_connections": [
                {
                    "name": "exc",
                    "edge_population": "chem",
                    "source": "src_set",
                    "target": "All",
                    "mod": "neuron_to_neuron_exc_syn",
                    "pointer_name": "vpre",
                }
            ]
        },
    )
    sim = SonataSimulationConfig(cfg_path)
    blocks = sim.continuous_connections()
    assert len(blocks) == 1
    assert blocks[0].name == "exc"
    assert blocks[0].edge_population == "chem"
    assert blocks[0].source == "src_set"


def test_no_blocks_returns_empty(tmp_path):
    cfg_path = _make_conf_with_blocks(tmp_path, {})
    sim = SonataSimulationConfig(cfg_path)
    assert sim.gap_junctions() == []
    assert sim.continuous_connections() == []


def test_mod_override_accepts_arbitrary_prefix():
    """mod_override is a helper prefix (resolved to ``<prefix>Helper`` when
    the synapse is built); any non-empty string is accepted at parse time.

    Previously it was restricted to Literal["GluSynapse"].
    """
    co = ConnectionOverrides(
        source="A", target="B", mod_override="MySynapse"
    )
    assert co.mod_override == "MySynapse"


def test_mod_override_rejects_empty_prefix():
    with pytest.raises(Exception):
        ConnectionOverrides(
            source="A", target="B", mod_override="   "
        )
