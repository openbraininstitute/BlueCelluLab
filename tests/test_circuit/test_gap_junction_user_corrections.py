"""Tests for the beta_features gap-junction user corrections ported from
neurodamus (``load_user_modifications``)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import neuron
import pytest

from bluecellulab.circuit.config import SonataSimulationConfig
from bluecellulab.circuit.gap_junction_user_corrections import (
    _deterministic_stoch,
    _perform_remove_channels,
    load_user_modifications,
)
from bluecellulab.circuit.node_id import CellId
from bluecellulab.exceptions import BluecellulabError


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


def _fake_sim(beta_features: dict, cells: dict) -> SimpleNamespace:
    return SimpleNamespace(
        cells=cells,
        pc=None,
        circuit_access=SimpleNamespace(
            config=SimpleNamespace(beta_features=lambda: beta_features)
        ),
    )


def _cell_with_section(name: str = "soma") -> SimpleNamespace:
    sec = neuron.h.Section(name=name)
    sec.insert("pas")
    return SimpleNamespace(cell=SimpleNamespace(all=[sec]), soma=sec)


def test_beta_features_parsed(tmp_path):
    cfg_path = _make_conf_with_blocks(
        tmp_path,
        {"beta_features": {"gjc": 0.2, "procedure_type": "validation_sim"}},
    )
    sim = SonataSimulationConfig(cfg_path)
    assert sim.beta_features() == {
        "gjc": 0.2,
        "procedure_type": "validation_sim",
    }


def test_beta_features_absent(tmp_path):
    cfg_path = _make_conf_with_blocks(tmp_path, {})
    sim = SonataSimulationConfig(cfg_path)
    assert sim.beta_features() == {}


def test_empty_settings_is_noop():
    sim = _fake_sim({}, {})
    holding, seclamp = load_user_modifications(sim)
    assert holding == {}
    assert seclamp == {}


def test_update_conductance_sets_all_gap_junctions():
    gj1, gj2 = SimpleNamespace(g=1.0), SimpleNamespace(g=1.0)
    cell = SimpleNamespace(gap_junctions=[gj1, gj2])
    sim = _fake_sim(
        {"gjc": 0.2, "procedure_type": "validation_sim"},
        {CellId("pop", 0): cell},
    )
    load_user_modifications(sim)
    assert gj1.g == pytest.approx(0.2)
    assert gj2.g == pytest.approx(0.2)


def test_update_conductance_uses_weight_fallback():
    gj = SimpleNamespace(weight=1.0)
    cell = SimpleNamespace(gap_junctions=[gj])
    sim = _fake_sim(
        {"gjc": 0.3, "procedure_type": "validation_sim"},
        {CellId("pop", 0): cell},
    )
    load_user_modifications(sim)
    assert gj.weight == pytest.approx(0.3)


def test_remove_channels_uninserts_mechanism():
    cell = _cell_with_section()
    sec = cell.cell.all[0]
    assert "pas" in dir(sec(0.5))
    sim = _fake_sim({}, {CellId("pop", 0): cell})
    _perform_remove_channels(sim, ["pas"])
    assert "g_pas" not in dir(sec(0.5))


def test_deterministic_stoch_noop_without_stochkv():
    cell = _cell_with_section()
    sim = _fake_sim({}, {CellId("pop", 0): cell})
    _deterministic_stoch(sim)  # must not raise


def test_manual_mecombo_info_adds_iclamp(tmp_path):
    cell = _cell_with_section()
    fname = tmp_path / "holding.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("holding_per_gid/0.5/a0", data=-0.07)
    sim = _fake_sim(
        {"gjc": 0.5, "manual_MEComboInfo_file": str(fname)},
        {CellId("pop", 0): cell},
    )
    holding, _ = load_user_modifications(sim)
    assert list(holding) == [0]
    assert holding[0].amp == pytest.approx(-0.07)
    assert holding[0].dur == 9e9


def test_find_holding_current_adds_seclamp(tmp_path):
    cell = _cell_with_section()
    fname = tmp_path / "v_per_gid.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("v_per_gid/a0", data=-70.0)
    sim = _fake_sim(
        {"procedure_type": "find_holding_current", "vc_amp": str(fname)},
        {CellId("pop", 0): cell},
    )
    _, seclamp = load_user_modifications(sim)
    assert list(seclamp) == [0]
    assert seclamp[0].amp1 == pytest.approx(-70.0)
    assert seclamp[0].dur1 == 9e9


def test_find_holding_current_rejects_mecombo_file(tmp_path):
    cell = _cell_with_section()
    sim = _fake_sim(
        {
            "procedure_type": "find_holding_current",
            "manual_MEComboInfo_file": str(tmp_path / "x.h5"),
            "vc_amp": str(tmp_path / "y.h5"),
        },
        {CellId("pop", 0): cell},
    )
    with pytest.raises(BluecellulabError):
        load_user_modifications(sim)


def test_update_gpas(tmp_path):
    sec = neuron.h.Section(name="cell.soma")
    sec.insert("pas")
    cell = SimpleNamespace(cell=SimpleNamespace(all=[sec]), soma=sec)
    seg_name = str(sec(0.5)).split(".", 1)[1]  # 'soma(0.5)'
    fname = tmp_path / "g_pas.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset(f"g_pas/0.5/a0/{seg_name}", data=[0.001, 0.002])
    sim = _fake_sim(
        {
            "gjc": 0.5,
            "load_g_pas_file": str(fname),
            "correction_iteration_load": 1,
        },
        {CellId("pop", 0): cell},
    )
    load_user_modifications(sim)
    assert sec(0.5).g_pas == pytest.approx(0.002)


def test_unknown_remove_channels_warns(caplog):
    cell = _cell_with_section()
    sim = _fake_sim(
        {"remove_channels": "bogus"}, {CellId("pop", 0): cell}
    )
    import logging

    with caplog.at_level(logging.WARNING):
        load_user_modifications(sim)
    assert "Unknown GJ remove_channels" in caplog.text


def test_corrections_gated_on_gapjunction_target_population():
    """As neurodamus, user corrections only run when
    ``beta_features["gapjunction_target_population"]`` names an instantiated
    node population."""
    from bluecellulab.circuit_simulation import CircuitSimulation

    gj = SimpleNamespace(g=1.0)
    cell = SimpleNamespace(gap_junctions=[gj])
    cells = {CellId("pop", 0): cell}
    beta = {"gjc": 0.2, "procedure_type": "validation_sim"}

    # Key absent -> no corrections (neurodamus gate).
    CircuitSimulation._apply_gap_junction_user_corrections(
        _fake_sim(beta, cells)
    )
    assert gj.g == 1.0

    # Names another population -> no corrections.
    CircuitSimulation._apply_gap_junction_user_corrections(
        _fake_sim({**beta, "gapjunction_target_population": "other"}, cells)
    )
    assert gj.g == 1.0

    # Names the instantiated population -> corrections applied.
    CircuitSimulation._apply_gap_junction_user_corrections(
        _fake_sim({**beta, "gapjunction_target_population": "pop"}, cells)
    )
    assert gj.g == pytest.approx(0.2)
