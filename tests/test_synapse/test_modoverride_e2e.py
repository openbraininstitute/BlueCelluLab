"""End-to-end ``modoverride`` tests through CircuitSimulation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bluecellulab import CircuitSimulation
from bluecellulab.circuit import CellId, SynapseProperty

script_dir = Path(__file__).parent.parent
projections_sim = script_dir / "examples" / "sonata_unit_test_sims" / "projections"


def _sim_config_with_overrides(tmp_path: Path, overrides: list[dict]) -> Path:
    """Copy the projections sim config with absolute paths, ``overrides`` as
    its only connection_overrides and an extra ``Pre0`` node set (node 0)."""
    config = json.loads((projections_sim / "simulation_config.json").read_text())
    config["manifest"]["$CIRCUIT_DIR"] = str(
        script_dir / "examples" / "circuit_hipp_mooc_most_central_10_SP_PC")
    config["network"] = str(projections_sim / "circuit_config.json")
    node_sets = json.loads((projections_sim / "node_sets.json").read_text())
    node_sets["Pre0"] = {"population": "hippocampus_neurons", "node_id": [0]}
    (tmp_path / "node_sets.json").write_text(json.dumps(node_sets))
    config["node_sets_file"] = str(tmp_path / "node_sets.json")
    config["output"]["output_dir"] = str(tmp_path / "output")
    config["connection_overrides"] = overrides
    path = tmp_path / "simulation_config.json"
    path.write_text(json.dumps(config))
    return path


def test_modoverride_connection_blocks_build_helper_classes(tmp_path):
    """SONATA ``connection_overrides`` drive the factory: source/target
    node-set filtering, last matching block wins, delayed blocks do not
    override, unmatched (projection) synapses stay native, and info_dict
    works on the cell."""
    config = _sim_config_with_overrides(tmp_path, [
        {"name": "All", "source": "Mosaic", "target": "Mosaic",
         "modoverride": "AMPANMDA"},
        {"name": "FromPre0", "source": "Pre0", "target": "Mosaic",
         "modoverride": "GABAAB"},
        # Would raise (no such helper) if applied to cell 4:
        {"name": "ToPre0", "source": "Mosaic", "target": "Pre0",
         "modoverride": "NotAHelperPrefix"},
        {"name": "Delayed", "source": "Mosaic", "target": "Mosaic",
         "delay": 5.0, "weight": 0.5, "modoverride": "NotAHelperPrefix"},
    ])
    sim = CircuitSimulation(config)
    cell_id = CellId("hippocampus_neurons", 4)
    sim.instantiate_gids([cell_id], add_synapses=True, add_projections=True)
    cell = sim.cells[cell_id]

    local = "hippocampus_neurons__hippocampus_neurons__chemical"
    mechs = Counter(
        (synapse.syn_id.projection == local,
         int(synapse.syn_description[SynapseProperty.PRE_GID]),
         synapse.mech_name)
        for synapse in cell.synapses.values()
    )
    local_mechs = {key[1:]: n for key, n in mechs.items() if key[0]}
    assert local_mechs == {(0, "GABAAB"): 1, (6, "AMPANMDA"): 4}
    assert {key[2] for key in mechs if not key[0]} == {"ProbAMPANMDA_EMS"}

    info = cell.info_dict
    assert len(info["synapses"]) == len(cell.synapses)


# ProbFilt sources are not distributed with BlueCelluLab: point this variable
# at a directory with ProbFiltAMPANMDA_EMS.mod and its helper HOC (e.g. a
# sonata_simplify output ``mod`` directory) to run the test below.
PROBFILT_DIR_ENV = "BLUECELLULAB_PROBFILT_DIR"
_BUNDLED_HELPER = (
    Path(__file__).parents[2] / "bluecellulab" / "hoc" / "AMPANMDAHelper.hoc"
)


@pytest.fixture
def stub_helper_dir(tmp_path):
    """Circuit helper dir with ``StubFiltHelper``: the bundled AMPANMDA
    helper renamed, declaring two extra mandatory fields.

    Runs everywhere (uses the test ``ProbAMPANMDA_EMS`` mechanism) and
    exercises the same path as a ProbFilt override.
    """
    text = _BUNDLED_HELPER.read_text().replace("AMPANMDAHelper", "StubFiltHelper")
    header = 'strdef StubFiltHelper_NeededAttributes\nStubFiltHelper_NeededAttributes = "w_corr;tau_corr"\n'
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / "StubFiltHelper.hoc").write_text(header + text)
    return helper_dir


@pytest.fixture
def probfilt_helper_dir(tmp_path):  # pragma: no cover - needs non-distributable ProbFilt sources
    """Compile ProbFiltAMPANMDA_EMS in a tmp dir, load it, return the helper
    dir; skip when the sources or the compiler are unavailable."""
    import neuron

    source = os.environ.get(PROBFILT_DIR_ENV)
    if not source:
        pytest.skip(f"{PROBFILT_DIR_ENV} not set")
    mod = Path(source) / "ProbFiltAMPANMDA_EMS.mod"
    helper = Path(source) / "ProbFiltAMPANMDA_EMSHelper.hoc"
    if not (mod.is_file() and helper.is_file()):
        pytest.skip(f"ProbFilt mod or helper missing in {source}")
    nrnivmodl = shutil.which("nrnivmodl") or str(Path(sys.executable).with_name("nrnivmodl"))
    (tmp_path / "mods").mkdir()
    shutil.copy(mod, tmp_path / "mods")
    try:
        result = subprocess.run(
            [nrnivmodl, "mods"], cwd=tmp_path, capture_output=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"cannot run nrnivmodl: {exc}")
    libs = sorted(tmp_path.glob("*/libnrnmech.*")) + sorted(tmp_path.glob("*/.libs/libnrnmech.*"))
    if result.returncode != 0 or not libs:
        pytest.skip("ProbFilt mechanism did not compile")
    neuron.h.nrn_load_dll(str(libs[0]))
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    shutil.copy(helper, helper_dir)
    return helper_dir


def _run_helper_end_to_end(helper_dir: Path, prefix: str, mechanism: str,
                           rng_object: bool, extra: dict,
                           set_extra_on_mechanism: bool = True) -> None:
    """Build an override synapse from ``helper_dir``, run it next to two
    hand-built references (same / different neurodamus seed) and compare."""
    import neuron

    from bluecellulab.rngsettings import RNGSettings
    from bluecellulab.synapse.synapse_types import GenericSpikeSynapse, SynapseHocArgs

    rng_settings = RNGSettings.get_instance()
    rng_settings.set_seeds(mode="Random123", base_seed=0)
    rng_settings.synapse_seed = 7
    post_gid, sid, popids = 3, 4, (1, 2)
    description = pd.Series({
        SynapseProperty.PRE_GID: 1, SynapseProperty.G_SYNX: 0.7,
        SynapseProperty.U_SYN: 0.5, SynapseProperty.D_SYN: 1.0,
        SynapseProperty.F_SYN: 10.0, SynapseProperty.DTC: 1.7,
        SynapseProperty.TYPE: 113, SynapseProperty.NRRP: 1, **extra,
    })

    sections = []  # keep the sections alive for the whole run

    def section():
        sec = neuron.h.Section()
        sec.insert("pas")
        sections.append(sec)
        return sec

    synapse = GenericSpikeSynapse(
        SimpleNamespace(id=post_gid), SynapseHocArgs(0.5, section()), ("", sid),
        description, popids, post_gid, None, prefix, helper_dirs=(str(helper_dir),),
    )
    assert synapse.helper_path == str(helper_dir / f"{prefix}Helper.hoc")
    assert synapse.is_inhibitory is False

    # References built by hand with the neurodamus seeds (tgid = post_gid + 1);
    # the control uses another synapse seed and must differ.
    references, rngs = [], []
    for synapse_seed in (7, 8):
        point_process = getattr(neuron.h, mechanism)(0.5, sec=section())
        point_process.synapseID = sid
        for name, value in (("tau_d_AMPA", 1.7), ("Use", 0.5), ("Dep", 1.0),
                            ("Fac", 10.0), ("Nrrp", 1),
                            *(extra.items() if set_extra_on_mechanism else ())):
            setattr(point_process, name, value)
        seeds = (post_gid + 1 + 250, sid + 100,
                 popids[0] * 65536 + popids[1] + synapse_seed + 300)
        if rng_object:  # pragma: no cover - object-only setRNG (ProbFilt)
            rng = neuron.h.Random()
            rng.Random123(*seeds)
            rng.uniform(0, 1)
            point_process.setRNG(rng)
            rngs.append(rng)
        else:
            point_process.setRNG(*seeds)
        references.append(point_process)

    stim = neuron.h.NetStim()
    stim.number, stim.interval, stim.start, stim.noise = 20, 2.0, 1.0, 0
    netcons, traces = [], []
    for point_process in (synapse.hsynapse, *references):
        netcon = neuron.h.NetCon(stim, point_process)
        netcon.weight[0] = 0.7
        netcons.append(netcon)
        traces.append(neuron.h.Vector().record(point_process._ref_g))
    neuron.h.finitialize(-65)
    neuron.h.continuerun(50)

    helper_trace, reference_trace, control_trace = (list(t) for t in traces)
    assert max(helper_trace) > 0  # it fires
    assert helper_trace == reference_trace  # same stream as neurodamus
    assert max(abs(a - b) for a, b in zip(helper_trace, control_trace)) > 0.01 * max(helper_trace)


def test_circuit_helper_end_to_end_real_neuron(stub_helper_dir):
    """A circuit-dir helper with extra mandatory fields fires and draws the
    neurodamus Random123 stream (runs in CI, unlike the ProbFilt test)."""
    _run_helper_end_to_end(stub_helper_dir, "StubFilt", "ProbAMPANMDA_EMS",
                           rng_object=False, extra={"w_corr": 0.5, "tau_corr": 5.0},
                           set_extra_on_mechanism=False)


def test_circuit_helper_missing_extra_field_raises(stub_helper_dir):
    """The stub helper's declared fields are mandatory at build time."""
    from bluecellulab.exceptions import BluecellulabError

    with pytest.raises(BluecellulabError, match="w_corr"):
        _run_helper_end_to_end(stub_helper_dir, "StubFilt", "ProbAMPANMDA_EMS",
                               rng_object=False, extra={"tau_corr": 5.0},
                               set_extra_on_mechanism=False)


def test_probfilt_helper_end_to_end_real_neuron(probfilt_helper_dir):  # pragma: no cover
    """A ProbFilt override synapse built from a circuit helper dir fires and
    draws the same Random123 stream as the neurodamus seed formula."""
    _run_helper_end_to_end(probfilt_helper_dir, "ProbFiltAMPANMDA_EMS",
                           "ProbFiltAMPANMDA_EMS", rng_object=True,
                           extra={"w_corr": 0.5, "tau_corr": 5.0})
