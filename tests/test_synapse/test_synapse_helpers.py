"""Tests for the neurodamus-style mod_override / helper-HOC machinery."""

from __future__ import annotations

from types import SimpleNamespace

import importlib_resources as resources
import pandas as pd
import pytest

from bluecellulab.circuit.config.sections import ConnectionOverrides
from bluecellulab.exceptions import BluecellulabError, ConfigError
from bluecellulab.circuit.synapse_properties import SynapseProperty
from bluecellulab.synapse import synapse_factory, synapse_helpers, synapse_types
from bluecellulab.synapse.synapse_types import (
    GenericSpikeSynapse,
    SynapseHocArgs,
    SynapseID,
)
from bluecellulab.synapse.synapse_helpers import build_helper_params


@pytest.mark.parametrize("prefix", ["AMPANMDA", "GABAAB", "ProbFilt5AMPANMDA_EMS"])
def test_mod_override_accepts_helper_prefix_before_mechanisms_load(prefix):
    """Helper prefixes are not NEURON mechanisms and mechanisms may load
    later: the config must not query NEURON."""
    co = ConnectionOverrides(source="A", target="B", mod_override=prefix)
    assert co.mod_override == prefix


@pytest.mark.parametrize("value", ["", "   "])
def test_mod_override_rejects_empty(value):
    with pytest.raises(ConfigError, match="non-empty helper prefix"):
        ConnectionOverrides(source="A", target="B", mod_override=value)


def test_bundled_helper_files_are_package_resources():
    for suffix in ("AMPANMDA", "Exp2Syn", "GABAAB", "GluSynapse"):
        helper = resources.files("bluecellulab").joinpath(
            "hoc", f"{suffix}Helper.hoc"
        )
        assert helper.is_file()


def test_bundled_hoc_directory_is_appended_to_search_path(monkeypatch):
    monkeypatch.delenv("HOC_LIBRARY_PATH", raising=False)

    bundled_dir = synapse_helpers._ensure_bundled_hoc_directory_on_search_path()

    assert bundled_dir in synapse_helpers.os.environ["HOC_LIBRARY_PATH"].split(
        synapse_helpers.os.pathsep
    )


@pytest.fixture
def clean_helper_search_dirs():
    """Isolate the registered helper search dirs around a test."""
    synapse_helpers.clear_helper_search_dirs()
    yield
    synapse_helpers.clear_helper_search_dirs()


@pytest.fixture
def helper_env(tmp_path, monkeypatch, clean_helper_search_dirs):
    """Isolated cwd / HOC_LIBRARY_PATH / registered dirs / fake bundled dir."""
    dirs = SimpleNamespace(
        cwd=tmp_path / "cwd",
        user=tmp_path / "user",
        registered=tmp_path / "registered",
        bundled=tmp_path / "bundled",
    )
    for directory in vars(dirs).values():
        directory.mkdir()
    monkeypatch.chdir(dirs.cwd)
    monkeypatch.delenv("HOC_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(
        synapse_helpers, "_bundled_hoc_directory", lambda: str(dirs.bundled)
    )
    return dirs


def _write_helper(directory, suffix, marker):
    """Write a minimal helper defining ``<suffix>Helper`` and a marker
    global."""
    (directory / f"{suffix}Helper.hoc").write_text(
        f"{marker}_{suffix} = 1\n"
        f"begintemplate {suffix}Helper\n"
        "public synapse\n"
        "objref synapse\n"
        "proc init() {}\n"
        f"endtemplate {suffix}Helper\n"
    )


def _loaded_marker(suffix):
    import neuron

    markers = [m for m in ("cwd", "user", "registered", "bundled")
               if hasattr(neuron.h, f"{m}_{suffix}")]
    assert len(markers) == 1, markers
    return markers[0]


def test_helper_search_dirs_precedence(helper_env, monkeypatch):
    """Cwd -> user HOC_LIBRARY_PATH (bundled excluded) -> registered ->
    bundled."""
    monkeypatch.setenv(
        "HOC_LIBRARY_PATH",
        synapse_helpers.os.pathsep.join([str(helper_env.user), str(helper_env.bundled)]),
    )
    synapse_helpers.register_helper_search_dirs([helper_env.registered])

    assert synapse_helpers._helper_search_dirs() == [
        str(helper_env.cwd),
        str(helper_env.user),
        str(helper_env.registered),
        str(helper_env.bundled),
    ]


def test_cwd_helper_beats_user_hoc_library_path_real_neuron(helper_env, monkeypatch):
    suffix = "CwdWinsRealNrn"
    for marker in ("cwd", "user", "registered", "bundled"):
        _write_helper(getattr(helper_env, marker), suffix, marker)
    monkeypatch.setenv("HOC_LIBRARY_PATH", str(helper_env.user))
    synapse_helpers.register_helper_search_dirs([helper_env.registered])
    try:
        synapse_helpers.load_synapse_helper(suffix)
        assert _loaded_marker(suffix) == "cwd"
        assert synapse_helpers._loaded_helpers[suffix] == str(
            helper_env.cwd / f"{suffix}Helper.hoc")
    finally:
        synapse_helpers._loaded_helpers.pop(suffix, None)


def test_user_hoc_library_path_beats_registered_dir_real_neuron(helper_env, monkeypatch):
    suffix = "UserWinsRealNrn"
    for marker in ("user", "registered", "bundled"):
        _write_helper(getattr(helper_env, marker), suffix, marker)
    monkeypatch.setenv("HOC_LIBRARY_PATH", str(helper_env.user))
    synapse_helpers.register_helper_search_dirs([helper_env.registered])
    try:
        synapse_helpers.load_synapse_helper(suffix)
        assert _loaded_marker(suffix) == "user"
    finally:
        synapse_helpers._loaded_helpers.pop(suffix, None)


def test_registered_dir_beats_bundled_real_neuron(helper_env):
    """No user HOC_LIBRARY_PATH: the bundled dir is appended to the path for
    dependencies, but a registered circuit dir still wins."""
    suffix = "RegisteredWinsRealNrn"
    for marker in ("registered", "bundled"):
        _write_helper(getattr(helper_env, marker), suffix, marker)
    synapse_helpers.register_helper_search_dirs([helper_env.registered])
    try:
        synapse_helpers.load_synapse_helper(suffix)
        assert _loaded_marker(suffix) == "registered"
        assert str(helper_env.bundled) in synapse_helpers.os.environ["HOC_LIBRARY_PATH"]
    finally:
        synapse_helpers._loaded_helpers.pop(suffix, None)


def test_bundled_helper_used_as_fallback_real_neuron(helper_env):
    suffix = "BundledFallbackRealNrn"
    _write_helper(helper_env.bundled, suffix, "bundled")
    try:
        synapse_helpers.load_synapse_helper(suffix)
        assert _loaded_marker(suffix) == "bundled"
    finally:
        synapse_helpers._loaded_helpers.pop(suffix, None)


def test_helper_loaded_once_without_redefinition(helper_env):
    """A second load (another population / cell, or after the cache is
    cleared) must not re-execute the HOC file: redefining a template is a HOC
    error."""
    import neuron

    suffix = "LoadOnceRealNrn"
    _write_helper(helper_env.bundled, suffix, "bundled")
    try:
        assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
        assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
        # Template already defined in NEURON: skipped even with a cold cache
        # and a different file of the same name earlier on the search path.
        synapse_helpers._loaded_helpers.pop(suffix)
        _write_helper(helper_env.cwd, suffix, "cwd")
        assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
        assert synapse_helpers._loaded_helpers[suffix] == "<preloaded>"
        assert not hasattr(neuron.h, f"cwd_{suffix}")
    finally:
        synapse_helpers._loaded_helpers.pop(suffix, None)


def _real_section():
    import neuron

    section = neuron.h.Section(name="helper_test_section")
    section.insert("pas")
    return section


_TM_DESCRIPTION = {
    SynapseProperty.PRE_GID: 1,
    SynapseProperty.G_SYNX: 0.7,
    SynapseProperty.U_SYN: 0.5,
    SynapseProperty.D_SYN: 600.0,
    SynapseProperty.F_SYN: 20.0,
    SynapseProperty.DTC: 1.7,
    SynapseProperty.NRRP: 2,
}


@pytest.mark.parametrize(
    "suffix, mechanism", [("AMPANMDA", "ProbAMPANMDA_EMS"), ("GABAAB", "ProbGABAAB_EMS")]
)
def test_bundled_helpers_build_real_neuron(suffix, mechanism, clean_helper_search_dirs):
    """``AMPANMDA``/``GABAAB`` overrides build from the bundled helpers."""
    section = _real_section()
    synapse = GenericSpikeSynapse(
        SimpleNamespace(id=3), SynapseHocArgs(0.5, section), ("", 4),
        pd.Series(dict(_TM_DESCRIPTION)), (0, 0), 3, None, suffix,
    )

    assert synapse.hsynapse.hname().startswith(mechanism)
    assert synapse.hsynapse.Dep == pytest.approx(600.0)
    assert synapse.hsynapse.Nrrp == pytest.approx(2)
    assert synapse.hsynapse.synapseID == 4
    assert synapse.hsynapse.conductance == pytest.approx(0.7)  # = weight


def test_mechanism_name_override_has_no_alias(helper_env):
    """``ProbAMPANMDA_EMS`` is a mechanism, not a helper prefix: no alias to
    ``AMPANMDAHelper``, so a clear missing-helper error is raised."""
    with pytest.raises(FileNotFoundError, match="ProbAMPANMDA_EMSHelper.hoc.*helper prefix"):
        synapse_helpers.load_synapse_helper("ProbAMPANMDA_EMS")


def test_register_helper_search_dirs_dedupes_and_ignores_missing(
    tmp_path, clean_helper_search_dirs
):
    existing = tmp_path / "exists"
    existing.mkdir()
    missing = tmp_path / "does_not_exist"

    synapse_helpers.register_helper_search_dirs(
        [existing, existing, missing, str(existing)]
    )

    assert synapse_helpers._extra_search_dirs == [str(existing)]


def test_missing_helper_error_lists_registered_dirs(helper_env):
    synapse_helpers.register_helper_search_dirs([helper_env.registered])

    with pytest.raises(FileNotFoundError, match=str(helper_env.registered)):
        synapse_helpers.load_synapse_helper("MissingRegisteredDirCoverage")


def test_load_synapse_helper_missing_raises():
    """load_synapse_helper raises FileNotFoundError when the helper HOC cannot
    be located."""
    from bluecellulab.synapse.synapse_helpers import load_synapse_helper

    with pytest.raises(FileNotFoundError):
        load_synapse_helper("ThisSuffixDoesNotExistAnywhere_XYZ")


def test_load_synapse_helper_reports_neuron_load_failure(helper_env, monkeypatch):
    suffix = "LoadFailureCoverage"
    _write_helper(helper_env.bundled, suffix, "bundled")
    fake_h = SimpleNamespace(load_file=lambda _: 0)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    with pytest.raises(FileNotFoundError, match="NEURON failed to load"):
        synapse_helpers.load_synapse_helper(suffix)


def test_from_sonata_reads_modoverride_one_word():
    """from_sonata must read the SONATA key 'modoverride' (one word, no
    underscore), matching the SONATA spec and libsonata.

    Previously this used 'mod_override' (underscore) which never matched
    the actual JSON key, so modoverride was silently ignored.
    """
    conn_entry = {
        "source": "Excitatory",
        "target": "Mosaic",
        "modoverride": "IClamp",
    }
    co = ConnectionOverrides.from_sonata(conn_entry)
    assert co.mod_override == "IClamp"


def test_from_sonata_modoverride_none_when_absent():
    """from_sonata should return None for mod_override when the key is not
    present in the SONATA connection override entry."""
    conn_entry = {
        "source": "Excitatory",
        "target": "Mosaic",
    }
    co = ConnectionOverrides.from_sonata(conn_entry)
    assert co.mod_override is None


def test_load_synapse_helper_uses_cache(monkeypatch):
    suffix = "CachedHelperCoverage"
    synapse_helpers._loaded_helpers[suffix] = "/some/path"
    fake_h = SimpleNamespace(load_file=pytest.fail)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    synapse_helpers._loaded_helpers.pop(suffix, None)


def test_load_synapse_helper_rejects_helper_without_template(helper_env, monkeypatch):
    suffix = "MissingTemplateCoverage"
    (helper_env.bundled / f"{suffix}Helper.hoc").write_text("// no template\n")
    fake_h = SimpleNamespace(load_file=lambda _: 1)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    with pytest.raises(AttributeError, match="did not define template"):
        synapse_helpers.load_synapse_helper(suffix)


def test_load_synapse_helper_skips_preloaded_template(monkeypatch):
    suffix = "LoadedTemplateCoverage"
    fake_h = SimpleNamespace(load_file=pytest.fail, LoadedTemplateCoverageHelper=object())
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    assert synapse_helpers.helper_available(suffix)
    synapse_helpers._loaded_helpers.pop(suffix, None)


def test_helper_params_use_neurodamus_names():
    params = build_helper_params(
        pd.Series({
            SynapseProperty.PRE_GID: 12,
            SynapseProperty.AXONAL_DELAY: 1.5,
            SynapseProperty.POST_SECTION_ID: 3,
            SynapseProperty.AFFERENT_SECTION_POS: 0.4,
            SynapseProperty.G_SYNX: 0.5,
            SynapseProperty.U_SYN: 0.3,
            SynapseProperty.D_SYN: 600.0,
            SynapseProperty.F_SYN: 20.0,
            SynapseProperty.DTC: 1.7,
            SynapseProperty.TYPE: 113,
            SynapseProperty.NRRP: 2,
            "custom_parameter": 3,
        }),
        ["custom_parameter"],
    )

    assert (params.sgid, params.delay, params.weight) == (12, 1.5, 0.5)
    assert (params.U, params.D, params.F, params.DTC) == (0.3, 600.0, 20.0, 1.7)
    assert (params.synType, params.nrrp) == (113, 2)
    assert (params.isec, params.ipt, params.offset) == (3, -1, 0.4)
    assert params.custom_parameter == 3
    assert not hasattr(params, "Nrrp")


def test_helper_params_defaults_optional_and_reserved_fields():
    """Optional fields get neurodamus defaults; NaN (outer join) counts as
    absent."""
    params = build_helper_params(
        pd.Series({SynapseProperty.CONDUCTANCE_RATIO: float("nan")}), [])

    assert params.maskValue == -1.0
    assert params.location == 0.5
    assert params.u_hill_coefficient == 0.0
    assert params.conductance_ratio == -1.0
    assert params.nrrp == -1.0


def test_helper_params_ignore_mask_value_from_edges():
    """maskValue is reserved: an edge value is never passed to the helper."""
    params = build_helper_params(pd.Series({"maskValue": 5.0}), ["maskValue"])

    assert params.maskValue == -1.0


def test_helper_params_pass_extra_fields_under_raw_name():
    """A helper declaring a standard SONATA name gets it unmapped and
    unscaled, while the neurodamus name keeps the mapped value."""
    params = build_helper_params(
        pd.Series({
            SynapseProperty.G_SYNX: 0.5,
            SynapseProperty.U_SYN: 0.3,
            "conductance": 0.5,
            "u_syn": 0.6,
        }),
        ["conductance", "u_syn"],
    )

    assert params.conductance == 0.5
    assert params.u_syn == 0.6
    assert params.U == 0.3


def test_generic_spike_synapse_scales_u_syn():
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.extracellular_calcium = 2.0
    description = pd.Series({
        SynapseProperty.U_HILL_COEFFICIENT: 1.0,
        SynapseProperty.U_SYN: 2.0,
    })

    result = synapse.update_syn_description(description)

    assert result[SynapseProperty.U_SYN] == 2.0 * result["u_scale_factor"]
    assert result["u_scale_factor"] == synapse.calc_u_scale_factor(1.0, 2.0)


def test_generic_spike_synapse_initializes_and_builds(monkeypatch):
    monkeypatch.setattr(GenericSpikeSynapse, "_build_via_helper", lambda self, _: None)
    cell_id = SimpleNamespace(id=21)
    description = pd.Series({SynapseProperty.PRE_GID: 4})

    synapse = GenericSpikeSynapse(
        cell_id,
        SynapseHocArgs(0.5, None),
        ("projection", 7),
        description,
        (2, 3),
        21,
        None,
        "Custom",
    )

    assert synapse.post_gid == 21
    assert synapse.mech_name == "not-yet-defined"


def test_generic_spike_synapse_update_removes_invalid_optional_values():
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.extracellular_calcium = None
    description = pd.Series({SynapseProperty.NRRP: "invalid", SynapseProperty.U_SYN: 2.0})

    result = synapse.update_syn_description(description)

    assert SynapseProperty.NRRP not in result
    assert result["u_scale_factor"] == 1.0
    assert result[SynapseProperty.U_SYN] == 2.0


def test_generic_spike_synapse_builds_from_helper(monkeypatch):
    active_section = {}

    class Section:
        def push(self):
            active_section["section"] = self

    class Helper:
        def __init__(self, *args):
            self.args = args
            self.created_in_section = active_section.get("section")
            self.synapse = "point-process"

    monkeypatch.setattr(synapse_helpers, "load_synapse_helper", lambda _: "TestHelper")
    monkeypatch.setattr(
        synapse_types.neuron,
        "h",
        SimpleNamespace(
            TestHelper=Helper,
            pop_section=lambda: active_section.pop("section", None),
        ),
    )
    section = Section()
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.post_gid = 41
    synapse.hoc_args = SimpleNamespace(location=0.25, section=section)
    synapse.syn_id = SynapseID("projection", 7)
    synapse.source_popid = 2
    synapse.target_popid = 3
    synapse.syn_description = pd.Series({SynapseProperty.G_SYNX: 0.9})
    synapse.persistent = []

    synapse._build_via_helper("Test")

    assert synapse.hsynapse == "point-process"
    assert synapse.mech_name == "Test"
    assert synapse.persistent[0].args[0] == 42
    assert synapse.persistent[0].created_in_section is section
    assert active_section == {}


def test_factory_uses_generic_synapse_for_mod_override(monkeypatch):
    created = SimpleNamespace()
    monkeypatch.setattr(synapse_factory.SynapseFactory, "determine_synapse_location", lambda *_: "location")
    monkeypatch.setattr(synapse_factory, "GenericSpikeSynapse", lambda *args, **kwargs: created)
    monkeypatch.setattr(
        synapse_factory.SynapseFactory,
        "apply_connection_modifiers",
        lambda modifiers, synapse: synapse,
    )
    cell = SimpleNamespace(cell_id="cell", post_gid=12)

    result = synapse_factory.SynapseFactory.create_synapse(
        cell,
        ("projection", 1),
        pd.Series(),
        SimpleNamespace(),
        (2, 3),
        None,
        {"ModOverride": "CustomMechanism"},
    )

    assert result is created


@pytest.mark.parametrize("mod_override", ["GluSynapse", "Exp2Syn"])
def test_factory_keeps_native_classes_for_glusynapse_and_exp2syn(monkeypatch, mod_override):
    """R6 option B: these values keep the native data-driven classes."""
    native = SimpleNamespace()
    monkeypatch.setattr(synapse_factory.SynapseFactory, "determine_synapse_location", lambda *_: "location")
    monkeypatch.setattr(synapse_factory, "GenericSpikeSynapse", pytest.fail)
    monkeypatch.setattr(
        synapse_factory.SynapseFactory, "determine_synapse_type",
        lambda _: synapse_factory.SynapseType.ALLEN_CHEMICAL,
    )
    monkeypatch.setattr(synapse_factory, "Exp2Syn", lambda *args, **kwargs: native)
    monkeypatch.setattr(
        synapse_factory.SynapseFactory, "apply_connection_modifiers",
        lambda modifiers, synapse: synapse,
    )
    cell = SimpleNamespace(cell_id="cell", post_gid=12)

    result = synapse_factory.SynapseFactory.create_synapse(
        cell, ("projection", 1), pd.Series(), SimpleNamespace(), (2, 3), None,
        {"ModOverride": mod_override},
    )

    assert result is native


def test_generic_spike_synapse_rejects_helper_without_synapse(monkeypatch):
    class Section:
        def push(self):
            pass

    class Helper:
        def __init__(self, *args):
            pass

    monkeypatch.setattr(synapse_helpers, "load_synapse_helper", lambda _: "TestHelper")
    monkeypatch.setattr(
        synapse_types.neuron,
        "h",
        SimpleNamespace(TestHelper=Helper, pop_section=lambda: None),
    )
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.post_gid = 41
    synapse.hoc_args = SimpleNamespace(location=0.25, section=Section())
    synapse.syn_id = SynapseID("projection", 7)
    synapse.source_popid = 2
    synapse.target_popid = 3
    synapse.syn_description = pd.Series()
    synapse.persistent = []

    with pytest.raises(AttributeError, match="does not expose"):
        synapse._build_via_helper("Test")


def _helper_synapse(monkeypatch, description, needed):
    """GenericSpikeSynapse wired to a fake helper declaring ``needed``."""
    calls = []

    class Section:
        def push(self):
            pass

    class Helper:
        def __init__(self, *args):
            calls.append(args)
            self.synapse = SimpleNamespace()

    monkeypatch.setattr(synapse_helpers, "load_synapse_helper", lambda _: "TestHelper")
    monkeypatch.setattr(
        synapse_types.neuron,
        "h",
        SimpleNamespace(
            TestHelper=Helper,
            TestHelper_NeededAttributes=needed,
            pop_section=lambda: None,
        ),
    )
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.post_gid = 41
    synapse.hoc_args = SimpleNamespace(location=0.25, section=Section())
    synapse.syn_id = SynapseID("projection", 7)
    synapse.source_popid = 2
    synapse.target_popid = 3
    synapse.syn_description = pd.Series(description)
    synapse.persistent = []
    return synapse, calls


@pytest.mark.parametrize(
    "description",
    [{"w_corr": 0.1}, {"w_corr": 0.1, "tau_corr": float("nan")}],
    ids=["absent", "nan"],
)
def test_missing_needed_attribute_raises_before_helper(monkeypatch, description):
    """``_NeededAttributes`` are mandatory: absent (or NaN from the outer
    join of populations) raises, naming helper, synapse and fields."""
    synapse, calls = _helper_synapse(monkeypatch, description, "w_corr;tau_corr")

    with pytest.raises(BluecellulabError) as excinfo:
        synapse._build_via_helper("Test")

    message = str(excinfo.value)
    assert "TestHelper" in message
    assert "('projection', 7)" in message
    assert "['tau_corr']" in message
    assert calls == []


def test_needed_attributes_present_builds(monkeypatch):
    synapse, calls = _helper_synapse(
        monkeypatch, {"w_corr": 0.1, "tau_corr": 2.0}, "w_corr;tau_corr;maskValue"
    )

    synapse._build_via_helper("Test")

    assert len(calls) == 1
    assert not hasattr(synapse.hsynapse, "conductance")  # not exposed: untouched


def test_conductance_set_to_weight_when_exposed(monkeypatch):
    synapse, _ = _helper_synapse(monkeypatch, {SynapseProperty.G_SYNX: 0.9}, "")
    helper_cls = synapse_types.neuron.h.TestHelper

    class Mechanism:
        conductance = 0.0

    class HelperWithConductance(helper_cls):
        def __init__(self, *args):
            super().__init__(*args)
            self.synapse = Mechanism()

    synapse_types.neuron.h.TestHelper = HelperWithConductance
    synapse._build_via_helper("Test")

    assert synapse.hsynapse.conductance == 0.9


def test_get_helper_needed_attributes_returns_declared_fields(monkeypatch):
    suffix = "NeededAttrsCoverage"
    fake_h = SimpleNamespace(
        load_file=lambda _: 1,
        NeededAttrsCoverageHelper=object(),
        NeededAttrsCoverageHelper_NeededAttributes="w_corr;tau_corr;w1_corr",
    )
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    attrs = synapse_helpers.get_helper_needed_attributes(suffix)
    assert attrs == ["w_corr", "tau_corr", "w1_corr"]
    synapse_helpers._loaded_helpers.pop(suffix, None)


def test_get_helper_needed_attributes_empty_when_no_metadata(monkeypatch):
    suffix = "NoAttrsCoverage"
    fake_h = SimpleNamespace(
        load_file=lambda _: 1,
        NoAttrsCoverageHelper=object(),
    )
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.get_helper_needed_attributes(suffix) == []
    synapse_helpers._loaded_helpers.pop(suffix, None)
