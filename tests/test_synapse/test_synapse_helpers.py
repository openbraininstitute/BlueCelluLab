"""Tests for the neurodamus-style mod_override / helper-HOC machinery."""

from __future__ import annotations

from types import SimpleNamespace

import importlib_resources as resources
import pandas as pd
import pytest

from bluecellulab.circuit.config.sections import ConnectionOverrides
from bluecellulab.circuit.synapse_properties import SynapseProperty
from bluecellulab.synapse import synapse_factory, synapse_helpers, synapse_types
from bluecellulab.synapse.synapse_types import (
    GenericSpikeSynapse,
    SynapseHocArgs,
    SynapseID,
    _SynParamsAdapter,
)


def test_mod_override_accepts_arbitrary_existing_mech():
    """ConnectionOverrides.mod_override only requires the SUFFIX to exist in
    NEURON; previously it was restricted to ``Literal["GluSynapse"]``."""
    co = ConnectionOverrides(
        source="A",
        target="B",
        mod_override="IClamp",
    )
    assert co.mod_override == "IClamp"


def test_mod_override_rejects_unknown_mech():
    with pytest.raises(Exception):
        ConnectionOverrides(
            source="A",
            target="B",
            mod_override="DefinitelyNotAMech_XYZ",
        )


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


def test_load_synapse_helper_prefers_external_helper(monkeypatch):
    suffix = "ExternalPrecedenceCoverage"
    calls = []

    class FakeH:
        ExternalPrecedenceCoverageHelper = object()

        def load_file(self, filename):
            calls.append(filename)
            return 1

    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=FakeH()))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    assert calls == [f"{suffix}Helper.hoc"]
    synapse_helpers._loaded_helpers.discard(suffix)


def test_load_synapse_helper_falls_back_to_bundled_path(monkeypatch):
    suffix = "BundledFallbackCoverage"
    calls = []

    class FakeH:
        BundledFallbackCoverageHelper = object()

        def load_file(self, filename):
            calls.append(filename)
            return int(synapse_helpers.os.path.isabs(filename))

    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=FakeH()))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    assert calls[0] == f"{suffix}Helper.hoc"
    assert calls[1].endswith(f"{suffix}Helper.hoc")
    assert calls[1] != calls[0]
    synapse_helpers._loaded_helpers.discard(suffix)


def test_external_helper_wins_over_bundled_with_real_neuron(tmp_path, monkeypatch):
    """Integration test using the real NEURON loader.

    When a helper HOC is discoverable through HOC_LIBRARY_PATH *and* a helper
    with the same name exists in the bundled directory, the external one must
    win. This locks in the user-override precedence guarantee end-to-end
    (real ``h.load_file`` + real HOC_LIBRARY_PATH resolution), rather than
    just the control flow of ``load_synapse_helper``.
    """
    import neuron

    suffix = "ExternalWinsRealNrn"
    helper_file = f"{suffix}Helper.hoc"

    external_dir = tmp_path / "external"
    bundled_dir = tmp_path / "bundled"
    external_dir.mkdir()
    bundled_dir.mkdir()

    # Both files define the same template but set a distinct marker global so
    # we can tell which file NEURON actually loaded.
    (external_dir / helper_file).write_text(
        f"external_marker_{suffix} = 1\n"
        f"begintemplate {suffix}Helper\n"
        "public synapse\n"
        "objref synapse\n"
        "proc init() {}\n"
        f"endtemplate {suffix}Helper\n"
    )
    (bundled_dir / helper_file).write_text(
        f"bundled_marker_{suffix} = 1\n"
        f"begintemplate {suffix}Helper\n"
        "public synapse\n"
        "objref synapse\n"
        "proc init() {}\n"
        f"endtemplate {suffix}Helper\n"
    )

    # External dir listed first; the loader appends the bundled dir after it.
    monkeypatch.setenv("HOC_LIBRARY_PATH", str(external_dir))
    monkeypatch.setattr(
        synapse_helpers, "_bundled_hoc_directory", lambda: str(bundled_dir)
    )
    synapse_helpers._loaded_helpers.discard(suffix)

    try:
        assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
        # The external file executed (its marker global is defined) ...
        assert hasattr(neuron.h, f"external_marker_{suffix}")
        # ... and the bundled file was not loaded.
        assert not hasattr(neuron.h, f"bundled_marker_{suffix}")
    finally:
        synapse_helpers._loaded_helpers.discard(suffix)


@pytest.fixture
def clean_helper_search_dirs():
    """Isolate the registered helper search dirs around a test."""
    synapse_helpers.clear_helper_search_dirs()
    yield
    synapse_helpers.clear_helper_search_dirs()


def test_registered_dir_searched_between_hoc_path_and_bundled(
    tmp_path, monkeypatch, clean_helper_search_dirs
):
    """Registered dirs are tried after HOC_LIBRARY_PATH and before bundled."""
    suffix = "RegisteredDirOrder"
    helper_file = f"{suffix}Helper.hoc"
    helper_dir = tmp_path / "circuit_mods"
    helper_dir.mkdir()
    (helper_dir / helper_file).write_text("// helper\n")
    synapse_helpers.register_helper_search_dirs([helper_dir])

    calls = []
    bundled_dir = synapse_helpers._bundled_hoc_directory()

    class FakeH:
        def load_file(self, filename):
            calls.append(filename)
            # only the bundled absolute path "loads"
            return int(
                synapse_helpers.os.path.dirname(filename) == bundled_dir
            )

    fake_h = FakeH()
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))
    monkeypatch.setattr(
        fake_h, f"{suffix}Helper", object(), raising=False
    )

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    assert calls == [
        helper_file,  # HOC_LIBRARY_PATH / cwd
        str(helper_dir / helper_file),  # registered circuit dir
        synapse_helpers.os.path.join(bundled_dir, helper_file),  # bundled
    ]
    synapse_helpers._loaded_helpers.discard(suffix)


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


def test_missing_helper_error_lists_registered_dirs(
    tmp_path, monkeypatch, clean_helper_search_dirs
):
    helper_dir = tmp_path / "circuit_mods"
    helper_dir.mkdir()
    synapse_helpers.register_helper_search_dirs([helper_dir])

    suffix = "MissingRegisteredDirCoverage"
    fake_h = SimpleNamespace(load_file=lambda _: 0)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    with pytest.raises(FileNotFoundError, match=str(helper_dir)):
        synapse_helpers.load_synapse_helper(suffix)


def test_load_synapse_helper_missing_raises():
    """load_synapse_helper raises FileNotFoundError when the helper HOC cannot
    be located on HOC_LIBRARY_PATH."""
    from bluecellulab.synapse.synapse_helpers import load_synapse_helper

    with pytest.raises((FileNotFoundError, AttributeError)):
        load_synapse_helper("ThisSuffixDoesNotExistAnywhere_XYZ")


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
    synapse_helpers._loaded_helpers.add(suffix)
    fake_h = SimpleNamespace(load_file=pytest.fail)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    synapse_helpers._loaded_helpers.discard(suffix)


def test_load_synapse_helper_rejects_helper_without_template(monkeypatch):
    suffix = "MissingTemplateCoverage"
    fake_h = SimpleNamespace(load_file=lambda _: 1)
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    with pytest.raises(AttributeError, match="did not define template"):
        synapse_helpers.load_synapse_helper(suffix)


def test_load_synapse_helper_loads_template(monkeypatch):
    suffix = "LoadedTemplateCoverage"
    fake_h = SimpleNamespace(load_file=lambda _: 1, LoadedTemplateCoverageHelper=object())
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.load_synapse_helper(suffix) == f"{suffix}Helper"
    assert synapse_helpers.helper_available(suffix)
    synapse_helpers._loaded_helpers.discard(suffix)


def test_syn_params_adapter_maps_enum_and_string_keys():
    adapter = _SynParamsAdapter(
        pd.Series({
            SynapseProperty.PRE_GID: 12,
            SynapseProperty.G_SYNX: 0.5,
            "custom_parameter": 3,
        })
    )

    assert adapter.sgid == 12
    assert adapter.weight == 0.5
    assert adapter.custom_parameter == 3


def test_syn_params_adapter_defaults_reserved_fields():
    """_SynParamsAdapter must default maskValue and location like neurodamus
    ``SynapseReader._reserved``."""
    adapter = _SynParamsAdapter(pd.Series())

    assert adapter.maskValue == -1.0
    assert adapter.location == 0.5


def test_syn_params_adapter_ignores_unassignable_attribute():
    _SynParamsAdapter(pd.Series({"__dict__": 3}))


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
    synapse_helpers._loaded_helpers.discard(suffix)


def test_get_helper_needed_attributes_empty_when_no_metadata(monkeypatch):
    suffix = "NoAttrsCoverage"
    fake_h = SimpleNamespace(
        load_file=lambda _: 1,
        NoAttrsCoverageHelper=object(),
    )
    monkeypatch.setattr(synapse_helpers, "neuron", SimpleNamespace(h=fake_h))

    assert synapse_helpers.get_helper_needed_attributes(suffix) == []
    synapse_helpers._loaded_helpers.discard(suffix)
