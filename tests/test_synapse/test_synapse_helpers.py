"""Tests for the neurodamus-style mod_override / helper-HOC machinery."""

from __future__ import annotations

from types import SimpleNamespace

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
    class Helper:
        def __init__(self, *args):
            self.args = args
            self.synapse = "point-process"

    monkeypatch.setattr(synapse_helpers, "load_synapse_helper", lambda _: "TestHelper")
    monkeypatch.setattr(synapse_types.neuron, "h", SimpleNamespace(TestHelper=Helper))
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.post_gid = 41
    synapse.hoc_args = SimpleNamespace(location=0.25)
    synapse.syn_id = SynapseID("projection", 7)
    synapse.source_popid = 2
    synapse.target_popid = 3
    synapse.syn_description = pd.Series({SynapseProperty.G_SYNX: 0.9})
    synapse.persistent = []

    synapse._build_via_helper("Test")

    assert synapse.hsynapse == "point-process"
    assert synapse.mech_name == "Test"
    assert synapse.persistent[0].args[0] == 42


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
    class Helper:
        def __init__(self, *args):
            pass

    monkeypatch.setattr(synapse_helpers, "load_synapse_helper", lambda _: "TestHelper")
    monkeypatch.setattr(synapse_types.neuron, "h", SimpleNamespace(TestHelper=Helper))
    synapse = GenericSpikeSynapse.__new__(GenericSpikeSynapse)
    synapse.post_gid = 41
    synapse.hoc_args = SimpleNamespace(location=0.25)
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
