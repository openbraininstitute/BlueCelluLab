"""Tests for the neurodamus-style mod_override / helper-HOC machinery."""

from __future__ import annotations

import pytest

from bluecellulab.circuit.config.sections import ConnectionOverrides


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
