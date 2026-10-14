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
"""Unit tests for the synapse_factory module."""


import json
from pathlib import Path
import pandas as pd
import pytest
from bluecellulab.cell.core import Cell

from bluecellulab.circuit.circuit_access import EmodelProperties
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit.synapse_properties import SynapseProperties, SynapseProperty, synapse_property_decoder
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.synapse import SynapseFactory
from bluecellulab.synapse.synapse_factory import SynapseType
from bluecellulab.synapse.synapse_types import GluSynapse


parent_dir = Path(__file__).resolve().parent.parent


@pytest.mark.v6
class TestSynapseFactory:

    def setup_method(self):
        emodel_properties = EmodelProperties(
            threshold_current=1.1433533430099487,
            holding_current=1.4146618843078613,
            AIS_scaler=1.4561502933502197,
            soma_scaler=1.0)
        self.cell = Cell(
            parent_dir / "examples/circuit_sonata_quick_scx/components/hoc/cADpyr_L2TPC.hoc",
            parent_dir / "examples/circuit_sonata_quick_scx/components/morphologies/asc/rr110330_C3_idA.asc",
            template_format="v6",
            emodel_properties=emodel_properties,
        )
        with open("tests/test_synapse/test-synapse-series.json") as f:
            self.syn_description = pd.Series(json.load(f, object_hook=synapse_property_decoder))

    def test_create_synapse(self):
        syn_id = ("a", 0)
        condition_parameters = Conditions.init_empty()
        popids = (0, 0)
        extracellular_calcium = 2.0
        connection_modifiers = {
            'ModOverride': 'GluSynapse',
            'Weight': 1.15143,
            'SpontMinis': 0.0,
            'SynapseConfigure': ['%s.Use = 1 %s.Use_GB = 1 %s.Use_p = 1 %s.gmax0_AMPA = gmax_p_AMPA %s.rho_GB = 1 %s.rho0_GB = 1 %s.gmax_AMPA = %s.gmax_p_AMPA']
        }

        self.cell.post_gid = 0

        synapse = SynapseFactory.create_synapse(
            self.cell, syn_id, self.syn_description, condition_parameters, popids, extracellular_calcium, connection_modifiers
        )
        assert isinstance(synapse, GluSynapse)
        assert synapse.weight == connection_modifiers["Weight"]

    def test_determine_synapse_location(self):
        res = SynapseFactory.determine_synapse_location(self.syn_description, self.cell)
        assert res.location == 0.9999999
        # set afferent_section_pos
        self.syn_description[SynapseProperty.AFFERENT_SECTION_POS] = 1.2
        res = SynapseFactory.determine_synapse_location(self.syn_description, self.cell)
        assert res.location == 0.9999999
        assert res.section.L == pytest.approx(9.530376893488256)
        self.syn_description[SynapseProperty.AFFERENT_SECTION_POS] = 0
        res = SynapseFactory.determine_synapse_location(self.syn_description, self.cell)
        assert res.location == 0.0000001

    def test_synlocation_to_segx(self):
        ipt = 13
        isec = 169
        syn_offset = 1.2331762313842773
        section = self.cell.get_psection(isec).hsection
        res = SynapseFactory.synlocation_to_segx(section, ipt, syn_offset)
        assert res == pytest.approx(0.9999999)
        res = SynapseFactory.synlocation_to_segx(section, ipt, syn_offset=-1.0)
        assert res == pytest.approx(0.9999999)

    def test_create_synapse_without_post_gid_uses_cell_id_fallback(self):
        syn_id = ("a", 0)
        condition_parameters = Conditions.init_empty()
        popids = (0, 0)
        extracellular_calcium = 2.0
        connection_modifiers = {
            "ModOverride": "GluSynapse",
            "Weight": 1.15143,
            "SpontMinis": 0.0,
            "SynapseConfigure": [
                "%s.Use = 1 %s.Use_GB = 1 %s.Use_p = 1 %s.gmax0_AMPA = gmax_p_AMPA %s.rho_GB = 1 %s.rho0_GB = 1 %s.gmax_AMPA = %s.gmax_p_AMPA"
            ],
        }

        self.cell.post_gid = None

        synapse = SynapseFactory.create_synapse(
            self.cell,
            syn_id,
            self.syn_description,
            condition_parameters,
            popids,
            extracellular_calcium,
            connection_modifiers,
        )

        assert isinstance(synapse, GluSynapse)
        assert synapse.post_gid == self.cell.cell_id.id

    def test_add_replay_synapse_without_post_gid_uses_cell_id_fallback(self):
        syn_id = ("a", 0)
        condition_parameters = Conditions.init_empty()
        popids = (0, 0)
        extracellular_calcium = 2.0
        connection_modifiers = {
            "Weight": 1.15143,
            "SpontMinis": 0.0,
            "SynapseConfigure": [
                "%s.Use = 1 %s.Use_GB = 1 %s.Use_p = 1 %s.gmax0_AMPA = gmax_p_AMPA %s.rho_GB = 1 %s.rho0_GB = 1 %s.gmax_AMPA = %s.gmax_p_AMPA"
            ],
        }

        self.cell.post_gid = None

        self.cell.add_replay_synapse(
            synapse_id=syn_id,
            syn_description=self.syn_description,
            connection_modifiers=connection_modifiers,
            condition_parameters=condition_parameters,
            popids=popids,
            extracellular_calcium=extracellular_calcium,
        )

        synapse = self.cell.synapses[syn_id]
        assert synapse.post_gid == self.cell.cell_id.id


def test_determine_synapse_type(caplog):
    syn_description = pd.Series({SynapseProperty.TYPE: 50})
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.GABAAB

    syn_description[SynapseProperty.TYPE] = 150
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.AMPANMDA


def test_plasticity_fields_do_not_select_glusynapse(caplog):
    """As in neurodamus, GluSynapse needs modoverride; the data alone never
    selects it. The ignored plasticity fields are reported once."""
    import logging

    from bluecellulab.synapse import synapse_factory

    synapse_factory._plasticity_warned = False
    syn_description = pd.Series({SynapseProperty.TYPE: 150})
    for prop in SynapseProperties.plasticity:
        syn_description[prop] = 1.0
    with caplog.at_level(logging.WARNING):
        assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.AMPANMDA
        SynapseFactory.determine_synapse_type(syn_description)
    assert caplog.text.count("modoverride='GluSynapse'") == 1


def test_tau1_does_not_select_exp2syn():
    """Exp2Syn comes only from modoverride, not from the presence of tau1."""
    syn_description = pd.Series({SynapseProperty.TYPE: 150, "tau1": 1.0})
    assert SynapseFactory.determine_synapse_type(syn_description) == SynapseType.AMPANMDA


def test_missing_synapse_type_points_to_exp2syn_override():
    with pytest.raises(BluecellulabError, match="modoverride='Exp2Syn'"):
        SynapseFactory.determine_synapse_type(pd.Series({"tau1": 1.0}))
