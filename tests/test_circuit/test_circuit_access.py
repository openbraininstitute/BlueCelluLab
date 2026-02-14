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
"""Unit tests for the circuit_access module."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from bluepysnap.exceptions import BluepySnapError

from bluecellulab.circuit import CellId, SonataCircuitAccess
from bluecellulab.circuit.circuit_access import EmodelProperties, get_synapse_connection_parameters
from bluecellulab.circuit import SynapseProperty


parent_dir = Path(__file__).resolve().parent.parent

hipp_circuit_with_projections = (
    parent_dir
    / "examples"
    / "sonata_unit_test_sims"
    / "projections"
    / "simulation_config.json"
)


def test_get_synapse_connection_parameters():
    """Test that the synapse connection parameters are correctly created."""
    circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)
    # both cells are Excitatory
    pre_cell_id = CellId("hippocampus_neurons", 1)
    post_cell_id = CellId("hippocampus_neurons", 2)
    connection_params = get_synapse_connection_parameters(
        circuit_access, pre_cell_id, post_cell_id)
    assert connection_params["add_synapse"] is True
    assert connection_params["Weight"] == 1.0
    assert connection_params["SpontMinis"] == 0.01
    syn_configure = connection_params["SynapseConfigure"]
    assert syn_configure[0] == "%s.NMDA_ratio = 1.22 %s.tau_r_NMDA = 3.9 %s.tau_d_NMDA = 148.5"
    assert syn_configure[1] == "%s.mg = 1.0"


def test_sonata_circuit_access_file_not_found():
    with pytest.raises(FileNotFoundError):
        SonataCircuitAccess("non_existing_file")


class TestSonataCircuitAccess:
    def setup_method(self):
        self.circuit_access = SonataCircuitAccess(hipp_circuit_with_projections)

    def test_available_cell_properties(self):
        assert self.circuit_access.available_cell_properties == {
            "x",
            "region",
            "rotation_angle_yaxis",
            "etype",
            "model_template",
            "synapse_class",
            "@dynamics:holding_current",
            "morph_class",
            "population",
            "morphology",
            "y",
            "layer",
            "mtype",
            "rotation_angle_xaxis",
            "z",
            "@dynamics:threshold_current",
            "model_type",
            "rotation_angle_zaxis",
        }

    def test_get_emodel_properties(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.get_emodel_properties(cell_id)
        assert res == EmodelProperties(
            threshold_current=0.33203125, holding_current=-0.116351104071555
        )

    def test_get_template_format(self):
        res = self.circuit_access.get_template_format()
        assert res == "v6"

    def test_get_cell_properties(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.get_cell_properties(cell_id, "layer")
        assert res.values == ["SP"]

    def test_extract_synapses(self):
        cell_id = CellId("hippocampus_neurons", 1)

        # intrinsic-only
        res = self.circuit_access.extract_synapses(cell_id, False)
        assert res.empty

        # intrinsic + all projections
        res = self.circuit_access.extract_synapses(cell_id, True)
        assert res.shape == (1742, 16)
        assert all(res["source_popid"] == 2126)
        assert all(res["source_population_name"] == "hippocampus_projections")
        assert all(res["target_popid"] == 378)
        assert all(res[SynapseProperty.POST_SEGMENT_ID] != -1)
        assert SynapseProperty.U_HILL_COEFFICIENT not in res.columns
        assert SynapseProperty.CONDUCTANCE_RATIO not in res.columns

        # specific projection selection
        projection = "hippocampus_projections__hippocampus_neurons__chemical"
        res = self.circuit_access.extract_synapses(cell_id, projection)
        assert res.shape == (1742, 16)

        res = self.circuit_access.extract_synapses(cell_id, [projection])
        assert res.shape == (1742, 16)

        # empty list == no projections requested (intrinsic-only)
        res = self.circuit_access.extract_synapses(cell_id, [])
        assert res.empty

    def test_target_contains_cell(self):
        target = "most_central_10_SP_PC"
        cell = CellId("hippocampus_neurons", 1)
        assert self.circuit_access.target_contains_cell(target, cell)
        cell2 = CellId("hippocampus_neurons", 100)
        assert not self.circuit_access.target_contains_cell(target, cell2)

    def test_get_target_cell_ids(self):
        target = "most_central_10_SP_PC"
        res = self.circuit_access.get_target_cell_ids(target)
        res_populations = {x.population_name for x in res}
        res_ids = {x.id for x in res}
        assert res_populations == {'hippocampus_neurons'}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_get_cell_ids_of_targets(self):
        targets = ["most_central_10_SP_PC", "Mosaic"]
        res = self.circuit_access.get_cell_ids_of_targets(targets)
        res_ids = {x.id for x in res}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_get_gids_of_mtypes(self):
        res = self.circuit_access.get_gids_of_mtypes(["SP_PC"])
        res_ids = {x.id for x in res}
        assert res_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    def test_morph_filepath(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.morph_filepath(cell_id)
        fname = "dend-mpg141216_A_idA_axon-mpg141017_a1-2_idC"\
            "_-_Scale_x1.000_y0.900_z1.000_-_Clone_12.swc"
        assert res.rsplit("/", 1)[-1] == fname

    def test_emodel_path(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.emodel_path(cell_id)
        fname = "CA1_pyr_cACpyr_mpg141017_a1_2_idC_2019032814340.hoc"
        assert res.rsplit("/", 1)[-1] == fname

    def test_is_valid_group(self):
        group = "most_central_10_SP_PC"
        assert self.circuit_access.is_valid_group(group)
        group2 = "most_central_10_SP_PC2"
        assert not self.circuit_access.is_valid_group(group2)

    def test_fetch_cell_info(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_cell_info(cell_id)
        assert res.shape == (18,)
        assert res.etype == "cACpyr"
        assert res.mtype == "SP_PC"
        assert res.rotation_angle_zaxis == pytest.approx(-3.141593)

    def test_fetch_mini_frequencies(self):
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (None, None)

    def test_node_properties_available(self):
        assert self.circuit_access.node_properties_available

    @patch("bluecellulab.circuit.SonataCircuitAccess.fetch_cell_info")
    def test_fetch_mini_frequencies_with_mock(self, mock_fetch_cell_info):
        mock_fetch_cell_info.return_value = pd.Series(
            {
                "exc-mini_frequency": 0.03,
                "inh-mini_frequency": 0.05,
            }
        )
        cell_id = CellId("hippocampus_neurons", 1)
        res = self.circuit_access.fetch_mini_frequencies(cell_id)
        assert res == (0.03, 0.05)

    def test_get_population_ids(self):
        source = "hippocampus_projections"
        target = "hippocampus_neurons"
        source_popid, target_popid = self.circuit_access.get_population_ids(source, target)
        assert source_popid == 2126
        assert target_popid == 378


def test_morph_filepath_asc():
    """Unit test for SonataCircuitAccess::morph_filepath for asc retrieval."""
    circuit_sonata_quick_scx_config = (
        parent_dir
        / "examples"
        / "sonata_unit_test_sims"
        / "condition_parameters"
        / "simulation_config.json"
    )

    circuit_access = SonataCircuitAccess(circuit_sonata_quick_scx_config)
    asc_morph = circuit_access.morph_filepath(CellId("NodeA", 1))
    assert asc_morph.endswith(".asc")


def test_get_emodel_properties_soma_scaler():
    """Test the retrieval of soma scaler value."""
    circuit_sonata_quick_scx_config = (
        parent_dir
        / "examples"
        / "sonata_unit_test_sims"
        / "condition_parameters"
        / "simulation_config.json"
    )

    circuit_access = SonataCircuitAccess(circuit_sonata_quick_scx_config)
    assert circuit_access.get_emodel_properties(CellId("NodeA", 0)).soma_scaler == 1.0
    assert circuit_access.get_emodel_properties(CellId("NodeA", 1)).soma_scaler == 1.002


def test_get_target_cell_ids_with_population_filter():
    access = object.__new__(SonataCircuitAccess)
    access.config = SimpleNamespace(
        get_node_sets=lambda: {"target": {"population": "popA", "node_id": [1, 2]}}
    )
    access._circuit = SimpleNamespace(
        nodes={"popA": SimpleNamespace(ids=lambda _: [1, 2])}
    )

    result = SonataCircuitAccess.get_target_cell_ids(access, "target")

    assert result == {CellId("popA", 1), CellId("popA", 2)}


def test_get_target_cell_ids_without_population_filter():
    access = object.__new__(SonataCircuitAccess)
    access.config = SimpleNamespace(get_node_sets=lambda: {"target": {"layer": "L23"}})
    access._circuit = SimpleNamespace(
        nodes=SimpleNamespace(
            ids=lambda _: [
                SimpleNamespace(population="popA", id=1),
                SimpleNamespace(population="popB", id=5),
            ]
        )
    )

    result = SonataCircuitAccess.get_target_cell_ids(access, "target")

    assert result == {CellId("popA", 1), CellId("popB", 5)}


def test_morph_filepath_h5v1_container_path():
    access = object.__new__(SonataCircuitAccess)
    node_population = SimpleNamespace(
        config={"alternate_morphologies": {"h5v1": "/data/merged-morphologies.h5"}},
        get=lambda _: {"morphology": "cell_42"},
        morph=SimpleNamespace(get_filepath=lambda *_args, **_kwargs: "unused"),
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    result = SonataCircuitAccess.morph_filepath(access, CellId("popA", 42))

    assert result == "/data/merged-morphologies.h5/cell_42.h5"


def test_morph_filepath_h5v1_empty_name_falls_back_to_asc():
    access = object.__new__(SonataCircuitAccess)
    node_population = SimpleNamespace(
        config={
            "alternate_morphologies": {
                "h5v1": "/data/merged-morphologies.h5",
                "neurolucida-asc": "/data/asc",
            }
        },
        get=lambda _: {"morphology": ""},
        morph=SimpleNamespace(get_filepath=lambda *_args, **kwargs: "/data/asc/cell.asc" if kwargs.get("extension") == "asc" else "/data/swc/cell.swc"),
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    result = SonataCircuitAccess.morph_filepath(access, CellId("popA", 42))

    assert result == "/data/asc/cell.asc"


def test_morph_filepath_asc_error_falls_back_to_default():
    access = object.__new__(SonataCircuitAccess)

    def _get_filepath(*_args, **kwargs):
        if kwargs.get("extension") == "asc":
            raise BluepySnapError("asc not available")
        return "/data/swc/cell.swc"

    node_population = SimpleNamespace(
        config={"alternate_morphologies": {"neurolucida-asc": "/data/asc"}},
        get=lambda _: {"morphology": "cell_42"},
        morph=SimpleNamespace(get_filepath=_get_filepath),
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    result = SonataCircuitAccess.morph_filepath(access, CellId("popA", 42))

    assert result == "/data/swc/cell.swc"


def test_morph_filepath_raises_informative_error_on_full_failure():
    access = object.__new__(SonataCircuitAccess)
    node_population = SimpleNamespace(
        config={},
        get=lambda _: {"morphology": "cell_42"},
        morph=SimpleNamespace(
            get_filepath=lambda *_args, **_kwargs: (_ for _ in ()).throw(BluepySnapError("missing"))
        ),
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    with pytest.raises(BluepySnapError, match="Could not determine morphology path"):
        SonataCircuitAccess.morph_filepath(access, CellId("popA", 42))


def test_emodel_path_fallback_from_model_template():
    access = object.__new__(SonataCircuitAccess)

    class _Models:
        @staticmethod
        def get_filepath(_cell_id):
            raise BluepySnapError("missing models helper")

    node_population = SimpleNamespace(
        models=_Models(),
        get=lambda _: {"model_template": "hoc:MyTemplate"},
        config={"biophysical_neuron_models_dir": "/data/emodels"},
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    result = SonataCircuitAccess.emodel_path(access, CellId("popA", 1))

    assert result == "/data/emodels/MyTemplate.hoc"


def test_emodel_path_raises_on_fallback_failure():
    access = object.__new__(SonataCircuitAccess)

    class _Models:
        @staticmethod
        def get_filepath(_cell_id):
            raise BluepySnapError("missing models helper")

    node_population = SimpleNamespace(
        models=_Models(),
        get=lambda _: {"model_template": "not_hoc_format"},
        config={},
    )
    access._circuit = SimpleNamespace(nodes={"popA": node_population})

    with pytest.raises(BluepySnapError, match="Could not determine emodel path"):
        SonataCircuitAccess.emodel_path(access, CellId("popA", 1))
