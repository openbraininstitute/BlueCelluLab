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
"""Unit tests for bluecellulab/simulation/modifications.py."""

from pathlib import Path
from unittest import mock

import pytest

from bluecellulab.circuit.config.sections import (
    ModificationBase,
    ModificationCompartmentSet,
    ModificationConfigureAllSections,
    ModificationSection,
    ModificationSectionList,
    ModificationTTX,
    modification_from_libsonata,
)
from bluecellulab.simulation.modifications import (
    _apply_configure_all_sections,
    _apply_section,
    _apply_section_list,
    _apply_ttx,
    _exec_on_section,
    apply_modifications,
    parse_section_configure,
)


# ---- parse_section_configure tests ----


class TestParseSectionConfigure:
    def test_simple_assignment(self):
        config, attrs = parse_section_configure("%s.cm = 2.0")
        assert config == "sec.cm = 2.0"
        assert attrs == {"cm"}

    def test_multiple_assignments(self):
        config, attrs = parse_section_configure("%s.gbar = 0; %s.cm = 1.5")
        assert "sec.gbar = 0" in config
        assert "sec.cm = 1.5" in config
        assert attrs == {"gbar", "cm"}

    def test_augmented_assignment(self):
        config, attrs = parse_section_configure("%s.gbar *= 0.5")
        assert config == "sec.gbar *= 0.5"
        assert "gbar" in attrs

    def test_invalid_non_assignment(self):
        with pytest.raises(ValueError, match="assignments"):
            parse_section_configure("%s.gbar")

    def test_invalid_no_placeholder(self):
        with pytest.raises(Exception):
            parse_section_configure("gbar = 0")


# ---- _exec_on_section tests ----


class TestExecOnSection:
    def test_applies_when_attr_exists(self):
        section = mock.MagicMock()
        section.cm = 1.0
        result = _exec_on_section("sec.cm = 2.0", section, {"cm"})
        assert result is True
        assert section.cm == 2.0

    def test_skips_when_attr_missing(self):
        section = mock.MagicMock(spec=[])  # no attributes
        result = _exec_on_section("sec.cm = 2.0", section, {"cm"})
        assert result is False


# ---- modification_from_libsonata tests ----


class TestModificationFromLibsonata:
    def _make_mock(self, type_name, **kwargs):
        m = mock.MagicMock()
        m.type.name = type_name
        m.name = kwargs.get("name", "test_mod")
        if "node_set" in kwargs:
            m.node_set = kwargs["node_set"]
        if "section_configure" in kwargs:
            m.section_configure = kwargs["section_configure"]
        if "compartment_set" in kwargs:
            m.compartment_set = kwargs["compartment_set"]
        return m

    def test_ttx(self):
        mod = self._make_mock("ttx", node_set="target")
        result = modification_from_libsonata(mod)
        assert isinstance(result, ModificationTTX)
        assert result.name == "test_mod"
        assert result.node_set == "target"

    def test_configure_all_sections(self):
        mod = self._make_mock(
            "configure_all_sections", node_set="target", section_configure="%s.cm = 2"
        )
        result = modification_from_libsonata(mod)
        assert isinstance(result, ModificationConfigureAllSections)
        assert result.section_configure == "%s.cm = 2"

    def test_section_list(self):
        mod = self._make_mock(
            "section_list", node_set="target", section_configure="apical.gbar = 0"
        )
        result = modification_from_libsonata(mod)
        assert isinstance(result, ModificationSectionList)

    def test_section(self):
        mod = self._make_mock(
            "section", node_set="target", section_configure="apic[10].gbar = 0"
        )
        result = modification_from_libsonata(mod)
        assert isinstance(result, ModificationSection)

    def test_compartment_set(self):
        mod = self._make_mock(
            "compartment_set", compartment_set="my_set", section_configure="gbar = 1.5"
        )
        result = modification_from_libsonata(mod)
        assert isinstance(result, ModificationCompartmentSet)
        assert result.compartment_set == "my_set"

    def test_unknown_type(self):
        mod = self._make_mock("unknown_type")
        with pytest.raises(ValueError, match="Unknown modification type"):
            modification_from_libsonata(mod)


# ---- Handler tests with mocked cells ----


def _make_mock_cell(sections=None, section_lists=None, enable_ttx=True):
    """Create a mock cell with sections and optional section lists."""
    cell = mock.MagicMock()
    if sections is None:
        sections = {}
    cell.sections = sections

    # Section list properties
    for list_name in ["somatic", "basal", "apical", "axonal"]:
        if section_lists and list_name in section_lists:
            setattr(
                type(cell),
                list_name,
                mock.PropertyMock(return_value=section_lists[list_name]),
            )
        else:
            setattr(type(cell), list_name, mock.PropertyMock(return_value=[]))

    if enable_ttx:
        cell.enable_ttx = mock.MagicMock()
    return cell


def _make_mock_section(name="soma[0]", attrs=None):
    """Create a mock NEURON section with given attributes."""
    section = mock.MagicMock()
    section.name.return_value = f"Cell.{name}"
    if attrs:
        for k, v in attrs.items():
            setattr(section, k, v)
    return section


def _make_circuit_access(target_cell_ids=None):
    """Create a mock circuit access."""
    ca = mock.MagicMock()
    if target_cell_ids is not None:
        ca.get_target_cell_ids.return_value = set(target_cell_ids)
    return ca


class TestApplyTTX:
    def test_enables_ttx_on_target_cells(self):
        cell_id = mock.MagicMock()
        cell = _make_mock_cell()
        cells = {cell_id: cell}
        mod = ModificationTTX(name="block", type="ttx", node_set="target")
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        _apply_ttx(cells, mod, ca)
        cell.enable_ttx.assert_called_once()

    def test_zero_match_warning(self, caplog):
        import logging

        cell_id = mock.MagicMock()
        cell = _make_mock_cell()
        cells = {cell_id: cell}
        mod = ModificationTTX(name="block", type="ttx", node_set="empty")
        ca = _make_circuit_access(target_cell_ids=[])

        with caplog.at_level(logging.WARNING):
            _apply_ttx(cells, mod, ca)
        assert "matched zero cells" in caplog.text


class TestApplyConfigureAllSections:
    def test_applies_to_matching_sections(self):
        sec1 = _make_mock_section("soma[0]", {"cm": 1.0})
        sec2 = _make_mock_section("dend[0]", {"cm": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(sections={"soma[0]": sec1, "dend[0]": sec2})
        cells = {cell_id: cell}
        mod = ModificationConfigureAllSections(
            name="set_cm",
            type="configure_all_sections",
            node_set="target",
            section_configure="%s.cm = 2.0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        _apply_configure_all_sections(cells, mod, ca)
        assert sec1.cm == 2.0
        assert sec2.cm == 2.0

    def test_skips_sections_missing_attr(self):
        sec1 = _make_mock_section("soma[0]", {"cm": 1.0})
        # Create a section that explicitly lacks the 'cm' attribute
        sec2 = _make_mock_section("dend[0]")
        del sec2.cm  # remove the auto-created attribute
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(sections={"soma[0]": sec1, "dend[0]": sec2})
        cells = {cell_id: cell}
        mod = ModificationConfigureAllSections(
            name="set_cm",
            type="configure_all_sections",
            node_set="target",
            section_configure="%s.cm = 2.0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        _apply_configure_all_sections(cells, mod, ca)
        assert sec1.cm == 2.0
        # sec2 should not have cm set since it was missing
        assert not hasattr(sec2, "cm")


class TestApplySectionList:
    def test_applies_to_section_list(self):
        sec = _make_mock_section("apic[0]", {"gbar": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(
            sections={"apic[0]": sec},
            section_lists={"apical": [sec]},
        )
        cells = {cell_id: cell}
        mod = ModificationSectionList(
            name="no_gbar",
            type="section_list",
            node_set="target",
            section_configure="apical.gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        _apply_section_list(cells, mod, ca)
        assert sec.gbar == 0

    def test_warns_empty_section_list(self, caplog):
        import logging

        cell_id = mock.MagicMock()
        cell = _make_mock_cell(section_lists={"apical": []})
        cells = {cell_id: cell}
        mod = ModificationSectionList(
            name="no_gbar",
            type="section_list",
            node_set="target",
            section_configure="apical.gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        with caplog.at_level(logging.WARNING):
            _apply_section_list(cells, mod, ca)
        assert "no 'apical' sections" in caplog.text

    def test_unknown_list_name_raises(self):
        mod = ModificationSectionList(
            name="bad",
            type="section_list",
            node_set="target",
            section_configure="unknown_list.gbar = 0",
        )
        with pytest.raises(ValueError, match="unknown section list name"):
            _apply_section_list({}, mod, _make_circuit_access([]))


class TestApplySection:
    def test_applies_to_named_section(self):
        sec = _make_mock_section("apic[10]", {"gbar": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(sections={"apic[10]": sec})
        cell.get_section.return_value = sec
        cells = {cell_id: cell}
        mod = ModificationSection(
            name="set_gbar",
            type="section",
            node_set="target",
            section_configure="apic[10].gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        _apply_section(cells, mod, ca)
        assert sec.gbar == 0

    def test_warns_missing_section(self, caplog):
        import logging

        cell_id = mock.MagicMock()
        cell = _make_mock_cell()
        cell.get_section.side_effect = ValueError("not found")
        cells = {cell_id: cell}
        mod = ModificationSection(
            name="set_gbar",
            type="section",
            node_set="target",
            section_configure="apic[10].gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        with caplog.at_level(logging.WARNING):
            _apply_section(cells, mod, ca)
        assert "does not have section" in caplog.text


class TestApplyConfigureAllSectionsZeroMatch:
    def test_warns_zero_sections(self, caplog):
        import logging

        cell_id = mock.MagicMock()
        # Cell with sections that all lack the referenced attribute
        sec = _make_mock_section("soma[0]")
        del sec.nonexistent_attr
        cell = _make_mock_cell(sections={"soma[0]": sec})
        cells = {cell_id: cell}
        mod = ModificationConfigureAllSections(
            name="zero_match",
            type="configure_all_sections",
            node_set="target",
            section_configure="%s.nonexistent_attr = 1",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        with caplog.at_level(logging.WARNING):
            _apply_configure_all_sections(cells, mod, ca)
        assert "applied to zero sections" in caplog.text


class _CellWithoutApical:
    """Helper cell-like object that raises AttributeError for 'apical'."""

    sections = {}

    @property
    def apical(self):
        raise AttributeError("no apical property")


class TestApplySectionListAttributeError:
    def test_warns_missing_property(self, caplog):
        """Test the AttributeError branch when cell lacks the section list property."""
        import logging

        cell_id = mock.MagicMock()
        cell = _CellWithoutApical()
        cells = {cell_id: cell}
        mod = ModificationSectionList(
            name="no_prop",
            type="section_list",
            node_set="target",
            section_configure="apical.gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        with caplog.at_level(logging.WARNING):
            _apply_section_list(cells, mod, ca)
        assert "has no 'apical' property" in caplog.text

    def test_invalid_section_configure_raises(self):
        mod = ModificationSectionList(
            name="bad",
            type="section_list",
            node_set="target",
            section_configure="= 0",
        )
        with pytest.raises(ValueError):
            _apply_section_list({}, mod, _make_circuit_access([]))


class TestApplySectionEdgeCases:
    def test_invalid_section_configure_raises(self):
        mod = ModificationSection(
            name="bad",
            type="section",
            node_set="target",
            section_configure="gbar = 0",  # no section[idx]. prefix
        )
        with pytest.raises(ValueError, match="cannot extract section name"):
            _apply_section({}, mod, _make_circuit_access([]))

    def test_zero_match_warning(self, caplog):
        import logging

        cell_id = mock.MagicMock()
        cell = _make_mock_cell()
        cell.get_section.side_effect = ValueError("not found")
        cells = {cell_id: cell}
        mod = ModificationSection(
            name="zero",
            type="section",
            node_set="target",
            section_configure="apic[10].gbar = 0",
        )
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        with caplog.at_level(logging.WARNING):
            _apply_section(cells, mod, ca)
        assert "applied to zero sections" in caplog.text


class TestApplyCompartmentSet:
    def _make_segment(self, attrs=None):
        seg = mock.MagicMock()
        if attrs:
            for k, v in attrs.items():
                setattr(seg, k, v)
        return seg

    def test_applies_to_resolved_segments(self):
        from bluecellulab.simulation.modifications import _apply_compartment_set

        cell_id = mock.MagicMock()
        cell_id.id = 0
        cell_id.population_name = "NodeA"

        seg = self._make_segment({"gbar": 1.0})
        section = mock.MagicMock()
        section.return_value = seg  # section(seg_x) returns the segment
        section.name.return_value = "Cell.soma[0]"

        cell = mock.MagicMock()
        cell.resolve_segments_from_compartment_set.return_value = [
            (section, "soma[0]", 0.5)
        ]
        cells = {cell_id: cell}

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.return_value = {
            "my_set": {
                "population": "NodeA",
                "compartment_set": [[0, "soma[0]", 0.5]],
            }
        }

        mod = ModificationCompartmentSet(
            name="set_gbar",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0.5",
        )
        _apply_compartment_set(cells, mod, ca)
        assert seg.gbar == 0.5

    def test_warns_missing_compartment_sets_file(self, caplog):
        import logging
        from bluecellulab.simulation.modifications import _apply_compartment_set

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.side_effect = ValueError("no file")

        mod = ModificationCompartmentSet(
            name="no_file",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0",
        )
        with caplog.at_level(logging.WARNING):
            _apply_compartment_set({}, mod, ca)
        assert "cannot load compartment_sets_file" in caplog.text

    def test_raises_missing_compartment_set_name(self):
        from bluecellulab.simulation.modifications import _apply_compartment_set

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.return_value = {}

        mod = ModificationCompartmentSet(
            name="missing",
            type="compartment_set",
            compartment_set="nonexistent",
            section_configure="gbar = 0",
        )
        with pytest.raises(ValueError, match="not found in compartment_sets file"):
            _apply_compartment_set({}, mod, ca)

    def test_warns_failed_segment_resolution(self, caplog):
        import logging
        from bluecellulab.simulation.modifications import _apply_compartment_set

        cell_id = mock.MagicMock()
        cell_id.id = 0
        cell_id.population_name = "NodeA"
        cell = mock.MagicMock()
        cell.resolve_segments_from_compartment_set.side_effect = ValueError("bad")
        cells = {cell_id: cell}

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.return_value = {
            "my_set": {
                "population": "NodeA",
                "compartment_set": [[0, "soma[0]", 0.5]],
            }
        }

        mod = ModificationCompartmentSet(
            name="fail_resolve",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0",
        )
        with caplog.at_level(logging.WARNING):
            _apply_compartment_set(cells, mod, ca)
        assert "failed to resolve segments" in caplog.text

    def test_zero_match_warning(self, caplog):
        import logging
        from bluecellulab.simulation.modifications import _apply_compartment_set

        cell_id = mock.MagicMock()
        cell_id.id = 0
        cell_id.population_name = "NodeA"

        # Segment missing the referenced attribute
        seg = mock.MagicMock(spec=[])
        section = mock.MagicMock()
        section.return_value = seg

        cell = mock.MagicMock()
        cell.resolve_segments_from_compartment_set.return_value = [
            (section, "soma[0]", 0.5)
        ]
        cells = {cell_id: cell}

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.return_value = {
            "my_set": {
                "population": "NodeA",
                "compartment_set": [[0, "soma[0]", 0.5]],
            }
        }

        mod = ModificationCompartmentSet(
            name="zero_seg",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0",
        )
        with caplog.at_level(logging.WARNING):
            _apply_compartment_set(cells, mod, ca)
        assert "applied to zero segments" in caplog.text

    def test_skips_wrong_population(self):
        from bluecellulab.simulation.modifications import _apply_compartment_set

        cell_id = mock.MagicMock()
        cell_id.id = 0
        cell_id.population_name = "NodeB"  # wrong population
        cell = mock.MagicMock()
        cells = {cell_id: cell}

        ca = mock.MagicMock()
        ca.config.get_compartment_sets.return_value = {
            "my_set": {
                "population": "NodeA",
                "compartment_set": [[0, "soma[0]", 0.5]],
            }
        }

        mod = ModificationCompartmentSet(
            name="wrong_pop",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0",
        )
        _apply_compartment_set(cells, mod, ca)
        # Cell should not have been touched
        cell.resolve_segments_from_compartment_set.assert_not_called()


class TestApplyModifications:
    def test_dispatches_ttx(self):
        cell_id = mock.MagicMock()
        cell = _make_mock_cell()
        cells = {cell_id: cell}
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        mod_ttx = ModificationTTX(name="ttx", type="ttx", node_set="target")
        apply_modifications(cells, [mod_ttx], ca)
        cell.enable_ttx.assert_called_once()

    def test_dispatches_configure_all_sections(self):
        sec = _make_mock_section("soma[0]", {"cm": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(sections={"soma[0]": sec})
        cells = {cell_id: cell}
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        mod = ModificationConfigureAllSections(
            name="cas",
            type="configure_all_sections",
            node_set="target",
            section_configure="%s.cm = 5.0",
        )
        apply_modifications(cells, [mod], ca)
        assert sec.cm == 5.0

    def test_dispatches_section_list(self):
        sec = _make_mock_section("apic[0]", {"gbar": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(
            sections={"apic[0]": sec},
            section_lists={"apical": [sec]},
        )
        cells = {cell_id: cell}
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        mod = ModificationSectionList(
            name="sl",
            type="section_list",
            node_set="target",
            section_configure="apical.gbar = 0",
        )
        apply_modifications(cells, [mod], ca)
        assert sec.gbar == 0

    def test_dispatches_section(self):
        sec = _make_mock_section("apic[10]", {"gbar": 1.0})
        cell_id = mock.MagicMock()
        cell = _make_mock_cell(sections={"apic[10]": sec})
        cell.get_section.return_value = sec
        cells = {cell_id: cell}
        ca = _make_circuit_access(target_cell_ids=[cell_id])

        mod = ModificationSection(
            name="sec",
            type="section",
            node_set="target",
            section_configure="apic[10].gbar = 0",
        )
        apply_modifications(cells, [mod], ca)
        assert sec.gbar == 0

    def test_dispatches_compartment_set(self):
        seg = mock.MagicMock()
        seg.gbar = 1.0
        section = mock.MagicMock()
        section.return_value = seg

        cell_id = mock.MagicMock()
        cell_id.id = 0
        cell_id.population_name = "NodeA"
        cell = mock.MagicMock()
        cell.resolve_segments_from_compartment_set.return_value = [
            (section, "soma[0]", 0.5)
        ]
        cells = {cell_id: cell}

        ca = mock.MagicMock()
        ca.get_target_cell_ids.return_value = set()
        ca.config.get_compartment_sets.return_value = {
            "my_set": {
                "population": "NodeA",
                "compartment_set": [[0, "soma[0]", 0.5]],
            }
        }

        mod = ModificationCompartmentSet(
            name="cs",
            type="compartment_set",
            compartment_set="my_set",
            section_configure="gbar = 0.5",
        )
        apply_modifications(cells, [mod], ca)
        assert seg.gbar == 0.5

    def test_unknown_type_raises(self):
        mod = ModificationBase(name="bad", type="unknown")
        with pytest.raises(ValueError, match="Unknown modification type"):
            apply_modifications({}, [mod], mock.MagicMock())


# ---- Integration test: parse modifications from simulation config ----

parent_dir = Path(__file__).resolve().parent.parent

modifications_conf_path = (
    parent_dir
    / "examples"
    / "sim_quick_scx_sonata_multicircuit"
    / "simulation_config_modifications.json"
)


def test_get_modifications_from_config():
    """Test that SonataSimulationConfig.get_modifications() parses all 5 types."""
    from bluecellulab.circuit.config import SonataSimulationConfig

    sim = SonataSimulationConfig(modifications_conf_path)
    mods = sim.get_modifications()
    assert len(mods) == 5

    assert isinstance(mods[0], ModificationTTX)
    assert mods[0].name == "TTX_block"
    assert mods[0].node_set == "Mosaic_A"
    assert mods[0].type == "ttx"

    assert isinstance(mods[1], ModificationConfigureAllSections)
    assert mods[1].name == "configure_all"
    assert mods[1].section_configure == "%s.cm = 2.0"

    assert isinstance(mods[2], ModificationSectionList)
    assert mods[2].name == "scale_soma"
    assert mods[2].section_configure == "somatic.cm *= 1.5"

    assert isinstance(mods[3], ModificationSection)
    assert mods[3].name == "set_dend0"
    assert mods[3].section_configure == "dend[0].cm = 5.0"

    assert isinstance(mods[4], ModificationCompartmentSet)
    assert mods[4].name == "set_compartment"
    assert mods[4].compartment_set == "Mosaic_A"
    assert mods[4].section_configure == "cm = 10.0"
