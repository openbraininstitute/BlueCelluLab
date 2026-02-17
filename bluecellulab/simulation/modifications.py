# Copyright 2023-2024 Blue Brain Project / EPFL
# Copyright 2025-2026 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module for applying SONATA condition modifications to cells.

Implements all five SONATA modification types:
- ttx: Block Na channels via TTXDynamicsSwitch
- configure_all_sections: Apply section_configure to all sections
- section_list: Apply section_configure to a named section list
- section: Apply section_configure to specific named sections
- compartment_set: Apply section_configure to segments in a compartment set

The section_configure parser follows neurodamus's ast.parse() + restricted exec()
pattern (neurodamus/modification_manager.py).
"""

from __future__ import annotations

import ast
import logging
import re

from bluecellulab.circuit.config.sections import (
    ModificationBase,
    ModificationCompartmentSet,
    ModificationConfigureAllSections,
    ModificationSection,
    ModificationSectionList,
    ModificationTTX,
)

logger = logging.getLogger(__name__)

# Mapping from SONATA section list names to Cell property names
SECTION_LIST_MAP: dict[str, str] = {
    "somatic": "somatic",
    "soma": "somatic",
    "basal": "basal",
    "dend": "basal",
    "apical": "apical",
    "apic": "apical",
    "axonal": "axonal",
    "axon": "axonal",
}


class _AttributeCollector(ast.NodeVisitor):
    """AST visitor that collects all attribute names referenced in an
    expression."""

    def __init__(self):
        self.attrs: set[str] = set()

    def visit_Attribute(self, node):
        self.attrs.add(node.attr)
        self.generic_visit(node)


def _validate_assignment(node) -> list:
    """Return assignment targets from an AST node, raising on non-
    assignments."""
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, ast.AugAssign):
        return [node.target]
    raise ValueError(
        "section_configure must consist of one or more semicolon-separated assignments"
    )


def parse_section_configure(
    config_str: str, placeholder: str = "%s"
) -> tuple[str, set[str]]:
    """Parse a section_configure string, returning sanitized code and
    referenced attrs.

    Args:
        config_str: The raw section_configure string (e.g. "%s.gbar = 0").
        placeholder: The placeholder to replace with 'sec' (default: "%s").

    Returns:
        Tuple of (sanitized_config_str, set_of_referenced_attribute_names).

    Raises:
        ValueError: If the config string contains non-assignment statements
            or references attributes not on the placeholder.
    """
    # Replace placeholder with internal variable name
    internal = config_str.replace(f"{placeholder}.", "__sec_wildcard__.")
    collector = _AttributeCollector()
    tree = ast.parse(internal)
    for elem in tree.body:
        targets = _validate_assignment(elem)
        for tgt in targets:
            if not isinstance(tgt, ast.Attribute) or not isinstance(tgt.value, ast.Name) or tgt.value.id != "__sec_wildcard__":
                raise ValueError(
                    "section_configure only supports single assignments "
                    f"of attributes of the section wildcard {placeholder}"
                )
        collector.visit(elem)
    sanitized = internal.replace("__sec_wildcard__.", "sec.")
    return sanitized, collector.attrs


def _exec_on_section(config: str, section, attrs: set[str]) -> bool:
    """Apply a sanitized config string to a NEURON section if it has all attrs.

    Returns True if applied, False if skipped due to missing attributes.
    """
    if all(hasattr(section, attr) for attr in attrs):
        exec(config, {"__builtins__": None}, {"sec": section})  # noqa: S102
        return True
    return False


def apply_modifications(
    cells: dict, modifications: list[ModificationBase], circuit_access
) -> None:
    """Apply a list of modifications to instantiated cells.

    Args:
        cells: Dict mapping CellId to Cell objects.
        modifications: List of Modification dataclass instances.
        circuit_access: CircuitAccess instance for resolving node_sets.
    """
    for mod in modifications:
        if isinstance(mod, ModificationTTX):
            _apply_ttx(cells, mod, circuit_access)
        elif isinstance(mod, ModificationConfigureAllSections):
            _apply_configure_all_sections(cells, mod, circuit_access)
        elif isinstance(mod, ModificationSectionList):
            _apply_section_list(cells, mod, circuit_access)
        elif isinstance(mod, ModificationSection):
            _apply_section(cells, mod, circuit_access)
        elif isinstance(mod, ModificationCompartmentSet):
            _apply_compartment_set(cells, mod, circuit_access)
        else:
            raise ValueError(f"Unknown modification type: {mod.type}")


def _resolve_target_cells(cells: dict, mod, circuit_access) -> list:
    """Resolve node_set to the subset of instantiated cells that match."""
    target_ids = circuit_access.get_target_cell_ids(mod.node_set)
    return [cell_id for cell_id in cells if cell_id in target_ids]


def _apply_ttx(cells: dict, mod: ModificationTTX, circuit_access) -> None:
    """Apply TTX modification — enable TTX on all target cells."""
    logger.info(
        "Applying modification '%s' (type=ttx) to node_set '%s'", mod.name, mod.node_set
    )
    target_cell_ids = _resolve_target_cells(cells, mod, circuit_access)
    count = 0
    for cell_id in target_cell_ids:
        cells[cell_id].enable_ttx()
        count += 1
        logger.debug("  TTX enabled on cell %s", cell_id)
    logger.info("Modification '%s' (ttx): enabled on %d cells", mod.name, count)
    if count == 0:
        logger.warning(
            "TTX modification '%s' matched zero cells in node_set '%s'",
            mod.name,
            mod.node_set,
        )


def _apply_configure_all_sections(
    cells: dict, mod: ModificationConfigureAllSections, circuit_access
) -> None:
    """Apply configure_all_sections — exec section_configure on all
    sections."""
    logger.info(
        "Applying modification '%s' (type=configure_all_sections) to node_set '%s'",
        mod.name,
        mod.node_set,
    )
    config, attrs = parse_section_configure(mod.section_configure, placeholder="%s")
    target_cell_ids = _resolve_target_cells(cells, mod, circuit_access)

    n_cells = 0
    n_sections = 0
    for cell_id in target_cell_ids:
        cell = cells[cell_id]
        cell_applied = 0
        for sec_name, section in cell.sections.items():
            if _exec_on_section(config, section, attrs):
                cell_applied += 1
                logger.debug("  Applied to section '%s' of cell %s", sec_name, cell_id)
        if cell_applied > 0:
            n_cells += 1
        n_sections += cell_applied

    logger.info(
        "Modification '%s' applied to %d sections across %d cells",
        mod.name,
        n_sections,
        n_cells,
    )
    if n_sections == 0:
        logger.warning(
            "configure_all_sections '%s' applied to zero sections, "
            "please check its section_configure for possible mistakes",
            mod.name,
        )


def _apply_section_list(
    cells: dict, mod: ModificationSectionList, circuit_access
) -> None:
    """Apply section_list — exec section_configure on a named section list."""
    logger.info(
        "Applying modification '%s' (type=section_list) to node_set '%s'",
        mod.name,
        mod.node_set,
    )

    # Extract list name from section_configure: e.g. "apical.gbar = 0" -> "apical"
    # The format is "<list_name>.attr = value [; <list_name>.attr = value ...]"
    match = re.match(r"^(\w+)\.", mod.section_configure)
    if not match:
        raise ValueError(
            f"section_list modification '{mod.name}': cannot extract section list name "
            f"from section_configure '{mod.section_configure}'"
        )
    list_name = match.group(1)

    prop_name = SECTION_LIST_MAP.get(list_name)
    if prop_name is None:
        raise ValueError(
            f"section_list modification '{mod.name}': unknown section list name '{list_name}'. "
            f"Supported: {list(SECTION_LIST_MAP.keys())}"
        )

    # Replace list_name. with sec. for exec
    config_str = mod.section_configure.replace(f"{list_name}.", "sec.")
    # Parse and validate the sanitized string
    collector = _AttributeCollector()
    tree = ast.parse(config_str)
    for elem in tree.body:
        _validate_assignment(elem)
        collector.visit(elem)
    attrs = collector.attrs

    target_cell_ids = _resolve_target_cells(cells, mod, circuit_access)
    n_cells = 0
    n_sections = 0
    for cell_id in target_cell_ids:
        cell = cells[cell_id]
        try:
            section_list = getattr(cell, prop_name)
        except AttributeError:
            logger.warning(
                "section_list '%s': cell %s has no '%s' property, skipping",
                mod.name,
                cell_id,
                prop_name,
            )
            continue

        if not section_list:
            logger.warning(
                "section_list '%s': cell %s has no '%s' sections, skipping",
                mod.name,
                cell_id,
                list_name,
            )
            continue

        cell_applied = 0
        for section in section_list:
            sec_name = section.name().split(".")[-1]
            if _exec_on_section(config_str, section, attrs):
                cell_applied += 1
                logger.debug("  Applied to section '%s' of cell %s", sec_name, cell_id)
        if cell_applied > 0:
            n_cells += 1
        n_sections += cell_applied

    logger.info(
        "Modification '%s' applied to %d sections across %d cells",
        mod.name,
        n_sections,
        n_cells,
    )
    if n_sections == 0:
        logger.warning(
            "section_list '%s' applied to zero sections, "
            "please check its section_configure for possible mistakes",
            mod.name,
        )


def _apply_section(cells: dict, mod: ModificationSection, circuit_access) -> None:
    """Apply section — exec section_configure on specific named sections."""
    logger.info(
        "Applying modification '%s' (type=section) to node_set '%s'",
        mod.name,
        mod.node_set,
    )

    # Extract section names from section_configure
    # Format: "apic[10].gbar = 0; apic[10].gbar2 = 1" or "dend[3].x = 5"
    # Find all unique "<name>[<idx>]." patterns
    section_names = list(
        dict.fromkeys(re.findall(r"(\w+\[\d+\])\.", mod.section_configure))
    )
    if not section_names:
        raise ValueError(
            f"section modification '{mod.name}': cannot extract section name(s) "
            f"from section_configure '{mod.section_configure}'"
        )

    # Build per-section config strings
    # For each unique section name, replace "<name>[idx]." with "sec."
    # and parse to get attrs
    section_configs: dict[str, tuple[str, set[str]]] = {}
    for sec_name in section_names:
        escaped = re.escape(sec_name)
        config_str = re.sub(escaped + r"\.", "sec.", mod.section_configure)
        # Filter to only statements that reference this section
        # (handle multi-section configs like "apic[10].x = 0; dend[3].y = 1")
        collector = _AttributeCollector()
        tree = ast.parse(config_str)
        for elem in tree.body:
            _validate_assignment(elem)
            collector.visit(elem)
        section_configs[sec_name] = (config_str, collector.attrs)

    target_cell_ids = _resolve_target_cells(cells, mod, circuit_access)
    n_cells = 0
    n_sections = 0
    for cell_id in target_cell_ids:
        cell = cells[cell_id]
        cell_applied = 0
        for sec_name, (config_str, attrs) in section_configs.items():
            try:
                section = cell.get_section(sec_name)
            except (ValueError, TypeError):
                logger.warning(
                    "section '%s': cell %s does not have section '%s', skipping",
                    mod.name,
                    cell_id,
                    sec_name,
                )
                continue
            if _exec_on_section(config_str, section, attrs):
                cell_applied += 1
                logger.debug("  Applied to section '%s' of cell %s", sec_name, cell_id)
        if cell_applied > 0:
            n_cells += 1
        n_sections += cell_applied

    logger.info(
        "Modification '%s' applied to %d sections across %d cells",
        mod.name,
        n_sections,
        n_cells,
    )
    if n_sections == 0:
        logger.warning(
            "section '%s' applied to zero sections, "
            "please check its section_configure for possible mistakes",
            mod.name,
        )


def _apply_compartment_set(
    cells: dict, mod: ModificationCompartmentSet, circuit_access
) -> None:
    """Apply compartment_set — exec section_configure on resolved segments."""
    logger.info(
        "Applying modification '%s' (type=compartment_set) to compartment_set '%s'",
        mod.name,
        mod.compartment_set,
    )

    # Load compartment sets
    try:
        compartment_sets = circuit_access.config.get_compartment_sets()
    except ValueError as e:
        logger.warning(
            "compartment_set '%s': cannot load compartment_sets_file: %s",
            mod.name,
            e,
        )
        return

    comp_name = mod.compartment_set
    if comp_name not in compartment_sets:
        raise ValueError(
            f"compartment_set modification '{mod.name}': compartment set "
            f"'{comp_name}' not found in compartment_sets file."
        )
    comp_entry = compartment_sets[comp_name]
    comp_nodes = comp_entry.get("compartment_set", [])
    population_name = comp_entry.get("population")

    # Parse section_configure — bare format: "attr = value"
    # Prefix with "seg." to make it executable
    config_str = re.sub(r"(\b\w+)\s*=", r"seg.\1 =", mod.section_configure)
    # But we need to be careful: only LHS identifiers should be prefixed.
    # Better approach: use ast to parse and validate
    # For compartment_set, the format is "attr = value [; attr = value ...]"
    # We prefix each bare assignment target with "seg."
    statements = [s.strip() for s in mod.section_configure.split(";") if s.strip()]
    prefixed_parts = []
    all_attrs: set[str] = set()
    for stmt in statements:
        prefixed = "seg." + stmt.strip()
        prefixed_parts.append(prefixed)
    config_str = "; ".join(prefixed_parts)

    # Parse to collect attrs
    collector = _AttributeCollector()
    tree = ast.parse(config_str)
    for elem in tree.body:
        _validate_assignment(elem)
        collector.visit(elem)
    all_attrs = collector.attrs

    n_cells = 0
    n_segments = 0
    for cell_id in cells:
        cell = cells[cell_id]
        node_id = cell_id.id if hasattr(cell_id, "id") else cell_id

        if (
            population_name is not None
            and getattr(cell_id, "population_name", None) != population_name
        ):
            continue

        try:
            resolved = cell.resolve_segments_from_compartment_set(node_id, comp_nodes)
        except (ValueError, TypeError) as e:
            logger.warning(
                "compartment_set '%s': failed to resolve segments for cell %s, skipping: %s",
                mod.name,
                cell_id,
                e,
            )
            continue

        cell_applied = 0
        for section, sec_name, seg_x in resolved:
            segment = section(seg_x)
            if all(hasattr(segment, attr) for attr in all_attrs):
                exec(config_str, {"__builtins__": None}, {"seg": segment})  # noqa: S102
                cell_applied += 1
                logger.debug(
                    "  Applied to segment '%s(%s)' of cell %s", sec_name, seg_x, cell_id
                )
        if cell_applied > 0:
            n_cells += 1
        n_segments += cell_applied

    logger.info(
        "Modification '%s' applied to %d segments across %d cells",
        mod.name,
        n_segments,
        n_cells,
    )
    if n_segments == 0:
        logger.warning(
            "compartment_set '%s' applied to zero segments, "
            "please check its section_configure for possible mistakes",
            mod.name,
        )
