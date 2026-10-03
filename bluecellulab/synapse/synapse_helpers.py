# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Neurodamus-compatible synapse helper HOC loading.

Implements the same convention as ``neurodamus``: a ``modoverride`` value is
a helper prefix and always resolves to the HOC template ``"{prefix}Helper"``
defined in ``"{prefix}Helper.hoc"`` (no aliases). The helper file is resolved
explicitly (see :func:`_resolve_helper_path`) and loaded by absolute path,
once per prefix. Circuits that ship their own helper HOCs can register extra
search directories via :func:`register_helper_search_dirs`.
"""
from __future__ import annotations

import logging
import os

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import importlib_resources as resources
import neuron
import pandas as pd

from bluecellulab.circuit.synapse_properties import (
    ND_BASE_FIELDS,
    ND_NAMES,
    ND_OPTIONAL_FIELDS,
    ND_RESERVED_FIELDS,
    SynapseProperty,
)
from bluecellulab.exceptions import BluecellulabError

logger = logging.getLogger(__name__)

# Helper prefix -> resolved helper file path (or "<preloaded>" when the
# template was already defined in NEURON before BlueCelluLab loaded it).
_loaded_helpers: dict[str, str] = {}

# Circuit-provided directories searched for ``<SUFFIX>Helper.hoc`` after
# ``HOC_LIBRARY_PATH`` and before the bundled fallback (e.g. a circuit's
# ``mechanisms_dir`` or ``biophysical_neuron_models_dir``).
_extra_search_dirs: list[str] = []


def register_helper_search_dirs(dirs: Iterable[str | os.PathLike]) -> None:
    """Register circuit dirs searched for ``<SUFFIX>Helper.hoc``.

    Registered directories are searched after ``HOC_LIBRARY_PATH`` and before
    the bundled fallback. De-duplicates; ignores non-existent directories.
    The list is process-global; one circuit should be loaded per process.
    """
    for d in dirs:
        path = os.fspath(d)
        if os.path.isdir(path) and path not in _extra_search_dirs:
            _extra_search_dirs.append(path)


def clear_helper_search_dirs() -> None:
    """Clear all registered helper search directories (for tests)."""
    _extra_search_dirs.clear()


def _bundled_hoc_directory() -> str:
    return str(resources.files("bluecellulab").joinpath("hoc"))


def _ensure_bundled_hoc_directory_on_search_path() -> str:
    """Make bundled helper dependencies available to relative HOC loads."""
    bundled_dir = _bundled_hoc_directory()
    current_paths = os.environ.get("HOC_LIBRARY_PATH", "").split(os.pathsep)
    current_paths = [path for path in current_paths if path]
    if bundled_dir not in current_paths:
        current_paths.append(bundled_dir)
        os.environ["HOC_LIBRARY_PATH"] = os.pathsep.join(current_paths)
    return bundled_dir


def _helper_search_dirs() -> list[str]:
    """Return the helper search directories in precedence order.

    Order: cwd, user ``HOC_LIBRARY_PATH`` entries (the bundled dir excluded,
    since BlueCelluLab appends it for dependencies), registered circuit dirs,
    then the bundled dir.
    """
    bundled_dir = os.path.normpath(_bundled_hoc_directory())
    user_dirs = [
        path for path in os.environ.get("HOC_LIBRARY_PATH", "").split(os.pathsep)
        if path and os.path.normpath(path) != bundled_dir
    ]
    return [os.getcwd(), *user_dirs, *_extra_search_dirs, bundled_dir]


def _resolve_helper_path(suffix: str) -> str | None:
    """Return the absolute path of the first ``<suffix>Helper.hoc`` found.

    See :func:`_helper_search_dirs` for the precedence. Returns None if no
    directory contains the helper file.
    """
    helper_file = f"{suffix}Helper.hoc"
    for directory in _helper_search_dirs():
        candidate = os.path.join(directory, helper_file)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_synapse_helper(suffix: str) -> str:
    """Load the ``<suffix>Helper`` HOC template and return its name.

    ``suffix`` is the ``modoverride`` helper prefix. The helper file is
    resolved by :func:`_resolve_helper_path` (cwd, user ``HOC_LIBRARY_PATH``,
    registered circuit dirs, bundled helpers) and loaded by absolute path.
    Each prefix is loaded at most once; if the template already exists in
    NEURON it is not loaded again (HOC cannot redefine a template). The
    bundled dir stays on ``HOC_LIBRARY_PATH`` so helpers can load
    dependencies such as ``RNGSettings.hoc``. The matching compiled MOD
    mechanism is still required separately.
    """
    helper_name = f"{suffix}Helper"
    if suffix in _loaded_helpers:
        return helper_name

    _ensure_bundled_hoc_directory_on_search_path()

    if hasattr(neuron.h, helper_name):
        _loaded_helpers[suffix] = "<preloaded>"
        return helper_name

    helper_file = f"{helper_name}.hoc"
    helper_path = _resolve_helper_path(suffix)
    if helper_path is None:
        raise FileNotFoundError(
            f"Could not find HOC helper '{helper_file}' for modoverride '{suffix}'. "
            f"modoverride is a helper prefix resolved to '<modoverride>Helper.hoc' "
            f"(bundled: AMPANMDA, GABAAB, GluSynapse, Exp2Syn). "
            f"Searched: {_helper_search_dirs()}"
        )
    if not neuron.h.load_file(helper_path):
        raise FileNotFoundError(
            f"NEURON failed to load HOC helper '{helper_path}' for modoverride '{suffix}'."
        )
    if not hasattr(neuron.h, helper_name):
        raise AttributeError(
            f"HOC helper '{helper_path}' did not define template '{helper_name}'."
        )
    _loaded_helpers[suffix] = helper_path
    logger.debug("Loaded synapse helper %s", helper_path)
    return helper_name


def helper_available(suffix: str) -> bool:
    """Return True if a helper template is already loaded for the SUFFIX."""
    return hasattr(neuron.h, f"{suffix}Helper")


def _is_missing(value: Any) -> bool:
    """Return True for None or a scalar NaN (column absent after the outer
    join of edge populations)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def build_helper_params(
    syn_description: pd.Series,
    needed: Iterable[str],
    helper_name: str = "helper",
    synapse_label: str = "",
    scale_vars: Iterable[str] = (),
) -> SimpleNamespace:
    """Build the parameter object passed to a ``<prefix>Helper`` template.

    Mirrors neurodamus ``SynapseParameters.make_synapse_parameters_array``:
    standard fields use neurodamus names (``weight``, ``U``, ``nrrp``, ...),
    reserved fields (``maskValue = -1``, ``location = 0.5``) are never read
    from the edges, optional fields get neurodamus defaults, and extra
    fields declared in ``needed`` are passed under their raw SONATA name.
    Fields in ``scale_vars`` (the helper's ``_UHillScaleVariables``) are
    multiplied by ``syn_description["u_scale_factor"]``, the constrained
    Hill factor already applied to ``U`` (neurodamus
    ``_patch_scale_U_param``).

    Raises:
        BluecellulabError: if a field listed in ``needed`` is missing.
    """
    params = SimpleNamespace(**ND_RESERVED_FIELDS)
    for name, prop in ND_BASE_FIELDS.items():
        value = syn_description.get(prop)
        if _is_missing(value):
            if name in ND_OPTIONAL_FIELDS:
                setattr(params, name, ND_OPTIONAL_FIELDS[name])
            continue
        setattr(params, name, value)

    # Synapse position, as neurodamus SonataReader._load_params_custom:
    # afferent_section_pos takes precedence (ipt = -1, offset = pos).
    if not _is_missing(syn_description.get(SynapseProperty.POST_SECTION_ID)):
        params.isec = syn_description[SynapseProperty.POST_SECTION_ID]
    section_pos = syn_description.get(SynapseProperty.AFFERENT_SECTION_POS)
    if not _is_missing(section_pos):
        params.ipt, params.offset = -1, section_pos
    else:
        for name, prop in (("ipt", SynapseProperty.POST_SEGMENT_ID),
                           ("offset", SynapseProperty.POST_SEGMENT_OFFSET)):
            value = syn_description.get(prop)
            if not _is_missing(value):
                setattr(params, name, value)

    needed = list(needed)
    for name in needed:
        if name not in ND_NAMES and not _is_missing(syn_description.get(name)):
            setattr(params, name, syn_description[name])

    missing = [name for name in needed if _is_missing(getattr(params, name, None))]
    if missing:
        raise BluecellulabError(
            f"Helper '{helper_name}' needs attribute(s) {missing} missing for "
            f"synapse {synapse_label}. The edge population must provide every "
            f"attribute in {helper_name}_NeededAttributes."
        )

    u_scale_factor = syn_description.get("u_scale_factor", 1.0)
    for name in scale_vars:
        if hasattr(params, name):
            setattr(params, name, getattr(params, name) * u_scale_factor)
    return params


def get_helper_needed_attributes(suffix: str) -> list[str]:
    """Read the ``_NeededAttributes`` metadata from a loaded helper HOC.

    Neurodamus helper HOCs declare a semicolon-separated global string
    ``<SUFFIX>Helper_NeededAttributes`` listing the SONATA edge fields
    the helper requires. This function loads the helper (if not already
    loaded) and returns those field names.

    Args:
        suffix: NMODL SUFFIX of the mechanism (e.g. ``"GluSynapse"``).

    Returns:
        List of attribute names, or an empty list if the helper does not
        declare ``_NeededAttributes``.
    """
    return _helper_metadata_list(suffix, "NeededAttributes")


def get_helper_uhill_scale_vars(suffix: str) -> list[str]:
    """Read the ``_UHillScaleVariables`` metadata from a loaded helper HOC.

    As in neurodamus ``SynapseReader.configure_override``, these fields
    are scaled by the same constrained Hill factor as ``U``.

    Returns:
        List of field names, or an empty list if not declared.
    """
    return _helper_metadata_list(suffix, "UHillScaleVariables")


def _helper_metadata_list(suffix: str, key: str) -> list[str]:
    """Return the ``<suffix>Helper_<key>`` semicolon-separated list."""
    helper_name = load_synapse_helper(suffix)
    attr_str = getattr(neuron.h, f"{helper_name}_{key}", None)
    if attr_str:
        return [a.strip() for a in attr_str.split(";") if a.strip()]
    return []
