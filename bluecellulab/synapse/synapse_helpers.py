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
once per prefix. Circuits that ship their own helper HOCs pass their
directories as ``extra_dirs`` (no process-global search state).
"""
from __future__ import annotations

import contextlib
import logging
import os

from collections.abc import Iterable, Iterator
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
# Process-global because HOC templates are: a template cannot be redefined.
_loaded_helpers: dict[str, str] = {}

# (prefix, extra_dirs) pairs already checked against _loaded_helpers.
_checked_requests: set[tuple[str, tuple[str, ...]]] = set()


def _bundled_hoc_directory() -> str:
    return str(resources.files("bluecellulab").joinpath("hoc"))


@contextlib.contextmanager
def _bundled_dir_on_hoc_library_path() -> Iterator[None]:
    """Append the bundled dir to ``HOC_LIBRARY_PATH`` while loading a helper.

    Helpers load dependencies by name (``load_file("RNGSettings.hoc")``),
    which NEURON resolves via ``HOC_LIBRARY_PATH`` at call time. The
    variable is restored afterwards so the change does not leak to the
    process or its children.
    """
    previous = os.environ.get("HOC_LIBRARY_PATH")
    paths = [path for path in (previous or "").split(os.pathsep) if path]
    os.environ["HOC_LIBRARY_PATH"] = os.pathsep.join([*paths, _bundled_hoc_directory()])
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HOC_LIBRARY_PATH", None)
        else:
            os.environ["HOC_LIBRARY_PATH"] = previous


def _helper_search_dirs(extra_dirs: Iterable[str | os.PathLike] = ()) -> list[str]:
    """Return the helper search directories in precedence order.

    Order: cwd, user ``HOC_LIBRARY_PATH`` entries (the bundled dir
    excluded), ``extra_dirs`` (circuit-provided, existing ones only), then
    the bundled dir.
    """
    bundled_dir = os.path.normpath(_bundled_hoc_directory())
    user_dirs = [
        path for path in os.environ.get("HOC_LIBRARY_PATH", "").split(os.pathsep)
        if path and os.path.normpath(path) != bundled_dir
    ]
    circuit_dirs = [os.fspath(d) for d in extra_dirs if os.path.isdir(d)]
    return [os.getcwd(), *user_dirs, *circuit_dirs, bundled_dir]


def _resolve_helper_path(
    suffix: str, extra_dirs: Iterable[str | os.PathLike] = ()
) -> str | None:
    """Return the absolute path of the first ``<suffix>Helper.hoc`` found.

    See :func:`_helper_search_dirs` for the precedence. Returns None if no
    directory contains the helper file.
    """
    helper_file = f"{suffix}Helper.hoc"
    for directory in _helper_search_dirs(extra_dirs):
        candidate = os.path.join(directory, helper_file)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _warn_if_other_path(suffix: str, extra_dirs: tuple[str, ...]) -> None:
    """Warn once per request if ``suffix`` would now resolve to a different
    file than the one already loaded (HOC cannot reload a template)."""
    request = (suffix, extra_dirs)
    if request in _checked_requests:
        return
    _checked_requests.add(request)
    loaded = _loaded_helpers[suffix]
    if loaded == "<preloaded>":
        return
    resolved = _resolve_helper_path(suffix, extra_dirs)
    if resolved is not None and resolved != loaded:
        logger.warning(
            "modoverride '%s' resolves to '%s', but '%sHelper' is already "
            "defined from '%s'; HOC templates cannot be reloaded, keeping it.",
            suffix, resolved, suffix, loaded,
        )


def load_synapse_helper(
    suffix: str, extra_dirs: Iterable[str | os.PathLike] = ()
) -> str:
    """Load the ``<suffix>Helper`` HOC template and return its name.

    ``suffix`` is the ``modoverride`` helper prefix. The helper file is
    resolved by :func:`_resolve_helper_path` (cwd, user ``HOC_LIBRARY_PATH``,
    ``extra_dirs`` of the circuit, bundled helpers) and loaded by absolute
    path. Each prefix is loaded at most once; if the template already
    exists in NEURON it is not loaded again (HOC cannot redefine a
    template), and a warning is logged if ``extra_dirs`` would select a
    different file. The bundled dir is on ``HOC_LIBRARY_PATH`` only while
    loading, so helpers can load dependencies such as ``RNGSettings.hoc``.
    The matching compiled MOD mechanism is still required separately.
    """
    helper_name = f"{suffix}Helper"
    dirs = tuple(os.fspath(d) for d in extra_dirs)
    if suffix in _loaded_helpers:
        _warn_if_other_path(suffix, dirs)
        return helper_name

    if hasattr(neuron.h, helper_name):
        _loaded_helpers[suffix] = "<preloaded>"
        return helper_name

    helper_file = f"{helper_name}.hoc"
    helper_path = _resolve_helper_path(suffix, dirs)
    if helper_path is None:
        raise FileNotFoundError(
            f"Could not find HOC helper '{helper_file}' for modoverride '{suffix}'. "
            f"modoverride is a helper prefix resolved to '<modoverride>Helper.hoc' "
            f"(bundled: AMPANMDA, GABAAB, GluSynapse, Exp2Syn). "
            f"Searched: {_helper_search_dirs(dirs)}"
        )
    with _bundled_dir_on_hoc_library_path():
        loaded = neuron.h.load_file(helper_path)
    if not loaded:
        raise FileNotFoundError(
            f"NEURON failed to load HOC helper '{helper_path}' for modoverride '{suffix}'."
        )
    if not hasattr(neuron.h, helper_name):
        raise AttributeError(
            f"HOC helper '{helper_path}' did not define template '{helper_name}'."
        )
    _loaded_helpers[suffix] = helper_path
    _checked_requests.add((suffix, dirs))
    logger.debug("Loaded synapse helper %s", helper_path)
    return helper_name


_warned_rng_modes: set[str] = set()


def warn_if_not_random123(mode: str) -> None:
    """Warn once per mode: helpers always seed Random123-style.

    Neurodamus supports only Random123. In ``Compatibility`` or
    ``UpdatedMCell`` mode, native synapses use MCellRan4 while helper
    synapses still use Random123 streams.
    """
    if mode == "Random123" or mode in _warned_rng_modes:
        return
    _warned_rng_modes.add(mode)
    logger.warning(
        "RNG mode '%s': modoverride helper synapses always use Random123 "
        "seeding (as neurodamus), unlike native synapses in this mode.", mode,
    )


def helper_loaded_from(suffix: str) -> str | None:
    """Return the file the ``<suffix>Helper`` template was loaded from.

    ``"<preloaded>"`` if it was already defined in NEURON, None if not
    loaded by BlueCelluLab.
    """
    return _loaded_helpers.get(suffix)


def helper_available(suffix: str) -> bool:
    """Return True if a helper template is already loaded for the SUFFIX."""
    return hasattr(neuron.h, f"{suffix}Helper")


def _is_missing(value: Any) -> bool:
    """Return True for None or a scalar NaN (column absent after the outer join
    of edge populations)."""
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


def get_helper_needed_attributes(
    suffix: str, extra_dirs: Iterable[str | os.PathLike] = ()
) -> list[str]:
    """Read the ``_NeededAttributes`` metadata from a loaded helper HOC.

    Neurodamus helper HOCs declare a semicolon-separated global string
    ``<SUFFIX>Helper_NeededAttributes`` listing the SONATA edge fields
    the helper requires. This function loads the helper (if not already
    loaded) and returns those field names.

    Args:
        suffix: ``modoverride`` helper prefix (e.g. ``"AMPANMDA"``).
        extra_dirs: circuit directories searched for the helper.

    Returns:
        List of attribute names, or an empty list if the helper does not
        declare ``_NeededAttributes``.
    """
    return _helper_metadata_list(suffix, "NeededAttributes", extra_dirs)


def get_helper_uhill_scale_vars(
    suffix: str, extra_dirs: Iterable[str | os.PathLike] = ()
) -> list[str]:
    """Read the ``_UHillScaleVariables`` metadata from a loaded helper HOC.

    As in neurodamus ``SynapseReader.configure_override``, these fields
    are scaled by the same constrained Hill factor as ``U``.

    Returns:
        List of field names, or an empty list if not declared.
    """
    return _helper_metadata_list(suffix, "UHillScaleVariables", extra_dirs)


def _helper_metadata_list(
    suffix: str, key: str, extra_dirs: Iterable[str | os.PathLike] = ()
) -> list[str]:
    """Return the ``<suffix>Helper_<key>`` semicolon-separated list."""
    helper_name = load_synapse_helper(suffix, extra_dirs)
    attr_str = getattr(neuron.h, f"{helper_name}_{key}", None)
    if attr_str:
        return [a.strip() for a in attr_str.split(";") if a.strip()]
    return []
