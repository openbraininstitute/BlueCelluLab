# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Neurodamus-compatible synapse helper HOC loading.

Implements the same convention as ``neurodamus``. A SUFFIX (e.g.
``"GluSynapse"``) names a compiled NMODL mechanism. The companion HOC
template is ``"{SUFFIX}Helper"``, expected to be loadable via
``h.load_file("{SUFFIX}Helper.hoc")`` (NEURON searches ``HOC_LIBRARY_PATH``
/ cwd). Circuits that ship their own helper HOCs can register extra search
directories via :func:`register_helper_search_dirs`.
"""
from __future__ import annotations

import logging
import os

from collections.abc import Iterable

import importlib_resources as resources
import neuron

logger = logging.getLogger(__name__)

_loaded_helpers: set[str] = set()

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


def load_synapse_helper(suffix: str) -> str:
    """Load a packaged or externally supplied synapse helper HOC template.

    Resolution order: (a) ``HOC_LIBRARY_PATH`` / cwd — external helpers take
    precedence, (b) each directory registered via
    :func:`register_helper_search_dirs` (e.g. the circuit's ``mechanisms_dir``
    or ``biophysical_neuron_models_dir``), (c) the standard helpers bundled
    with BlueCelluLab. The matching compiled MOD mechanism is still required
    separately.
    """
    helper_name = f"{suffix}Helper"
    if suffix in _loaded_helpers:
        return helper_name

    helper_file = f"{helper_name}.hoc"
    # Keep this first: helper HOCs (including circuit-provided ones) load
    # dependencies such as "RNGSettings.hoc" relatively from the bundled dir.
    bundled_dir = _ensure_bundled_hoc_directory_on_search_path()

    # Search HOC_LIBRARY_PATH first, preserving support for custom helpers.
    loaded = neuron.h.load_file(helper_file)
    if not loaded:
        for extra_dir in _extra_search_dirs:
            candidate = os.path.join(extra_dir, helper_file)
            if os.path.isfile(candidate):
                loaded = neuron.h.load_file(candidate)
                if loaded:
                    break
    if not loaded:
        bundled_path = os.path.join(bundled_dir, helper_file)
        loaded = neuron.h.load_file(bundled_path)

    if not loaded:
        raise FileNotFoundError(
            f"Could not load HOC helper '{helper_file}' for mod_override '{suffix}'. "
            f"HOC_LIBRARY_PATH={os.environ.get('HOC_LIBRARY_PATH', '<unset>')}; "
            f"registered search dirs={_extra_search_dirs or '<none>'}"
        )
    if not hasattr(neuron.h, helper_name):
        raise AttributeError(
            f"HOC helper '{helper_file}' did not define template '{helper_name}'."
        )
    _loaded_helpers.add(suffix)
    logger.debug("Loaded synapse helper %s", helper_file)
    return helper_name


def helper_available(suffix: str) -> bool:
    """Return True if a helper template is already loaded for the SUFFIX."""
    return hasattr(neuron.h, f"{suffix}Helper")


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
    helper_name = load_synapse_helper(suffix)
    attr_str = getattr(neuron.h, f"{helper_name}_NeededAttributes", None)
    if attr_str:
        return [a.strip() for a in attr_str.split(";") if a.strip()]
    return []
