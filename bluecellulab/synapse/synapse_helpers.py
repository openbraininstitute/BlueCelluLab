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
/ cwd).
"""
from __future__ import annotations

import logging
import os

import importlib_resources as resources
import neuron

logger = logging.getLogger(__name__)

_loaded_helpers: set[str] = set()


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

    External helpers found through ``HOC_LIBRARY_PATH`` take precedence over
    the standard helpers bundled with BlueCelluLab. The matching compiled MOD
    mechanism is still required separately.
    """
    helper_name = f"{suffix}Helper"
    if suffix in _loaded_helpers:
        return helper_name

    helper_file = f"{helper_name}.hoc"
    bundled_dir = _ensure_bundled_hoc_directory_on_search_path()

    # Search HOC_LIBRARY_PATH first, preserving support for custom helpers.
    loaded = neuron.h.load_file(helper_file)
    if not loaded:
        bundled_path = os.path.join(bundled_dir, helper_file)
        loaded = neuron.h.load_file(bundled_path)

    if not loaded:
        raise FileNotFoundError(
            f"Could not load HOC helper '{helper_file}' for mod_override '{suffix}'. "
            f"HOC_LIBRARY_PATH={os.environ.get('HOC_LIBRARY_PATH', '<unset>')}"
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
