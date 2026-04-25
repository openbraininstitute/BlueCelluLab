# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Neurodamus-compatible synapse helper HOC loading.

Implements the same convention as `neurodamus`:
- A SUFFIX (e.g. ``"GluSynapse"``) names a compiled NMODL mechanism.
- The companion HOC template is ``"{SUFFIX}Helper"``, expected to be
  loadable via ``h.load_file("{SUFFIX}Helper.hoc")`` (NEURON searches
  ``HOC_LIBRARY_PATH`` / cwd).
"""
from __future__ import annotations

import logging
import os

import neuron

logger = logging.getLogger(__name__)

_loaded_helpers: set[str] = set()


def load_synapse_helper(suffix: str) -> str:
    """Load the helper HOC template for a given mechanism SUFFIX.

    Args:
        suffix: NMODL SUFFIX of the mechanism (e.g. ``"GluSynapse"``).

    Returns:
        The helper template name (``"{suffix}Helper"``).

    Raises:
        FileNotFoundError: if the helper HOC cannot be loaded.
        AttributeError: if the helper template is missing after load.
    """
    helper_name = f"{suffix}Helper"
    if suffix in _loaded_helpers:
        return helper_name

    helper_file = f"{helper_name}.hoc"
    loaded = neuron.h.load_file(helper_file)
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
