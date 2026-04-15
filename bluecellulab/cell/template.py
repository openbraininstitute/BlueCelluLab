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
"""Module for handling NEURON hoc templates."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import string
from typing import NamedTuple, Optional
import uuid

import neuron

from bluecellulab.cell.morphio_wrapper import split_morphology_path
from bluecellulab.circuit import EmodelProperties
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.type_aliases import HocObjectType

import logging

logger = logging.getLogger(__name__)


def public_hoc_cell(cell: HocObjectType) -> HocObjectType:
    """Retrieve the hoc cell to access public hoc functions/attributes."""
    if hasattr(cell, "getCell"):
        return cell.getCell()
    elif hasattr(cell, "CellRef"):
        return cell.CellRef
    else:
        raise BluecellulabError("""Public cell properties cannot be accessed
         from the hoc model. Either getCell() or CellRef needs to be provided""")


class TemplateParams(NamedTuple):
    template_filepath: str | Path
    morph_filepath: str | Path
    template_format: str
    emodel_properties: Optional[EmodelProperties]


class NeuronTemplate:
    """Loads and manages a NEURON HOC cell template together with its
    morphology.

    Supports four morphology path formats:

    - ``.asc`` / ``.swc``: individual morphology files.
    - ``.h5`` single file: an individual HDF5 morphology file.
    - ``.h5`` container: a path of the form ``container.h5/cell_name`` where
      ``container.h5`` is an HDF5 morphologies container and
      ``cell_name`` is the key of the morphology inside it
      (with or without a trailing ``.h5`` suffix).
    - v5-style directory: a directory path passed directly to legacy HOC
      templates that locate the morphology file internally.
    """

    def __init__(
        self, template_filepath: str | Path, morph_filepath: str | Path,
        template_format: str, emodel_properties: Optional[EmodelProperties]
    ) -> None:
        """Load the HOC template and validate the morphology path.

        The morphology path is validated with `_is_valid_morphology_path`
        before the template is loaded into NEURON.  Four formats are accepted:

        - ``/dir/cell.asc`` or ``/dir/cell.swc`` — plain morphology file.
        - ``/dir/cell.h5`` — single HDF5 morphology file.
        - ``/dir/merged.h5/cell_name`` — cell inside an HDF5 container
          (``cell_name`` may optionally carry a ``.h5`` suffix as produced by
          `bluecellulab.circuit.circuit_access.SonataCircuitAccess.morph_filepath`).
        - ``/dir/morphologies/`` — directory, used by v5-style templates that
          locate the morphology file themselves.

        Args:
            template_filepath: Path to the ``.hoc`` template file.
            morph_filepath: Path to the morphology.  For H5 containers this is
                ``<container.h5>/<cell_name>``; for v5 templates this is the
                directory that contains the morphology file.
            template_format: One of ``"v5"``, ``"v6"``, or ``"bluepyopt"``.
                Controls how arguments are passed to the HOC constructor.
            emodel_properties: Optional e-model parameters (threshold current,
                holding current, AIS scaler).  Required for ``v6`` templates
                whose HOC defines ``_NeededAttributes``.

        Raises:
            FileNotFoundError: If ``template_filepath`` does not exist on disk
                or if ``morph_filepath`` cannot be resolved to a valid file,
                directory, or HDF5 container entry.
        """
        if isinstance(template_filepath, Path):
            template_filepath = str(template_filepath)
        if isinstance(morph_filepath, Path):
            morph_filepath = str(morph_filepath)

        if not os.path.exists(template_filepath):
            raise FileNotFoundError(f"Couldn't find template file: {template_filepath}")

        # Check morphology path - handle H5 container paths
        if not self._is_valid_morphology_path(morph_filepath):
            raise FileNotFoundError(f"Couldn't find morphology file: {morph_filepath}")

        self.template_name = self.load(template_filepath)
        self.morph_filepath = morph_filepath
        self.template_format = template_format
        self.emodel_properties = emodel_properties

    def get_cell(self, gid: Optional[int]) -> HocObjectType:
        """Returns the hoc object matching the template format."""
        morph_filepath = str(self.morph_filepath)

        # Use split_morphology_path to locate the collection directory.
        # For H5 containers (morph_dir is an .h5 file), morph_fname must end
        # with ".h5" so the HOC extension check routes to morphio_read.
        # We cannot use morph_name + ".h5" here because split_morphology_path
        # uses os.path.splitext, which splits on the LAST dot and therefore
        # mangles cell names that contain dots (e.g. the Scale/Clone suffix).
        # Instead, use os.path.relpath to recover the full bare cell name.
        morph_dir, morph_name, morph_ext = split_morphology_path(morph_filepath)
        if os.path.isfile(morph_dir) and morph_dir.endswith('.h5'):
            bare_name = os.path.relpath(morph_filepath, morph_dir)
            if bare_name.endswith('.h5'):
                bare_name = bare_name[:-3]
            morph_fname = bare_name + '.h5'
        else:
            morph_fname = morph_name + morph_ext

        if self.template_format == "v6":
            attr_names = getattr(
                neuron.h, self.template_name.split('_bluecellulab')[0] + "_NeededAttributes", None
            )
            if attr_names is not None:
                if self.emodel_properties is None:
                    raise BluecellulabError(
                        "EmodelProperties must be provided for template "
                        "format v6 that specifies _NeededAttributes"
                    )
                cell = getattr(neuron.h, self.template_name)(
                    gid,
                    morph_dir,
                    morph_fname,
                    *[self.emodel_properties.__getattribute__(name) for name in attr_names.split(";")]
                )
            else:
                cell = getattr(neuron.h, self.template_name)(
                    gid,
                    morph_dir,
                    morph_fname,
                )
        elif self.template_format == "bluepyopt":
            cell = getattr(neuron.h, self.template_name)(morph_dir, morph_fname)
        else:
            cell = getattr(neuron.h, self.template_name)(gid, self.morph_filepath)

        return cell

    def load(self, template_filename: str) -> str:
        """Read a cell template. If template name already exists, rename it.

        Args:
            template_filename: path string containing template file.

        Returns:
            resulting template name
        """
        with open(template_filename) as template_file:
            template_content = template_file.read()

        match = re.search(r"begintemplate\s*(\S*)", template_content)
        template_name = match.group(1)  # type:ignore

        logger.debug("This Neuron version supports renaming templates, enabling...")
        # add bluecellulab to the template name, so that we don't interfere with
        # templates load outside of bluecellulab
        template_name = "%s_bluecellulab" % template_name
        template_name = get_neuron_compliant_template_name(template_name)
        unique_id = uuid.uuid4().hex
        template_name = f"{template_name}_{unique_id}"

        template_content = re.sub(
            r"begintemplate\s*(\S*)",
            "begintemplate %s" % template_name,
            template_content,
        )
        template_content = re.sub(
            r"endtemplate\s*(\S*)",
            "endtemplate %s" % template_name,
            template_content,
        )

        neuron.h(template_content)

        return template_name

    def _is_valid_morphology_path(self, morph_filepath: str) -> bool:
        """Return True if *morph_filepath* points to a loadable morphology.

        Accepts four cases:

        1. **Regular file** (``.asc``, ``.swc``, ``.h5``): ``os.path.isfile``
           returns True immediately.
        2. **Directory** (v5-style templates): ``os.path.isdir`` returns True
           immediately.
        3. **H5 container** (``container.h5/cell_name``): the path does not
           exist as a file, so the method walks up via ``os.path.dirname``
           until it finds an existing ``.h5`` file.  It then opens that file
           with ``h5py`` and checks that *cell_name* is a top-level key.
           The cell name may carry a trailing ``.h5`` suffix (as appended by
           ``SonataCircuitAccess``); that suffix is stripped before the lookup
           because HDF5 keys are stored without extensions.
        4. **Non-existent path**: returns False if the walk-up reaches the
           filesystem root without finding any existing entry.

        The walk-up strategy mirrors neurodamus ``split_morphology_path`` and
        correctly handles cell names that contain dots (e.g.
        ``cell_x1.000_y0.950_-_Clone_0``) without mis-splitting on the last
        dot as ``os.path.splitext`` would.

        Args:
            morph_filepath: Morphology path string to validate.

        Returns:
            True if the path resolves to a valid morphology, False otherwise.
        """
        # Regular file or directory (v5-style templates receive a directory) — always valid
        if os.path.isfile(morph_filepath) or os.path.isdir(morph_filepath):
            return True

        # Walk up via os.path.dirname (same as neurodamus split_morphology_path)
        candidate = morph_filepath
        while not os.path.exists(candidate):
            parent = os.path.dirname(candidate)
            if parent == candidate:
                # Reached filesystem root without finding anything
                return False
            candidate = parent

        # If candidate is an H5 container file, validate the cell name inside
        if os.path.isfile(candidate) and candidate.endswith('.h5'):
            # Cell name is the relative path after the container.
            # Strip trailing .h5 extension if present: circuit code (e.g.
            # sonata_circuit_access) appends .h5 to match neurodamus
            # convention, but h5py keys are bare names without extensions.
            cell_name = os.path.relpath(morph_filepath, candidate)
            if cell_name.endswith('.h5'):
                cell_name = cell_name[:-3]
            try:
                import h5py
                with h5py.File(candidate, 'r') as f:
                    return cell_name in f
            except Exception:
                return False

        return False


def shorten_and_hash_string(label: str, keep_length=40, hash_length=9) -> str:
    """Converts a string to a shorter string if required.

    Args:
        label: A string to be converted.
        keep_length: Length of the original string to keep.
        hash_length: Length of the hash to generate, should not be more than 20.

    Returns:
        If the length of the original label is shorter than the sum of 'keep_length'
        and 'hash_length' plus one, the original string is returned. Otherwise, a
        string with structure <partial>_<hash> is returned, where <partial> is the
        first part of the original string with length equal to <keep_length> and the
        last part is a hash of 'hash_length' characters, based on the original string.
    """
    if hash_length > 20:
        raise ValueError(
            "Parameter hash_length should not exceed 20, "
            " received: {}".format(hash_length)
        )

    if len(label) <= keep_length + hash_length + 1:
        return label

    hash_string = hashlib.sha1(label.encode("utf-8")).hexdigest()
    return "{}_{}".format(label[0:keep_length], hash_string[0:hash_length])


def check_compliance_with_neuron(template_name: str) -> bool:
    """Verify that a given name is compliant with the rules for a NEURON.

    A name should be a non-empty alphanumeric string, and start with a
    letter. Underscores are allowed. The length should not exceed 50
    characters.
    """
    max_len = 50
    return (
        bool(template_name)
        and template_name[0].isalpha()
        and template_name.replace("_", "").isalnum()
        and len(template_name) <= max_len
    )


def get_neuron_compliant_template_name(name: str) -> str:
    """Get template name that is compliant with NEURON based on given name."""
    template_name = name
    if not check_compliance_with_neuron(template_name):
        template_name = template_name.lstrip(string.digits).replace("-", "_")
        template_name = shorten_and_hash_string(
            template_name, keep_length=40, hash_length=9
        )
        logger.debug("Converted template name %s to %s to make it "
                     "NEURON compliant" % (name, template_name))
    return template_name
