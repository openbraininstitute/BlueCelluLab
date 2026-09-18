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
import logging
import os
from pathlib import Path
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from bluecellulab import importer
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.mod_compilation import ModCompilationError


@patch("os.path.isdir", return_value=False)  # when x86_64 isdir returns False
def test_import_mod_lib_no_env_no_folder(mocked_isdir):
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        assert importer.import_mod_lib(mock_neuron) == "No mechanisms are loaded."


def test_import_mod_lib_env_var_set_folder_exists():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch(
            "os.path.isdir", return_value=True
        ):  # when x86_64 isdir returns True and env var is set
            with pytest.raises(
                BluecellulabError,
                match="BLUECELLULAB_MOD_LIBRARY_PATH is set and current directory contains the x86_64 folder. Please remove one of them.",
            ):
                importer.import_mod_lib(mock_neuron)


def test_import_mod_lib_env_var_set():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch("os.path.isdir", return_value=False):
            assert importer.import_mod_lib(mock_neuron) == "/fake/path"


def test_import_mod_lib_so_file():
    mock_neuron = MagicMock()
    fake_so_path = "/fake/path/to/library.so"
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": fake_so_path}):
        with patch("os.path.isdir", return_value=False):
            importer.import_mod_lib(mock_neuron)
            mock_neuron.h.nrn_load_dll.assert_called_with(fake_so_path)


def test_import_mod_lib_no_env_with_folder():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=True):
            assert importer.import_mod_lib(mock_neuron).endswith("x86_64")


def test_import_mod_lib_mechanisms_dirs_compiles_and_loads():
    """When mechanisms_dirs is given (and no env var), compile + load it."""
    mock_neuron = MagicMock()
    mechanisms_dirs = [Path("/fake/mod")]
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=False):
            with patch(
                "bluecellulab.importer.compile_mechanisms",
                return_value=Path("/fake/build/x86_64/libnrnmech.so"),
            ) as mocked_compile:
                res = importer.import_mod_lib(mock_neuron, mechanisms_dirs)

    mocked_compile.assert_called_once_with(mechanisms_dirs)
    mock_neuron.h.nrn_load_dll.assert_called_with("/fake/build/x86_64/libnrnmech.so")
    assert res == "/fake/build/x86_64/libnrnmech.so"


def test_import_mod_lib_mechanisms_dirs_no_mods_found():
    """compile_mechanisms returning None (no mechanisms_dir mods) is handled."""
    mock_neuron = MagicMock()
    mechanisms_dirs = [Path("/fake/mod")]
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=False):
            with patch("bluecellulab.importer.compile_mechanisms", return_value=None):
                res = importer.import_mod_lib(mock_neuron, mechanisms_dirs)

    assert res == "No mechanisms are loaded."
    mock_neuron.h.nrn_load_dll.assert_not_called()


def test_import_mod_lib_empty_mechanisms_dirs_still_compiles():
    """A SONATA circuit with no `mechanisms_dir` of its own (empty list, not
    None) still triggers compilation, since BlueCelluLab's own bundled mod
    files still need to be compiled/loaded."""
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=False):
            with patch(
                "bluecellulab.importer.compile_mechanisms",
                return_value=Path("/fake/build/x86_64/libnrnmech.so"),
            ) as mocked_compile:
                res = importer.import_mod_lib(mock_neuron, [])

    mocked_compile.assert_called_once_with([])
    assert res == "/fake/build/x86_64/libnrnmech.so"


def test_import_mod_lib_none_mechanisms_dirs_skips_compile_path():
    """`mechanisms_dirs=None` (no SONATA circuit at all) takes the manual
    x86_64/no-mechanisms fallback instead of compiling anything."""
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=False):
            with patch("bluecellulab.importer.compile_mechanisms") as mocked_compile:
                res = importer.import_mod_lib(mock_neuron, None)

    mocked_compile.assert_not_called()
    assert res == "No mechanisms are loaded."


def test_import_mod_lib_mechanisms_dirs_and_x86_64_folder_raises():
    """Circuit-driven mods and a manual x86_64 folder can't coexist."""
    mock_neuron = MagicMock()
    mechanisms_dirs = [Path("/fake/mod")]
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=True):
            with pytest.raises(BluecellulabError, match="SONATA circuit"):
                importer.import_mod_lib(mock_neuron, mechanisms_dirs)


def test_import_mod_lib_mechanisms_dirs_compile_error_wrapped():
    """A ModCompilationError from compilation is wrapped as BluecellulabError."""
    mock_neuron = MagicMock()
    mechanisms_dirs = [Path("/fake/mod")]
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=False):
            with patch(
                "bluecellulab.importer.compile_mechanisms",
                side_effect=ModCompilationError("boom"),
            ):
                with pytest.raises(BluecellulabError, match="Failed to compile circuit mod files"):
                    importer.import_mod_lib(mock_neuron, mechanisms_dirs)


def test_import_mod_lib_env_var_takes_precedence_over_mechanisms_dirs():
    """The env var override wins even if mechanisms_dirs is also given."""
    mock_neuron = MagicMock()
    mechanisms_dirs = [Path("/fake/mod")]
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch("os.path.isdir", return_value=False):
            with patch("bluecellulab.importer.compile_mechanisms") as mocked_compile:
                res = importer.import_mod_lib(mock_neuron, mechanisms_dirs)

    mocked_compile.assert_not_called()
    assert res == "/fake/path"


@patch.object(
    importer.resources,
    "files",
    return_value=Path("/fake/path/"),
)
def test_import_neurodamus(mocked_resources):
    mock_neuron = MagicMock()
    importer.import_hoc(mock_neuron)
    assert mock_neuron.h.load_file.called
    # Check that it was called with the expected arguments
    mock_neuron.h.load_file.assert_any_call("/fake/path/hoc/Cell.hoc")


def test_print_header(caplog):
    # Creating a dummy ModuleType object with an attribute '__file__'
    dummy_neuron = ModuleType("dummy_neuron")
    dummy_neuron.__file__ = "/path/to/neuron"

    mod_lib_path = "/path/to/mod_lib"
    with caplog.at_level(logging.DEBUG):
        importer.print_header(dummy_neuron, mod_lib_path)

    assert "Imported NEURON from: /path/to/neuron" in caplog.text
    assert "Mod lib: /path/to/mod_lib" in caplog.text


def test_print_header_with_decorator(caplog):
    """Ensure the decorator loading mod files work as expected."""
    with caplog.at_level(logging.DEBUG):
        @importer.load_mod_files
        def x():
            pass

        x()  # call 3 times to ensure the decorator is called only once
        x()
        x()

    assert caplog.text.count("Loading the mod files.") == 1


class TestLoadModFilesForCircuit:
    """Tests for `load_mod_files_for_circuit`, isolating the module-level
    `run_once` state of `_load_mod_files` between tests."""

    def setup_method(self):
        importer._load_mod_files.has_run = False
        importer._load_mod_files.loaded_with = None

    def teardown_method(self):
        importer._load_mod_files.has_run = False
        importer._load_mod_files.loaded_with = None

    def test_first_call_loads_with_given_dirs(self):
        mechanisms_dirs = [Path("/fake/mod")]
        with patch("bluecellulab.importer.import_mod_lib", return_value="loaded") as mocked:
            importer.load_mod_files_for_circuit(mechanisms_dirs)

        mocked.assert_called_once_with(importer.neuron, mechanisms_dirs)
        assert importer._load_mod_files.loaded_with == mechanisms_dirs

    def test_second_call_with_same_dirs_is_noop_and_does_not_warn(self, caplog):
        mechanisms_dirs = [Path("/fake/mod")]
        with patch("bluecellulab.importer.import_mod_lib", return_value="loaded") as mocked:
            importer.load_mod_files_for_circuit(mechanisms_dirs)
            with caplog.at_level(logging.WARNING):
                importer.load_mod_files_for_circuit(mechanisms_dirs)

        assert mocked.call_count == 1
        assert "cannot also be loaded" not in caplog.text

    def test_second_call_with_different_dirs_warns(self, caplog):
        first_dirs = [Path("/fake/mod_a")]
        second_dirs = [Path("/fake/mod_b")]
        with patch("bluecellulab.importer.import_mod_lib", return_value="loaded") as mocked:
            importer.load_mod_files_for_circuit(first_dirs)
            with caplog.at_level(logging.WARNING):
                importer.load_mod_files_for_circuit(second_dirs)

        # underlying loader still only runs once (NEURON constraint)
        assert mocked.call_count == 1
        assert "cannot also be loaded" in caplog.text


class TestLegacyMorphioWrapperAlias:
    """``importer._register_legacy_morphio_wrapper_alias``.

    Emodel hoc templates generated by older BluePyOpt/neurodamus pipelines load
    the H5 morphology reader with a bare ``from morphio_wrapper import
    MorphIOWrapper``. bluecellulab ships it as
    ``bluecellulab.cell.morphio_wrapper``, so without the alias such templates
    print ".h5 morphlogy used but cannot load 'morphio_wrapper'." and call
    ``quit()``, terminating the whole process.
    """

    def test_alias_is_registered(self):
        import sys

        from bluecellulab.cell import morphio_wrapper

        # importing bluecellulab (done at module import) runs import_hoc,
        # which registers the alias
        assert sys.modules["morphio_wrapper"] is morphio_wrapper

    def test_bare_import_works(self):
        """This is exactly what the hoc template's nrnpython call does."""
        from morphio_wrapper import MorphIOWrapper  # NOQA

        assert MorphIOWrapper.__module__ == "bluecellulab.cell.morphio_wrapper"

    def test_registration_is_idempotent(self):
        import sys

        from bluecellulab.cell import morphio_wrapper

        importer._register_legacy_morphio_wrapper_alias()
        importer._register_legacy_morphio_wrapper_alias()
        assert sys.modules["morphio_wrapper"] is morphio_wrapper

    def test_existing_third_party_module_takes_precedence(self):
        """``setdefault`` must not clobber a real top-level module."""
        import sys

        sentinel = ModuleType("morphio_wrapper")
        original = sys.modules.get("morphio_wrapper")
        sys.modules["morphio_wrapper"] = sentinel
        try:
            importer._register_legacy_morphio_wrapper_alias()
            assert sys.modules["morphio_wrapper"] is sentinel
        finally:
            if original is not None:
                sys.modules["morphio_wrapper"] = original
            else:
                del sys.modules["morphio_wrapper"]

    def test_import_hoc_registers_the_alias(self):
        """The alias must be installed before any hoc template is loaded."""
        import sys

        original = sys.modules.pop("morphio_wrapper", None)
        try:
            with patch.object(
                importer.resources, "files", return_value=Path("/fake/path/")
            ):
                importer.import_hoc(MagicMock())
            assert "morphio_wrapper" in sys.modules
        finally:
            if original is not None:
                sys.modules["morphio_wrapper"] = original
