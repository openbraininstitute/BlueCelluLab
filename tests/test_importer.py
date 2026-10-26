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
import platform
from pathlib import Path
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from bluecellulab import importer
from bluecellulab.exceptions import BluecellulabError


def test_import_mod_lib_no_env_nothing_to_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {}, clear=True):
        assert importer.import_mod_lib(MagicMock()) == "No mechanisms are loaded."


def test_import_mod_lib_reports_arch_dir(tmp_path, monkeypatch):
    """NEURON auto-loads ``./<arch>``; nothing is compiled or loaded here."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / platform.machine()).mkdir()
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        res = importer.import_mod_lib(mock_neuron)
    assert res == str((tmp_path / platform.machine()).absolute())
    mock_neuron.h.nrn_load_dll.assert_not_called()


def test_import_mod_lib_warns_on_changed_mod_files(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with patch(
        "bluecellulab.importer.changed_mod_files", return_value=[Path("/src/a.mod")]
    ):
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                importer.import_mod_lib(MagicMock())
    assert "MOD files changed since" in caplog.text
    assert "a.mod" in caplog.text


@pytest.mark.parametrize("arch", ["x86_64", "arm64"])
def test_import_mod_lib_env_var_set_folder_exists(arch, tmp_path, monkeypatch):
    mock_neuron = MagicMock()
    monkeypatch.chdir(tmp_path)
    (tmp_path / arch).mkdir()
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch("platform.machine", return_value=arch):
            with pytest.raises(
                BluecellulabError,
                match=f"BLUECELLULAB_MOD_LIBRARY_PATH is set and current directory contains the {arch} folder. Please remove one of them.",
            ):
                importer.import_mod_lib(mock_neuron)


def test_import_mod_lib_env_var_set(tmp_path, monkeypatch):
    mock_neuron = MagicMock()
    monkeypatch.chdir(tmp_path)
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        assert importer.import_mod_lib(mock_neuron) == "/fake/path"


@pytest.mark.parametrize(
    "lib_path,expected",
    [
        ("/opt/mech/arm64/libnrnmech.so", Path("/opt/mech")),
        ("/opt/mech/x86_64", Path("/opt/mech")),
        ("/opt/mech", Path("/opt/mech")),
    ],
)
def test_mod_lib_output_dir(lib_path, expected, monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert importer._mod_lib_output_dir(lib_path) == expected


def test_import_mod_lib_env_var_warns_on_changed_sources(tmp_path, monkeypatch):
    """A library loaded via the env var is checked against its own build dir."""
    mock_neuron = MagicMock()
    monkeypatch.chdir(tmp_path)
    lib_dir = tmp_path / "opt" / "mech" / "x86_64"
    lib_dir.mkdir(parents=True)
    so = lib_dir / "libnrnmech.so"
    so.write_text("")
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": str(so)}):
        with patch("platform.machine", return_value="x86_64"):
            with patch(
                "bluecellulab.importer.changed_mod_files", return_value=[]
            ) as changed:
                importer.import_mod_lib(mock_neuron)
    changed.assert_called_once_with(tmp_path / "opt" / "mech")


def test_import_mod_lib_so_file(tmp_path, monkeypatch):
    mock_neuron = MagicMock()
    monkeypatch.chdir(tmp_path)
    fake_so_path = "/fake/path/to/library.so"
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": fake_so_path}):
        importer.import_mod_lib(mock_neuron)
        mock_neuron.h.nrn_load_dll.assert_called_with(fake_so_path)


class TestCheckCircuitMechanisms:
    def _circuit_dir(self, tmp_path):
        d = tmp_path / "mod"
        d.mkdir()
        (d / "Chan.mod").write_text("NEURON {\n\tSUFFIX Chan\n}\n")
        return d

    def test_warns_with_command_when_missing(self, tmp_path, caplog):
        d = self._circuit_dir(tmp_path)
        with patch("bluecellulab.importer.registered_mechanisms", return_value={"pas"}):
            with caplog.at_level(logging.WARNING):
                importer.check_circuit_mechanisms([d])
        assert "Chan" in caplog.text
        assert "bluecellulab compile-mods" in caplog.text

    def test_silent_when_loaded(self, tmp_path, caplog):
        d = self._circuit_dir(tmp_path)
        with patch("bluecellulab.importer.registered_mechanisms", return_value={"Chan"}):
            with caplog.at_level(logging.WARNING):
                importer.check_circuit_mechanisms([d])
        assert caplog.text == ""

    def test_warns_on_missing_dir(self, tmp_path, caplog):
        with patch("bluecellulab.importer.registered_mechanisms", return_value=set()):
            with caplog.at_level(logging.WARNING):
                importer.check_circuit_mechanisms([tmp_path / "gone"])
        assert "does not exist" in caplog.text
        assert "gone" in caplog.text

    @pytest.mark.parametrize("dirs", [None, []])
    def test_no_dirs_is_noop(self, dirs):
        with patch("bluecellulab.importer.registered_mechanisms") as mocked:
            importer.check_circuit_mechanisms(dirs)
        mocked.assert_not_called()


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
    def setup_method(self):
        importer._load_mod_files.has_run = False

    def teardown_method(self):
        importer._load_mod_files.has_run = False

    def test_loads_once_and_checks_each_time(self):
        dirs = [Path("/fake/mod")]
        with patch("bluecellulab.importer.import_mod_lib", return_value="x") as load:
            with patch("bluecellulab.importer.check_circuit_mechanisms") as check:
                importer.load_mod_files_for_circuit(dirs)
                importer.load_mod_files_for_circuit(dirs)
        load.assert_called_once_with(importer.neuron)
        assert check.call_count == 2
        check.assert_called_with(dirs)


def test_load_mod_files_retries_after_library_failure(monkeypatch):
    monkeypatch.setattr(importer._load_mod_files, "has_run", False)
    with patch.object(importer, "import_mod_lib", side_effect=[RuntimeError("failed"), "lib"]) as load:
        with pytest.raises(RuntimeError, match="failed"):
            importer._load_mod_files()
        importer._load_mod_files()
        importer._load_mod_files()
    assert load.call_count == 2


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

        # initialize_neuron() runs import_hoc, which registers the alias
        importer.initialize_neuron()
        assert sys.modules["morphio_wrapper"] is morphio_wrapper

    def test_bare_import_works(self):
        """This is exactly what the hoc template's nrnpython call does."""
        importer.initialize_neuron()
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
