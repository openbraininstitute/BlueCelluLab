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
"""Tests for bluecellulab.mod_compilation."""

import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bluecellulab import mod_compilation as mc

Options = mc.Options


def write(path, contents):
    path.parent.mkdir(exist_ok=True, parents=True)
    if isinstance(contents, str):
        contents = contents.encode()
    path.write_bytes(contents)
    return path


def test_default_mod_build_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "custom"))
    assert mc.default_mod_build_dir([]) == (tmp_path / "custom").absolute()


def test_default_mod_build_dir_deterministic(monkeypatch):
    monkeypatch.delenv(mc.MOD_BUILD_DIR_ENV_VAR, raising=False)
    d1 = mc.default_mod_build_dir([Path("/a"), Path("/b")])
    d2 = mc.default_mod_build_dir([Path("/b"), Path("/a")])
    d3 = mc.default_mod_build_dir([Path("/a"), Path("/c")])
    assert d1 == d2
    assert d1 != d3


def test__metadata_path(tmp_path):
    assert mc._metadata_path(tmp_path) == tmp_path / "modules.json"


def test__md5sum(tmp_path):
    path = write(tmp_path / "a.txt", b"hello")
    assert mc._md5sum(path) == "5d41402abc4b2a76b9719d911017c592"


def test__get_mod_files(tmp_path, caplog):
    d1 = tmp_path / "a"
    d1.mkdir()
    d2 = tmp_path / "b"
    d2.mkdir()

    write(d1 / "x.mod", b"content_x")
    write(d2 / "y.mod", b"content_y")
    write(d2 / "x.mod", b"content_x_override")  # same name in d2 overrides d1
    write(d2 / "z.mod", b"content_y")  # same content as `y.mod`

    with caplog.at_level(logging.WARNING):
        res = mc._get_mod_files([d1, d2])

    assert "Already seen `x.mod`" in caplog.text
    assert "Already added a file with the same contents" in caplog.text
    names = {p.name for p in res}
    assert "x.mod" in names
    assert "y.mod" in names
    assert next(p for p in res if p.name == "x.mod").parent == d2.absolute()


def test__get_mod_files_missing_dir(tmp_path):
    """Non-existent input dirs are skipped instead of raising."""
    assert mc._get_mod_files([tmp_path / "does_not_exist"]) == {}


def test__place_mod_files(tmp_path):
    output_dir = tmp_path
    mod_dir = output_dir / mc.MOD_FILES_PATH

    # Create a pre-existing .mod file that should be cleaned
    mod_dir.mkdir(parents=True)
    (mod_dir / "old.mod").write_text("old")

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.mod").write_text("aaa")
    (src_dir / "b.mod").write_text("bbb")

    result = mc._place_mod_files(output_dir, [src_dir / "a.mod", src_dir / "b.mod"])

    assert result == mod_dir.absolute()
    assert not (mod_dir / "old.mod").exists()
    assert (mod_dir / "a.mod").read_text() == "aaa"
    assert (mod_dir / "b.mod").read_text() == "bbb"


def test__generate_mod_metadata(tmp_path):
    path = write(tmp_path / "a.mod", b"mod content")

    mod_files = {path: "abc123"}
    options = Options(incflags="-DFOO", loadflags="")

    meta = mc._generate_mod_metadata(mod_files, options)
    assert meta["version"] == 1
    assert meta["hashes"] == [["a.mod", "abc123"]]
    assert meta["incflags"] == "-DFOO"


def test__check_cache_missing(tmp_path):
    options = Options(incflags="", loadflags="")
    assert not mc._check_cache({}, tmp_path, options)


def test__write_cache_and_check_cache(tmp_path):
    path = write(tmp_path / "a.mod", b"content")
    mod_files = {path: "hash1"}
    options0 = Options(incflags="-DX", loadflags="")

    mc._write_cache(mod_files, tmp_path, options0)
    assert mc._metadata_path(tmp_path).exists()

    # metadata matches but compiled lib is still missing -> not a hit
    assert not mc._check_cache(mod_files, tmp_path, options0)

    write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
    assert mc._check_cache(mod_files, tmp_path, options0)

    # different options -> no hit even with same mod files
    options1 = Options(incflags="-DX", loadflags="-different")
    assert not mc._check_cache(mod_files, tmp_path, options1)


def test__output_dynamic_file(tmp_path):
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    arch = platform.machine()
    base = tmp_path / arch
    base.mkdir()
    write(base / f"libnrnmech{ext}", b"")

    libnrnmech = mc._get_dynamic_file(tmp_path, "libnrnmech")
    assert libnrnmech == base / f"libnrnmech{ext}"
    assert libnrnmech.exists()


def test__build_mod_files_no_mods(tmp_path):
    options = Options(incflags="", loadflags="")
    with pytest.raises(mc.ModCompilationError, match="No mod files found"):
        mc._build_mod_files([], tmp_path, "echo", options)


def test__build_mod_files_compiles_and_caches(tmp_path):
    options = Options(incflags="", loadflags="")
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    def fake_run(cmd, cwd, stdout, check):
        # Simulate nrnivmodl producing the compiled library.
        write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        result = mc._build_mod_files([input_dir], tmp_path, "echo", options)
        assert mock_run.call_count == 1
        assert result == mc._get_dynamic_file(tmp_path, "libnrnmech")

        # Second call should be a cache hit: nrnivmodl not invoked again.
        result2 = mc._build_mod_files([input_dir], tmp_path, "echo", options)
        assert mock_run.call_count == 1
        assert result2 == result


def test__build_mod_files_nrnivmodl_not_found(tmp_path):
    options = Options(incflags="", loadflags="")
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    with patch("bluecellulab.mod_compilation.shutil.which", return_value=None):
        with pytest.raises(mc.ModCompilationError, match="nrnivmodl not found"):
            mc._build_mod_files([input_dir], tmp_path, None, options)


def test__build_mod_files_compile_failure(tmp_path):
    options = Options(incflags="", loadflags="")
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    with patch(
        "bluecellulab.mod_compilation.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1),
    ):
        with pytest.raises(mc.ModCompilationError, match="Failed to compile"):
            mc._build_mod_files([input_dir], tmp_path, "echo", options)


def test__build_mod_files_missing_output_after_success(tmp_path):
    """nrnivmodl reports success but the expected library is absent."""
    options = Options(incflags="", loadflags="")
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    with patch(
        "bluecellulab.mod_compilation.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    ):
        with pytest.raises(mc.ModCompilationError, match="does not exist"):
            mc._build_mod_files([input_dir], tmp_path, "echo", options)


def test_extract_mechanisms_dir_unset(tmp_path):
    circuit_config = (
        Path(__file__).parent / "examples/circuit_sonata_quick_scx/circuit_sonata.json"
    )
    assert mc.extract_mechanisms_dir(circuit_config) == []


def test_extract_mechanisms_dir_set(tmp_path):
    src = Path(__file__).parent / "examples/circuit_sonata_quick_scx/circuit_sonata.json"
    with src.open() as fd:
        js = json.load(fd)

    mods_dir = tmp_path / "mod"
    mods_dir.mkdir()
    js["networks"]["nodes"][0]["populations"]["NodeA"]["mechanisms_dir"] = str(mods_dir)

    circuit_config = tmp_path / "circuit_sonata.json"
    with circuit_config.open("w") as fd:
        json.dump(js, fd)

    res = mc.extract_mechanisms_dir(circuit_config)
    assert res == [mods_dir.absolute()]


def test_compile_mechanisms_empty_input_without_internal_mods():
    assert mc.compile_mechanisms([], include_internal_mods=False) is None


def test_compile_mechanisms_empty_input_includes_internal_mods_by_default(tmp_path, monkeypatch):
    """With no circuit-specific input dirs, BlueCelluLab's own bundled mod
    files are still gathered and compiled by default."""
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))

    captured_input_dirs = {}

    def fake_get_mod_files(input_dirs):
        captured_input_dirs["dirs"] = list(input_dirs)
        return {tmp_path / "fake.mod": "hash"}

    def fake_run(cmd, cwd, stdout, check):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation._get_mod_files", side_effect=fake_get_mod_files):
        with patch("bluecellulab.mod_compilation._check_cache", return_value=False):
            with patch("bluecellulab.mod_compilation._place_mod_files", return_value=tmp_path):
                with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run):
                    result = mc.compile_mechanisms([], nrnivmodl_path="echo")

    assert mc._internal_mods_path() in captured_input_dirs["dirs"]
    assert result == mc._get_dynamic_file(tmp_path / "cache", "libnrnmech")


def test_compile_mechanisms_uses_default_build_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    def fake_run(cmd, cwd, stdout, check):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run):
        result = mc.compile_mechanisms([input_dir], nrnivmodl_path="echo")

    assert result == mc._get_dynamic_file(tmp_path / "cache", "libnrnmech")


def test_internal_mods_path_exists_and_contains_expected_files():
    """BlueCelluLab bundles the shared "Class 2" technical mod files that
    neurodamus also bundles internally (vecevent, TTX switch, etc.), so
    circuits don't need to carry them in their own mechanisms_dir.
    See https://github.com/openbraininstitute/prod-build-circuit/issues/32
    """
    path = mc._internal_mods_path()
    assert path.exists()
    names = {f.name for f in path.glob("*.mod")}
    assert names == {
        "vecevent.mod",
        "TTXDynamicsSwitch.mod",
        "InhPoissonStim.mod",
        "ConductanceSource.mod",
        "MembraneCurrentSource.mod",
    }


def test_compile_mechanisms_include_internal_mods_false_uses_only_input_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", b"mod file")

    def fake_run(cmd, cwd, stdout, check):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms([input_dir], nrnivmodl_path="echo", include_internal_mods=False)

    # only a.mod from input_dir, not the 5 internal mods, should have been gathered
    mod_dir_arg = Path(mock_run.call_args[0][0][-1])
    assert {f.name for f in mod_dir_arg.glob("*.mod")} == {"a.mod"}


class TestCompileLock:
    def test_lock_is_released_after_use(self, tmp_path):
        with mc._CompileLock(tmp_path):
            assert (tmp_path / ".compile.lock").exists()
        assert not (tmp_path / ".compile.lock").exists()

    def test_lock_times_out_if_held(self, tmp_path):
        lock_path = tmp_path / ".compile.lock"
        tmp_path.mkdir(parents=True, exist_ok=True)
        os.close(os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR))
        try:
            with patch.object(mc, "_LOCK_POLL_INTERVAL_S", 0.01):
                with pytest.raises(mc.ModCompilationError, match="Timed out waiting for lock"):
                    with mc._CompileLock(tmp_path, timeout=0.05):
                        pass  # pragma: no cover
        finally:
            lock_path.unlink()
