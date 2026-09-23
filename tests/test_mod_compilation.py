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


def mod(name, mechanism, keyword="POINT_PROCESS", extra=""):
    """Return the text of a minimal mod file declaring `mechanism`."""
    return f"NEURON {{\n\t{keyword} {mechanism}\n}}\n{extra}"


# --------------------------------------------------------------------------
# mechanism declaration parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keyword", ["SUFFIX", "POINT_PROCESS", "ARTIFICIAL_CELL"]
)
def test_declared_mechanisms_all_keywords(tmp_path, keyword):
    path = write(tmp_path / "a.mod", mod("a", "Foo", keyword))
    assert mc.declared_mechanisms(path) == {"Foo"}


def test_declared_mechanisms_ignores_comment_blocks(tmp_path):
    """A declaration quoted inside COMMENT ... ENDCOMMENT is not real."""
    path = write(
        tmp_path / "a.mod",
        "COMMENT\nSUFFIX NotReal\nENDCOMMENT\n" + mod("a", "Real", "SUFFIX"),
    )
    assert mc.declared_mechanisms(path) == {"Real"}


def test_declared_mechanisms_ignores_line_comments(tmp_path):
    path = write(tmp_path / "a.mod", ": SUFFIX NotReal\n" + mod("a", "Real", "SUFFIX"))
    assert mc.declared_mechanisms(path) == {"Real"}


def test_declared_mechanisms_none_found(tmp_path):
    path = write(tmp_path / "a.mod", b"nothing declared here")
    assert mc.declared_mechanisms(path) == set()


def test_declared_mechanisms_unreadable_file(tmp_path):
    assert mc.declared_mechanisms(tmp_path / "missing.mod") == set()


def test_declared_mechanisms_on_bundled_files():
    """The bundled technical mod files declare the mechanisms we expect.

    In particular ``vecevent.mod`` provides ``VecStim`` and
    ``InhPoissonStim.mod`` provides ``InhPoissonStim``: the filenames differ
    from the mechanism names, which is exactly why resolution cannot be done
    on filenames alone.
    """
    internal = mc._internal_mods_path()
    declared = {
        f.name: mc.declared_mechanisms(f) for f in internal.glob("*.mod")
    }
    assert declared == {
        "vecevent.mod": {"VecStim"},
        "TTXDynamicsSwitch.mod": {"TTXDynamicsSwitch"},
        "InhPoissonStim.mod": {"InhPoissonStim"},
        "ConductanceSource.mod": {"ConductanceSource"},
        "MembraneCurrentSource.mod": {"MembraneCurrentSource"},
    }


# --------------------------------------------------------------------------
# precedence configuration
# --------------------------------------------------------------------------


def test_circuit_mods_take_precedence_default(monkeypatch):
    """Defaults to simulator-wins, matching neurodamus."""
    monkeypatch.delenv(mc.MOD_PRECEDENCE_ENV_VAR, raising=False)
    assert mc.circuit_mods_take_precedence() is False


@pytest.mark.parametrize("value", ["circuit", "CIRCUIT", " circuit "])
def test_circuit_mods_take_precedence_enabled(monkeypatch, value):
    monkeypatch.setenv(mc.MOD_PRECEDENCE_ENV_VAR, value)
    assert mc.circuit_mods_take_precedence() is True


def test_circuit_mods_take_precedence_unknown_value_warns(monkeypatch, caplog):
    monkeypatch.setenv(mc.MOD_PRECEDENCE_ENV_VAR, "nonsense")
    with caplog.at_level(logging.WARNING):
        assert mc.circuit_mods_take_precedence() is False
    assert "Ignoring unknown" in caplog.text


# --------------------------------------------------------------------------
# select_mod_files: filename and mechanism level resolution
# --------------------------------------------------------------------------


def test_select_mod_files_filename_last_wins(tmp_path, caplog):
    """Same filename in two dirs: the later dir wins, as in neurodamus."""
    d1, d2 = tmp_path / "a", tmp_path / "b"
    write(d1 / "x.mod", mod("x", "X", "SUFFIX"))
    write(d2 / "x.mod", mod("x", "X", "SUFFIX", extra=": newer"))
    write(d1 / "y.mod", mod("y", "Y", "SUFFIX"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files([d1, d2], already_registered=set())

    assert {p.name for p in res} == {"x.mod", "y.mod"}
    assert next(p for p in res if p.name == "x.mod").parent == d2.absolute()
    assert "both provide" in caplog.text


def test_select_mod_files_same_mechanism_different_filenames(tmp_path, caplog):
    """The case filename dedup cannot catch, and NEURON dies on."""
    circuit, internal = tmp_path / "circuit", tmp_path / "internal"
    write(circuit / "VecStim.mod", mod("legacy", "VecStim", "ARTIFICIAL_CELL"))
    write(internal / "vecevent.mod", mod("current", "VecStim", "ARTIFICIAL_CELL"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files(
            [circuit, internal], internal_dir=internal, already_registered=set()
        )

    # Exactly one provider of VecStim survives.
    assert {p.name for p in res} == {"vecevent.mod"}
    assert "provides mechanism 'VecStim'" in caplog.text
    assert "BlueCelluLab's copy takes precedence" in caplog.text
    assert f"{mc.MOD_PRECEDENCE_ENV_VAR}=circuit" in caplog.text


def test_select_mod_files_circuit_precedence(tmp_path, caplog):
    """With circuit precedence the circuit's own copy is kept instead."""
    circuit, internal = tmp_path / "circuit", tmp_path / "internal"
    write(circuit / "VecStim.mod", mod("legacy", "VecStim", "ARTIFICIAL_CELL"))
    write(internal / "vecevent.mod", mod("current", "VecStim", "ARTIFICIAL_CELL"))

    # Caller orders the dirs; internal first means the circuit copy wins.
    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files(
            [internal, circuit], internal_dir=internal, already_registered=set()
        )

    assert {p.name for p in res} == {"VecStim.mod"}
    assert "The circuit copy takes precedence" in caplog.text
    # No point advising an env var that is already in effect.
    assert f"{mc.MOD_PRECEDENCE_ENV_VAR}=circuit" not in caplog.text


def test_select_mod_files_same_filename_gets_helpful_warning(tmp_path, caplog):
    """A circuit shipping a technical mod under our own filename.

    This is resolved by filename, but the warning still has to be the one
    aimed at circuit owners rather than a terse override notice.
    """
    circuit, internal = tmp_path / "circuit", tmp_path / "internal"
    write(circuit / "TTXDynamicsSwitch.mod", mod("c", "TTXDynamicsSwitch", "SUFFIX"))
    write(internal / "TTXDynamicsSwitch.mod", mod("i", "TTXDynamicsSwitch", "SUFFIX"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files(
            [circuit, internal], internal_dir=internal, already_registered=set()
        )

    assert {p.name for p in res} == {"TTXDynamicsSwitch.mod"}
    assert "should be removed from circuit mechanisms directories" in caplog.text
    # Names match, so the message must not repeat the filename awkwardly.
    assert "which BlueCelluLab also supplies." in caplog.text


def test_select_mod_files_two_circuit_files_clashing(tmp_path, caplog):
    """Neither file is ours, so no advice about technical mod files."""
    d1, d2 = tmp_path / "a", tmp_path / "b"
    write(d1 / "one.mod", mod("one", "Dup", "SUFFIX"))
    write(d2 / "two.mod", mod("two", "Dup", "SUFFIX"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files([d1, d2], internal_dir=None, already_registered=set())

    assert {p.name for p in res} == {"two.mod"}
    assert "both provide mechanism 'Dup'" in caplog.text
    assert "should be removed from circuit" not in caplog.text


def test_select_mod_files_skips_already_registered(tmp_path):
    """Mechanisms NEURON already has cannot be registered again."""
    d = tmp_path / "mods"
    write(d / "a.mod", mod("a", "Have", "SUFFIX"))
    write(d / "b.mod", mod("b", "Missing", "SUFFIX"))

    res = mc.select_mod_files([d], already_registered={"Have"})
    assert {p.name for p in res} == {"b.mod"}


def test_select_mod_files_keeps_unparseable_files(tmp_path):
    """A file we cannot parse is compiled rather than silently dropped."""
    d = tmp_path / "mods"
    write(d / "weird.mod", b"no declaration at all")
    res = mc.select_mod_files([d], already_registered=set())
    assert {p.name for p in res} == {"weird.mod"}


def test_select_mod_files_missing_dir(tmp_path, caplog):
    """A declared directory that is not there is skipped, but not silently.

    A circuit whose `mechanisms_dir` does not exist would otherwise contribute
    nothing and only fail much later with "... is not a MECHANISM".
    """
    with caplog.at_level(logging.WARNING):
        assert mc.select_mod_files([tmp_path / "nope"], already_registered=set()) == {}
    assert "does not exist" in caplog.text


def test_default_incflags_disables_reportinglib():
    """Neurodamus's Class 1 reporting mods must not break a circuit build.

    `SonataReports.mod` and `SonataReportHelper.mod` include
    `bbp/sonata/reports.h`, which is unavailable here; without this flag their
    presence in a circuit's mechanisms directory fails the whole compilation.
    """
    assert mc.DEFAULT_INCFLAGS == "-DDISABLE_REPORTINGLIB"
    assert Options().incflags == mc.DEFAULT_INCFLAGS


def test_compile_mechanisms_passes_default_incflags(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms([], nrnivmodl_path="echo", already_registered=set())

    cmd = mock_run.call_args[0][0]
    assert "-incflags" in cmd
    assert mc.DEFAULT_INCFLAGS in cmd


def test_select_mod_files_returns_md5sums(tmp_path):
    d = tmp_path / "mods"
    path = write(d / "a.mod", b"hello")
    res = mc.select_mod_files([d], already_registered=set())
    assert res[path.absolute()] == "5d41402abc4b2a76b9719d911017c592"


# --------------------------------------------------------------------------
# cache keying
# --------------------------------------------------------------------------


def test_default_mod_build_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "custom"))
    assert mc.default_mod_build_dir({}) == (tmp_path / "custom").absolute()


def test_default_mod_build_dir_keyed_on_selected_files(monkeypatch):
    """Two runs compiling different file sets must not share a cache dir.

    Keying on the input directories instead would make a run that skipped
    some mechanisms invalidate the cache of a run that did not.
    """
    monkeypatch.delenv(mc.MOD_BUILD_DIR_ENV_VAR, raising=False)
    full = {Path("/x/a.mod"): "h1", Path("/x/b.mod"): "h2"}
    subset = {Path("/x/a.mod"): "h1"}
    reordered = {Path("/x/b.mod"): "h2", Path("/x/a.mod"): "h1"}
    moved = {Path("/other/a.mod"): "h1", Path("/other/b.mod"): "h2"}
    changed = {Path("/x/a.mod"): "h1", Path("/x/b.mod"): "DIFFERENT"}

    assert mc.default_mod_build_dir(full) == mc.default_mod_build_dir(reordered)
    # Same names and contents from elsewhere compile to the same thing.
    assert mc.default_mod_build_dir(full) == mc.default_mod_build_dir(moved)
    assert mc.default_mod_build_dir(full) != mc.default_mod_build_dir(subset)
    assert mc.default_mod_build_dir(full) != mc.default_mod_build_dir(changed)


def test__metadata_path(tmp_path):
    assert mc._metadata_path(tmp_path) == tmp_path / "modules.json"


def test__md5sum(tmp_path):
    path = write(tmp_path / "a.txt", b"hello")
    assert mc._md5sum(path) == "5d41402abc4b2a76b9719d911017c592"


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


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------


def test__build_mod_files_no_mods(tmp_path):
    options = Options(incflags="", loadflags="")
    with pytest.raises(mc.ModCompilationError, match="No mod files found"):
        mc._build_mod_files({}, tmp_path, "echo", options)


def test__build_mod_files_compiles_and_caches(tmp_path):
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")
    mod_files = {path: mc._md5sum(path)}

    def fake_run(cmd, **kwargs):
        # Simulate nrnivmodl producing the compiled library.
        write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        result = mc._build_mod_files(mod_files, tmp_path, "echo", options)
        assert mock_run.call_count == 1
        assert result == mc._get_dynamic_file(tmp_path, "libnrnmech")

        # Second call should be a cache hit: nrnivmodl not invoked again.
        result2 = mc._build_mod_files(mod_files, tmp_path, "echo", options)
        assert mock_run.call_count == 1
        assert result2 == result


def test__build_mod_files_nrnivmodl_not_found(tmp_path):
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    with patch("bluecellulab.mod_compilation.shutil.which", return_value=None):
        with pytest.raises(mc.ModCompilationError, match="nrnivmodl not found"):
            mc._build_mod_files({path: "h"}, tmp_path, None, options)


def test__build_mod_files_compile_failure(tmp_path):
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    with patch(
        "bluecellulab.mod_compilation.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=1),
    ):
        with pytest.raises(mc.ModCompilationError, match="Failed to compile"):
            mc._build_mod_files({path: "h"}, tmp_path, "echo", options)


def test__build_mod_files_missing_output_after_success(tmp_path):
    """nrnivmodl reports success but the expected library is absent."""
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    with patch(
        "bluecellulab.mod_compilation.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0),
    ):
        with pytest.raises(mc.ModCompilationError, match="does not exist"):
            mc._build_mod_files({path: "h"}, tmp_path, "echo", options)


def test__build_mod_files_captures_output_not_inherited_streams(tmp_path):
    """Compiler output must be captured, never written to `sys.stderr`.

    Under Jupyter, `sys.stdout`/`sys.stderr` are ipykernel streams with no
    usable file descriptor, so handing one to a subprocess raises and any
    notebook that triggers a compilation dies.
    """
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok")

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc._build_mod_files({path: "h"}, tmp_path, "echo", options)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT


def test__build_mod_files_failure_includes_compiler_output(tmp_path):
    """The compiler's own message is the useful part of a build failure."""
    options = Options(incflags="", loadflags="")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    with patch(
        "bluecellulab.mod_compilation.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="syntax error near line 3"
        ),
    ):
        with pytest.raises(mc.ModCompilationError, match="syntax error near line 3"):
            mc._build_mod_files({path: "h"}, tmp_path, "echo", options)


def test__build_mod_files_passes_flags(tmp_path):
    options = Options(incflags="-DFOO", loadflags="-lbar")
    path = write(tmp_path / "inputs" / "a.mod", b"mod file")

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc._build_mod_files({path: "h"}, tmp_path, "echo", options)

    cmd = mock_run.call_args[0][0]
    assert "-incflags" in cmd and "-DFOO" in cmd
    assert "-loadflags" in cmd and "-lbar" in cmd


# --------------------------------------------------------------------------
# SONATA discovery
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# compile_mechanisms
# --------------------------------------------------------------------------


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


def test_compile_mechanisms_empty_input_without_internal_mods():
    assert (
        mc.compile_mechanisms(
            [], include_internal_mods=False, already_registered=set()
        )
        is None
    )


def test_compile_mechanisms_returns_none_when_nothing_left(tmp_path):
    """Everything we would supply is already registered in NEURON."""
    already = {
        "VecStim",
        "TTXDynamicsSwitch",
        "InhPoissonStim",
        "ConductanceSource",
        "MembraneCurrentSource",
    }
    with patch("bluecellulab.mod_compilation._build_mod_files") as mock_build:
        assert mc.compile_mechanisms([], already_registered=already) is None
    mock_build.assert_not_called()


def test_compile_mechanisms_includes_internal_mods_by_default(tmp_path, monkeypatch):
    """With no circuit dirs, the bundled technical mod files are still built."""
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        result = mc.compile_mechanisms(
            [], nrnivmodl_path="echo", already_registered=set()
        )

    assert result == mc._get_dynamic_file(tmp_path / "cache", "libnrnmech")
    staged = Path(mock_run.call_args[0][0][-1])
    assert {f.name for f in staged.glob("*.mod")} == {
        "vecevent.mod",
        "TTXDynamicsSwitch.mod",
        "InhPoissonStim.mod",
        "ConductanceSource.mod",
        "MembraneCurrentSource.mod",
    }


def test_compile_mechanisms_combines_circuit_and_internal_mods(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    circuit = tmp_path / "inputs"
    write(circuit / "Ca.mod", mod("Ca", "Ca", "SUFFIX"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms([circuit], nrnivmodl_path="echo", already_registered=set())

    staged = Path(mock_run.call_args[0][0][-1])
    assert {f.name for f in staged.glob("*.mod")} == {
        "Ca.mod",
        "vecevent.mod",
        "TTXDynamicsSwitch.mod",
        "InhPoissonStim.mod",
        "ConductanceSource.mod",
        "MembraneCurrentSource.mod",
    }


def test_compile_mechanisms_include_internal_mods_false_uses_only_input_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    input_dir = tmp_path / "inputs"
    write(input_dir / "a.mod", mod("a", "A", "SUFFIX"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms(
            [input_dir],
            nrnivmodl_path="echo",
            include_internal_mods=False,
            already_registered=set(),
        )

    staged = Path(mock_run.call_args[0][0][-1])
    assert {f.name for f in staged.glob("*.mod")} == {"a.mod"}


def test_compile_mechanisms_precedence_env_var_selects_circuit_copy(tmp_path, monkeypatch):
    """End to end: the escape hatch keeps a circuit's own technical mod."""
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setenv(mc.MOD_PRECEDENCE_ENV_VAR, "circuit")
    circuit = tmp_path / "inputs"
    write(circuit / "VecStim.mod", mod("legacy", "VecStim", "ARTIFICIAL_CELL"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms([circuit], nrnivmodl_path="echo", already_registered=set())

    staged = {f.name for f in Path(mock_run.call_args[0][0][-1]).glob("*.mod")}
    assert "VecStim.mod" in staged
    assert "vecevent.mod" not in staged


def test_compile_mechanisms_default_precedence_selects_bundled_copy(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.delenv(mc.MOD_PRECEDENCE_ENV_VAR, raising=False)
    circuit = tmp_path / "inputs"
    write(circuit / "VecStim.mod", mod("legacy", "VecStim", "ARTIFICIAL_CELL"))

    def fake_run(cmd, **kwargs):
        write(mc._get_dynamic_file(tmp_path / "cache", "libnrnmech"), b"")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("bluecellulab.mod_compilation.subprocess.run", side_effect=fake_run) as mock_run:
        mc.compile_mechanisms([circuit], nrnivmodl_path="echo", already_registered=set())

    staged = {f.name for f in Path(mock_run.call_args[0][0][-1]).glob("*.mod")}
    assert "vecevent.mod" in staged
    assert "VecStim.mod" not in staged


def test_compile_mechanisms_queries_neuron_when_not_told(tmp_path, monkeypatch):
    """`already_registered=None` falls back to asking NEURON."""
    monkeypatch.setenv(mc.MOD_BUILD_DIR_ENV_VAR, str(tmp_path / "cache"))

    with patch(
        "bluecellulab.mod_compilation.registered_mechanisms", return_value={"VecStim"}
    ) as mock_reg:
        with patch("bluecellulab.mod_compilation._build_mod_files") as mock_build:
            mc.compile_mechanisms([], already_registered=None)

    mock_reg.assert_called_once()
    staged = mock_build.call_args[0][0]
    assert "vecevent.mod" not in {p.name for p in staged}


def test_registered_mechanisms_reports_builtins():
    """NEURON always has some built-in mechanisms, e.g. `pas` and `hh`."""
    names = mc.registered_mechanisms()
    assert "pas" in names
    assert "hh" in names


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
