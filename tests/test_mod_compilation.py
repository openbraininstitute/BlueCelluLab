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
    internal = mc.internal_mods_path()
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
        res = mc.select_mod_files([d1, d2])

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
            [circuit, internal], internal_dir=internal
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
            [internal, circuit], internal_dir=internal
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
            [circuit, internal], internal_dir=internal
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
        res = mc.select_mod_files([d1, d2], internal_dir=None)

    assert {p.name for p in res} == {"two.mod"}
    assert "both provide mechanism 'Dup'" in caplog.text
    assert "should be removed from circuit" not in caplog.text


def _multi_mod(*mechanisms):
    """A MOD file declaring several mechanisms (one SUFFIX per NEURON block)."""
    return "\n".join(f"NEURON {{\n\tSUFFIX {m}\n}}" for m in mechanisms).encode()


def test_select_mod_files_multi_mechanism_loser_does_not_evict(tmp_path, caplog):
    """X declares A and B; Y (low) declares A; Z (high) declares B.

    X loses B to Z and is skipped, so it must not take A away from Y.
    """
    low, mid, high = tmp_path / "low", tmp_path / "mid", tmp_path / "high"
    write(low / "y.mod", _multi_mod("A"))
    write(mid / "x.mod", _multi_mod("A", "B"))
    write(high / "z.mod", _multi_mod("B"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files([low, mid, high])

    assert {p.name for p in res} == {"y.mod", "z.mod"}
    assert "will not be available" not in caplog.text


def test_select_mod_files_multi_mechanism_warns_when_mechanism_lost(tmp_path, caplog):
    """If the skipped file was the only provider of A, say so."""
    mid, high = tmp_path / "mid", tmp_path / "high"
    write(mid / "x.mod", _multi_mod("A", "B"))
    write(high / "z.mod", _multi_mod("B"))

    with caplog.at_level(logging.WARNING):
        res = mc.select_mod_files([mid, high])

    assert {p.name for p in res} == {"z.mod"}
    assert "Mechanism 'A' is only provided by 'x.mod'" in caplog.text


def test_select_mod_files_multi_mechanism_winner_evicts_all(tmp_path):
    """A higher-priority file declaring A and B replaces both lower ones."""
    low, high = tmp_path / "low", tmp_path / "high"
    write(low / "a.mod", _multi_mod("A"))
    write(low / "b.mod", _multi_mod("B"))
    write(high / "ab.mod", _multi_mod("A", "B"))

    res = mc.select_mod_files([low, high])
    assert {p.name for p in res} == {"ab.mod"}


def test_select_mod_files_keeps_unparseable_files(tmp_path):
    """A file we cannot parse is compiled rather than silently dropped."""
    d = tmp_path / "mods"
    write(d / "weird.mod", b"no declaration at all")
    res = mc.select_mod_files([d])
    assert {p.name for p in res} == {"weird.mod"}


def test_default_incflags_disables_reportinglib():
    """Neurodamus's Class 1 reporting mods must not break a circuit build.

    `SonataReports.mod` and `SonataReportHelper.mod` include
    `bbp/sonata/reports.h`, which is unavailable here; without this flag their
    presence in a circuit's mechanisms directory fails the whole compilation.
    """
    assert mc.DEFAULT_INCFLAGS == "-DDISABLE_REPORTINGLIB"
    assert Options().incflags == mc.DEFAULT_INCFLAGS


def test_select_mod_files_returns_md5sums(tmp_path):
    d = tmp_path / "mods"
    path = write(d / "a.mod", b"hello")
    res = mc.select_mod_files([d])
    assert res[path.absolute()] == "5d41402abc4b2a76b9719d911017c592"


# --------------------------------------------------------------------------
# cache keying
# --------------------------------------------------------------------------


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
    assert meta["version"] == 2
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

    lib = write(mc._get_dynamic_file(tmp_path, "libnrnmech"), b"")
    meta_mtime = mc._metadata_path(tmp_path).stat().st_mtime
    os.utime(lib, (meta_mtime - 10, meta_mtime - 10))
    assert mc._check_cache(mod_files, tmp_path, options0)

    # different options -> no hit even with same mod files
    options1 = Options(incflags="-DX", loadflags="-different")
    assert not mc._check_cache(mod_files, tmp_path, options1)

    # library rebuilt afterwards (e.g. plain nrnivmodl) -> no hit
    os.utime(lib, (meta_mtime + 10, meta_mtime + 10))
    assert not mc._check_cache(mod_files, tmp_path, options0)


def test__check_cache_corrupt_metadata_is_a_miss(tmp_path):
    """A truncated modules.json (killed build) means rebuild, not crash."""
    mc._metadata_path(tmp_path).write_text('{"version": ')
    assert not mc._check_cache({}, tmp_path, Options())


def test__write_cache_is_atomic(tmp_path):
    """No temp file is left behind and the result is valid JSON."""
    mc._write_cache({}, tmp_path, Options())
    assert [p.name for p in tmp_path.iterdir()] == ["modules.json"]
    assert json.loads(mc._metadata_path(tmp_path).read_text())["version"] == mc.VERSION


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
        assert mc._check_cache(mod_files, tmp_path, options)
        result2 = mc._build_mod_files(mod_files, tmp_path, "echo", options, up_to_date=True)
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


def _write_circuit_config(tmp_path, *, node_dir=None, edge_dir=None):
    src = Path(__file__).parent / "examples/circuit_sonata_quick_scx/circuit_sonata.json"
    js = json.loads(src.read_text())
    if node_dir is not None:
        js["networks"]["nodes"][0]["populations"]["NodeA"]["mechanisms_dir"] = str(node_dir)
    if edge_dir is not None:
        js["networks"]["edges"][0]["populations"]["NodeA__NodeA__chemical"][
            "mechanisms_dir"
        ] = str(edge_dir)
    path = tmp_path / "circuit_sonata.json"
    path.write_text(json.dumps(js))
    return path


def test_extract_mechanisms_dir_includes_edge_populations(tmp_path):
    """SONATA lets an edge population override `mechanisms_dir` too."""
    edge_dir = tmp_path / "edge_mod"
    config = _write_circuit_config(tmp_path, edge_dir=edge_dir)
    assert mc.extract_mechanisms_dir(config) == [edge_dir.absolute()]


def test_extract_mechanisms_dir_is_ordered_and_deduplicated(tmp_path):
    """Order decides which duplicate mechanism wins, so it must be stable.

    Node populations come before edge populations, and a directory declared
    by both appears once.
    """
    node_dir, edge_dir = tmp_path / "node_mod", tmp_path / "edge_mod"
    config = _write_circuit_config(tmp_path, node_dir=node_dir, edge_dir=edge_dir)
    assert mc.extract_mechanisms_dir(config) == [node_dir.absolute(), edge_dir.absolute()]

    config = _write_circuit_config(tmp_path, node_dir=node_dir, edge_dir=node_dir)
    assert mc.extract_mechanisms_dir(config) == [node_dir.absolute()]


# --------------------------------------------------------------------------
# bundled mod files and NEURON queries
# --------------------------------------------------------------------------


def test_registered_mechanisms_reports_builtins():
    """NEURON always has some built-in mechanisms, e.g. `pas` and `hh`."""
    names = mc.registered_mechanisms()
    assert "pas" in names
    assert "hh" in names


# --------------------------------------------------------------------------
# compile_mods / CLI
# --------------------------------------------------------------------------


def _fake_nrnivmodl(cmd, cwd, **_):
    lib = Path(cwd) / platform.machine() / f"libnrnmech{'.dylib' if sys.platform == 'darwin' else '.so'}"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_text("lib")
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")


def _compile(*args, **kwargs):
    with patch(
        "bluecellulab.mod_compilation.subprocess.run", side_effect=_fake_nrnivmodl
    ) as run:
        result = mc.compile_mods(*args, nrnivmodl_path="nrnivmodl", load=False, **kwargs)
    return result, run


def _compiled_names(output_dir):
    return sorted(p.name for p in (output_dir / mc.MOD_FILES_PATH).glob("*.mod"))


def test_neuron_version_does_not_import_neuron(monkeypatch):
    """Importing NEURON would auto-load the ./<arch> build about to be replaced."""
    monkeypatch.delitem(sys.modules, "neuron", raising=False)
    with patch.object(mc.importlib.metadata, "version", return_value="9.0.1") as version:
        assert mc.neuron_version() == "9.0.1"
    version.assert_called_once_with("neuron")
    assert "neuron" not in sys.modules


def test_neuron_version_falls_back_to_nightly():
    def version(name):
        if name == "neuron":
            raise mc.importlib.metadata.PackageNotFoundError(name)
        return "9.1.dev0"

    with patch.object(mc.importlib.metadata, "version", side_effect=version):
        assert mc.neuron_version() == "9.1.dev0"


def test_neuron_version_raises_without_neuron():
    with patch.object(
        mc.importlib.metadata,
        "version",
        side_effect=mc.importlib.metadata.PackageNotFoundError,
    ):
        with pytest.raises(mc.ModCompilationError, match="NEURON package is required"):
            mc.neuron_version()


def test_metadata_includes_neuron_version(monkeypatch, tmp_path):
    path = write(tmp_path / "a.mod", mod("a", "A"))
    monkeypatch.setattr(mc, "neuron_version", lambda: "8.2.4")
    old = mc._generate_mod_metadata({path: "h"}, Options())
    monkeypatch.setattr(mc, "neuron_version", lambda: "9.0.0")
    new = mc._generate_mod_metadata({path: "h"}, Options())
    assert old["neuron_version"] == "8.2.4" and new["neuron_version"] == "9.0.0"


def test_internal_mods_path_exists_and_contains_expected_files():
    path = mc.internal_mods_path()
    names = {p.name for p in path.glob("*.mod")}
    assert {"vecevent.mod", "InhPoissonStim.mod", "TTXDynamicsSwitch.mod"} <= names


def test_compile_mods_into_output_dir_with_internal_mods(tmp_path):
    circuit = tmp_path / "circuit"
    write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    lib, run = _compile([circuit], output_dir=out)
    assert lib.parent == out / platform.machine()
    assert "Chan.mod" in _compiled_names(out)
    assert "vecevent.mod" in _compiled_names(out)
    assert run.call_args.kwargs["cwd"] == str(out)
    assert "-DDISABLE_REPORTINGLIB" in run.call_args.args[0]


def test_compile_mods_without_internal_mods(tmp_path):
    circuit = tmp_path / "circuit"
    write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    _compile([circuit], output_dir=tmp_path / "out", with_internal_mods=False)
    assert _compiled_names(tmp_path / "out") == ["Chan.mod"]


def test_compile_mods_skips_unchanged_and_rebuilds_on_change(tmp_path):
    circuit = tmp_path / "circuit"
    chan = write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    _compile([circuit], output_dir=out)
    _, run = _compile([circuit], output_dir=out)
    run.assert_not_called()
    write(chan, mod("Chan", "Chan", keyword="SUFFIX", extra=": edited\n"))
    _, run = _compile([circuit], output_dir=out)
    run.assert_called_once()


def test_compile_mods_dedups_legacy_vecstim(tmp_path, caplog):
    """A circuit's legacy VecStim.mod must not be compiled next to vecevent.mod."""
    circuit = tmp_path / "circuit"
    write(circuit / "VecStim.mod", mod("VecStim", "VecStim", keyword="ARTIFICIAL_CELL"))
    with caplog.at_level(logging.WARNING):
        _compile([circuit], output_dir=tmp_path / "out")
    names = _compiled_names(tmp_path / "out")
    assert "vecevent.mod" in names and "VecStim.mod" not in names


def test_compile_mods_precedence_env_keeps_circuit_copy(tmp_path, monkeypatch):
    monkeypatch.setenv(mc.MOD_PRECEDENCE_ENV_VAR, "circuit")
    circuit = tmp_path / "circuit"
    write(circuit / "VecStim.mod", mod("VecStim", "VecStim", keyword="ARTIFICIAL_CELL"))
    _compile([circuit], output_dir=tmp_path / "out")
    names = _compiled_names(tmp_path / "out")
    assert "VecStim.mod" in names and "vecevent.mod" not in names


def test_compile_mods_reads_circuit_config(tmp_path):
    node_dir = tmp_path / "node_mods"
    write(node_dir / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    config = _write_circuit_config(tmp_path, node_dir=node_dir)
    _compile(circuit_config=config, output_dir=tmp_path / "out")
    assert "Chan.mod" in _compiled_names(tmp_path / "out")


def test_compile_mods_missing_dir_raises(tmp_path):
    with pytest.raises(mc.ModCompilationError, match="does not exist"):
        _compile([tmp_path / "missing"], output_dir=tmp_path / "out")


def test_compile_mods_refuses_staging_dir_as_input(tmp_path, monkeypatch):
    """`bcl-compile-mods mod_files` in the cwd must not delete the sources."""
    monkeypatch.chdir(tmp_path)
    src = write(tmp_path / mc.MOD_FILES_PATH / "Chan.mod", mod("Chan", "Chan", "SUFFIX"))
    with pytest.raises(mc.ModCompilationError, match="staged"):
        _compile([mc.MOD_FILES_PATH])
    assert src.exists()


def test_compile_mods_rebuilds_after_plain_nrnivmodl(tmp_path):
    """A library newer than modules.json was not built from what it records."""
    circuit = tmp_path / "circuit"
    write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    lib, _ = _compile([circuit], output_dir=out)
    meta_mtime = mc._metadata_path(out).stat().st_mtime
    os.utime(lib, (meta_mtime + 10, meta_mtime + 10))
    _, run = _compile([circuit], output_dir=out)
    run.assert_called_once()


class TestLoadIntoRunningNeuron:
    def _lib(self, tmp_path):
        path = write(tmp_path / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
        return tmp_path / "lib.so", {path: "h"}

    def test_loads_when_mechanisms_absent(self, tmp_path):
        lib, files = self._lib(tmp_path)
        with patch.object(mc, "registered_mechanisms", return_value={"pas"}):
            with patch("neuron.h") as h:
                mc._load_into_running_neuron(lib, files, rebuilt=True)
        h.nrn_load_dll.assert_called_once_with(str(lib))

    def test_warns_to_restart_after_rebuild(self, tmp_path, caplog):
        lib, files = self._lib(tmp_path)
        with patch.object(mc, "registered_mechanisms", return_value={"Chan"}):
            with patch("neuron.h") as h, caplog.at_level(logging.WARNING):
                mc._load_into_running_neuron(lib, files, rebuilt=True)
        h.nrn_load_dll.assert_not_called()
        assert "Restart" in caplog.text

    def test_silent_when_same_build_already_loaded(self, tmp_path, caplog):
        lib, files = self._lib(tmp_path)
        with patch.object(mc, "registered_mechanisms", return_value={"Chan"}):
            with patch("neuron.h") as h, caplog.at_level(logging.WARNING):
                mc._load_into_running_neuron(lib, files, rebuilt=False)
        h.nrn_load_dll.assert_not_called()
        assert caplog.text == ""


def test_changed_mod_files(tmp_path):
    circuit = tmp_path / "circuit"
    chan = write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    assert mc.changed_mod_files(out) == []
    _compile([circuit], output_dir=out, with_internal_mods=False)
    assert mc.changed_mod_files(out) == []
    write(chan, mod("Chan", "Chan", keyword="SUFFIX", extra=": edited\n"))
    assert mc.changed_mod_files(out) == [chan.absolute()]
    chan.unlink()
    assert mc.changed_mod_files(out) == [chan.absolute()]


def test_changed_mod_files_reports_added_files(tmp_path):
    circuit = tmp_path / "circuit"
    write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    _compile([circuit], output_dir=out, with_internal_mods=False)
    new = write(circuit / "New.mod", mod("New", "New", keyword="SUFFIX"))
    assert mc.changed_mod_files(out) == [new.absolute()]


def test_changed_mod_files_silent_after_plain_nrnivmodl(tmp_path):
    """Rebuilt later with plain nrnivmodl: modules.json is stale, don't trust
    it."""
    circuit = tmp_path / "circuit"
    chan = write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
    out = tmp_path / "out"
    lib, _ = _compile([circuit], output_dir=out, with_internal_mods=False)
    write(chan, mod("Chan", "Chan", keyword="SUFFIX", extra=": edited\n"))
    meta_mtime = mc._metadata_path(out).stat().st_mtime
    os.utime(lib, (meta_mtime + 10, meta_mtime + 10))
    assert mc.changed_mod_files(out) == []


def test_changed_mod_files_ignores_corrupt_metadata(tmp_path):
    write(tmp_path / "modules.json", "{not json")
    assert mc.changed_mod_files(tmp_path) == []


class TestMain:
    def test_compiles_and_prints_library(self, tmp_path, capsys):
        circuit = tmp_path / "circuit"
        write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
        with patch(
            "bluecellulab.mod_compilation.subprocess.run", side_effect=_fake_nrnivmodl
        ):
            with patch("bluecellulab.mod_compilation.shutil.which", return_value="nrnivmodl"):
                rc = mc.main(["--input-dir", str(circuit), "--output-dir", str(tmp_path / "out")])
        assert rc == 0
        assert platform.machine() in capsys.readouterr().out

    def test_positional_dirs_like_nrnivmodl(self, tmp_path):
        circuit = tmp_path / "circuit"
        write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
        out = tmp_path / "out"
        with patch(
            "bluecellulab.mod_compilation.subprocess.run", side_effect=_fake_nrnivmodl
        ):
            with patch("bluecellulab.mod_compilation.shutil.which", return_value="nrnivmodl"):
                assert mc.main([str(circuit), "--output-dir", str(out)]) == 0
        assert "Chan.mod" in _compiled_names(out)

    def test_error_returns_nonzero(self, tmp_path):
        assert mc.main(["--input-dir", str(tmp_path / "missing")]) == 1

    def test_nothing_to_compile_is_usage_error(self):
        with pytest.raises(SystemExit):
            mc.main(["--no-with-internal-mods"])
