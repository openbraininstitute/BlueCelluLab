# Copyright 2025 Open Brain Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for bluecellulab.cli."""

import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import bluecellulab.cli as cli
import bluecellulab.mod_compilation as mc


def write(path, contents):
    path.parent.mkdir(exist_ok=True, parents=True)
    if isinstance(contents, str):
        contents = contents.encode()
    path.write_bytes(contents)
    return path


def mod(name, mechanism, keyword="POINT_PROCESS", extra=""):
    """Return the text of a minimal mod file declaring `mechanism`."""
    return f"NEURON {{\n\t{keyword} {mechanism}\n}}\n{extra}"


def _fake_nrnivmodl(cmd, cwd=None, **_):
    if "--version" in cmd:  # the nrniv probe in neuron_version()
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")
    lib = Path(cwd) / platform.machine() / f"libnrnmech{'.dylib' if sys.platform == 'darwin' else '.so'}"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_text("lib")
    return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")


def _compiled_names(output_dir):
    return sorted(p.name for p in (output_dir / mc.MOD_FILES_PATH).glob("*.mod"))


class TestTopLevel:
    def test_no_subcommand_is_usage_error(self):
        with pytest.raises(SystemExit):
            cli.main([])

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0
        assert "bluecellulab" in capsys.readouterr().out

    def test_module_form_runs(self):
        """`python -m bluecellulab` exposes the same CLI."""
        res = subprocess.run(
            [sys.executable, "-m", "bluecellulab", "compile-mods", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        assert "compile-mods" in res.stdout


class TestCompileMods:
    def test_broken_circuit_config_returns_error(self, tmp_path, capsys):
        """An invalid --circuit-config is an error, not a traceback."""
        bad = write(tmp_path / "circuit_config.json", "{not json")
        assert cli.main(["compile-mods", "--circuit-config", str(bad)]) == 1
        assert "ERROR" in capsys.readouterr().err

    def test_restores_logger_state(self, tmp_path):
        """In-process callers must keep their logging configuration."""
        before = (mc.logger.handlers[:], mc.logger.level, mc.logger.propagate)
        bad = write(tmp_path / "circuit_config.json", "{not json")
        cli.main(["compile-mods", "--circuit-config", str(bad)])
        assert (mc.logger.handlers, mc.logger.level, mc.logger.propagate) == before

    def test_compiles_and_prints_library(self, tmp_path, capsys):
        circuit = tmp_path / "circuit"
        write(circuit / "Chan.mod", mod("Chan", "Chan", keyword="SUFFIX"))
        with patch(
            "bluecellulab.mod_compilation.subprocess.run", side_effect=_fake_nrnivmodl
        ):
            with patch("bluecellulab.mod_compilation.shutil.which", return_value="nrnivmodl"):
                rc = cli.main(
                    ["compile-mods", "--input-dir", str(circuit), "--output-dir", str(tmp_path / "out")]
                )
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
                assert cli.main(["compile-mods", str(circuit), "--output-dir", str(out)]) == 0
        assert "Chan.mod" in _compiled_names(out)

    def test_error_returns_nonzero(self, tmp_path):
        assert cli.main(["compile-mods", "--input-dir", str(tmp_path / "missing")]) == 1

    def test_nothing_to_compile_is_usage_error(self):
        with pytest.raises(SystemExit):
            cli.main(["compile-mods", "--no-with-internal-mods"])
