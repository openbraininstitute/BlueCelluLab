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
"""Tests for the lazy ``bluecellulab`` package initialisation (PEP 562).

Import behaviour is asserted in fresh subprocesses: the test process itself
has already imported parts of the package, so in-process checks on
``sys.modules`` would prove nothing.
"""

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import bluecellulab

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_python(code, cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        timeout=60,
    )


def test_import_does_not_load_neuron():
    res = run_python("import sys, bluecellulab; assert 'neuron' not in sys.modules")
    assert res.returncode == 0, res.stderr


def test_cli_import_does_not_load_neuron():
    res = run_python("import sys, bluecellulab.cli; assert 'neuron' not in sys.modules")
    assert res.returncode == 0, res.stderr


def test_public_attribute_initialises_neuron():
    res = run_python(
        "import sys, bluecellulab\n"
        "_ = bluecellulab.Cell\n"
        "import neuron\n"
        "assert hasattr(neuron.h, 'TStim')\n"
        "assert neuron.h.celsius == 34.0\n"
        "assert 'morphio_wrapper' in sys.modules\n"
    )
    assert res.returncode == 0, res.stderr


def test_from_import_initialises_neuron():
    res = run_python(
        "from bluecellulab import Cell\n"
        "import neuron\n"
        "assert hasattr(neuron.h, 'TStim')\n"
    )
    assert res.returncode == 0, res.stderr


def test_direct_submodule_import_initialises_on_first_use():
    """``from bluecellulab.cell import ...`` must still get a usable object.

    ``create_ball_stick`` only needs the built-in hh mechanism, so this does
    not depend on a compiled ``./<arch>`` in the working directory.
    """
    res = run_python(
        "from bluecellulab.cell import create_ball_stick\n"
        "import neuron\n"
        "create_ball_stick()\n"
        "assert hasattr(neuron.h, 'TStim')\n"
    )
    assert res.returncode == 0, res.stderr


def test_neuron_attribute_initialises():
    """``bluecellulab.neuron`` stays available, as before lazy loading."""
    res = run_python(
        "import bluecellulab\n"
        "assert bluecellulab.neuron.h.celsius == 34.0\n"
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "setup",
    [
        "NeuronGlobals.get_instance()",
        "set_neuron_globals(temperature=34.0, v_init=-65.0)",
    ],
)
def test_direct_globals_initialise_and_preserve_values(setup, tmp_path):
    res = run_python(
        "from unittest.mock import patch\n"
        "from bluecellulab.simulation.neuron_globals import NeuronGlobals, set_neuron_globals\n"
        "from bluecellulab import importer\n"
        "import bluecellulab, neuron\n"
        "with patch.object(importer, 'import_hoc', wraps=importer.import_hoc) as load:\n"
        f"    {setup}\n"
        "    ng = NeuronGlobals.get_instance()\n"
        "    assert hasattr(neuron.h, 'TStim')\n"
        "    assert ng.temperature == neuron.h.celsius == 34.0\n"
        "    assert ng.v_init == neuron.h.v_init == -65.0\n"
        "    ng.temperature = 31.0\n"
        "    ng.v_init = -72.0\n"
        "    importer.initialize_neuron()\n"
        "    bluecellulab.Simulation()\n"
        "    assert NeuronGlobals.get_instance() is ng\n"
        "    assert ng.temperature == neuron.h.celsius == 31.0\n"
        "    assert ng.v_init == neuron.h.v_init == -72.0\n"
        "    assert load.call_count == 1\n",
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "initialize",
    [
        "importer.initialize_neuron()",
        "NeuronGlobals.get_instance()",
        "importer._load_mod_files()",
    ],
)
def test_initialization_can_retry_after_hoc_failure(initialize, tmp_path):
    res = run_python(
        "from unittest.mock import DEFAULT, patch\n"
        "from bluecellulab import importer\n"
        "from bluecellulab.simulation.neuron_globals import NeuronGlobals\n"
        "with patch.object(importer, 'import_hoc', wraps=importer.import_hoc,\n"
        "                  side_effect=[RuntimeError('failed'), DEFAULT]) as load:\n"
        "    try:\n"
        f"        {initialize}\n"
        "    except RuntimeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('expected initialization failure')\n"
        "    assert NeuronGlobals._instance is None\n"
        f"    {initialize}\n"
        "    assert load.call_count == 2\n"
        "import neuron\n"
        "assert hasattr(neuron.h, 'TStim')\n"
        "assert NeuronGlobals.get_instance().temperature == neuron.h.celsius == 34.0\n",
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr


def test_singleton_is_not_published_after_global_failure(tmp_path):
    res = run_python(
        "from unittest.mock import PropertyMock, patch\n"
        "from bluecellulab.simulation.neuron_globals import NeuronGlobals\n"
        "from bluecellulab.importer import initialize_neuron\n"
        "with patch.object(NeuronGlobals, 'v_init', new_callable=PropertyMock,\n"
        "                  side_effect=RuntimeError('failed')):\n"
        "    try:\n"
        "        initialize_neuron()\n"
        "    except RuntimeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('expected initialization failure')\n"
        "    assert NeuronGlobals._instance is None\n"
        "initialize_neuron()\n"
        "assert NeuronGlobals.get_instance().v_init == -65.0\n",
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr


def test_unknown_attribute_raises_without_neuron():
    res = run_python(
        "import sys, bluecellulab\n"
        "try:\n"
        "    bluecellulab.does_not_exist\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('no AttributeError')\n"
        "assert 'neuron' not in sys.modules\n"
    )
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "module",
    [
        "bluecellulab.cell",
        "bluecellulab.cell.core",
        "bluecellulab.circuit_simulation",
        "bluecellulab.importer",
        "bluecellulab.rngsettings",
        "bluecellulab.simulation",
        "bluecellulab.tools",
    ],
)
def test_heavy_submodule_imports_standalone(module):
    """Each NEURON-importing module must import cleanly on its own."""
    res = run_python(f"import {module}")
    assert res.returncode == 0, res.stderr


def test_all_submodules_importable():
    res = run_python(
        "import pkgutil, bluecellulab\n"
        "for m in pkgutil.walk_packages(bluecellulab.__path__, 'bluecellulab.'):\n"
        "    __import__(m.name)\n"
    )
    assert res.returncode == 0, res.stderr


def test_dir_lists_lazy_names():
    assert {"Cell", "Simulation", "CircuitSimulation", "neuron"} <= set(dir(bluecellulab))


def test_lazy_modules_cover_package():
    """New top-level submodules must be added to ``_LAZY_MODULES``, otherwise
    ``bluecellulab.<name>`` only resolves after an explicit import."""
    present = {
        m.name for m in pkgutil.iter_modules(bluecellulab.__path__)
    } - {"__main__"}
    assert present <= set(bluecellulab._LAZY_MODULES)
