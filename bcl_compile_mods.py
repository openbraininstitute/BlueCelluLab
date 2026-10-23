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
"""Entry point of ``bcl-compile-mods``.

The CLI must not import the ``bluecellulab`` package: its ``__init__`` imports
NEURON, which auto-loads an existing ``./<arch>`` build from the working
directory — the very library this command may be about to rebuild. Instead,
``bluecellulab/mod_compilation.py`` is loaded directly by file path; it only
needs the standard library and ``libsonata``.
"""

import importlib.util
import sys
from pathlib import Path


def _load_mod_compilation():
    """Load ``bluecellulab/mod_compilation.py`` without the package."""
    src = Path(__file__).resolve().parent / "bluecellulab" / "mod_compilation.py"
    spec = importlib.util.spec_from_file_location("_bcl_mod_compilation", src)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass resolution looks the module up by name in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main = _load_mod_compilation().main
