"""Package for performing single cell or network simulations and experiments.

The NEURON-backed part of the public API is resolved lazily (PEP 562):
``import bluecellulab`` neither imports NEURON nor loads the hoc dependencies.
That work happens once, on first access to a name in ``_LAZY_INIT`` below,
through `bluecellulab.importer.initialize_neuron`, so lightweight entry
points such as the ``bluecellulab`` command line do not import NEURON (which
would auto-load an existing ``./<arch>`` build from the working directory and
fail in headless environments).
"""

import importlib
import importlib.util
import pkgutil
from typing import TYPE_CHECKING

try:
    import bluepy
    BLUEPY_AVAILABLE = True
except ImportError:
    BLUEPY_AVAILABLE = False

from .verbosity import *  # noqa: F403

if TYPE_CHECKING:
    import neuron
    from bluecellulab.importer import import_hoc
    from bluecellulab.simulation.neuron_globals import NeuronGlobals
    from .cell import Cell, create_ball_stick, MorphIOWrapper
    from .circuit import EmodelProperties
    from .connection import Connection
    from .plotwindow import PlotWindow
    from .dendrogram import Dendrogram
    from .psection import PSection
    from .psegment import PSegment
    from .simulation import Simulation
    from .rngsettings import RNGSettings
    from .circuit_simulation import CircuitSimulation

__all__ = [
    "BLUEPY_AVAILABLE",
    "Cell",
    "CircuitSimulation",
    "Connection",
    "Dendrogram",
    "ENV_VERBOSE_LEVEL",
    "EmodelProperties",
    "MorphIOWrapper",
    "NeuronGlobals",
    "PSection",
    "PSegment",
    "PlotWindow",
    "RNGSettings",
    "Simulation",
    "VERBOSE_LEVEL",
    "create_ball_stick",
    "import_hoc",
    "set_verbose",  # noqa: F822
    "set_verbose_from_env",  # noqa: F822
]

# Public names resolved on first access, mapped to the module providing them.
# Each of these triggers `importer.initialize_neuron` first, matching the
# state a plain `import bluecellulab` used to leave behind.
_LAZY_INIT = {
    "Cell": "bluecellulab.cell",
    "create_ball_stick": "bluecellulab.cell",
    "MorphIOWrapper": "bluecellulab.cell",
    "EmodelProperties": "bluecellulab.circuit",
    "Connection": "bluecellulab.connection",
    "PlotWindow": "bluecellulab.plotwindow",
    "Dendrogram": "bluecellulab.dendrogram",
    "PSection": "bluecellulab.psection",
    "PSegment": "bluecellulab.psegment",
    "Simulation": "bluecellulab.simulation",
    "RNGSettings": "bluecellulab.rngsettings",
    "CircuitSimulation": "bluecellulab.circuit_simulation",
    "NeuronGlobals": "bluecellulab.simulation.neuron_globals",
    "import_hoc": "bluecellulab.importer",
    "neuron": "neuron",
}


def __getattr__(name: str):
    """Resolve a lazy public name, initialising NEURON when needed.

    Public names in ``_LAZY_INIT`` run `importer.initialize_neuron` first.
    Any other name is tried as a submodule WITHOUT initialising NEURON:
    a submodule's own top-level code may still import neuron (and auto-load
    ./<arch>), but resolving it through `__getattr__` — which the package
    itself uses internally, e.g. `from bluecellulab import circuit` — must
    not run full initialisation in the middle of another import.
    """
    if name in _LAZY_INIT:
        from bluecellulab.importer import initialize_neuron
        initialize_neuron()
        module = importlib.import_module(_LAZY_INIT[name])
        if module.__name__.rpartition(".")[2] != name:
            module = getattr(module, name)
    else:
        try:
            spec = importlib.util.find_spec(f"{__name__}.{name}")
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            )
        module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list:
    submodules = {m.name for m in pkgutil.iter_modules(__path__)}
    return sorted(set(__all__) | set(_LAZY_INIT) | submodules | set(globals()))
