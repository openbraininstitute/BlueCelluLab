Compiling Mechanisms
====================================

Welcome to a brief tutorial on compiling mechanisms in BlueCelluLab!

In order to facilitate smooth simulations with this tool, one must navigate through specific steps, especially given its dependency on the NEURON simulator. This guide aims to detail the necessary steps and considerations for compiling neuron mechanisms in BlueCelluLab.

Compiling with bluecellulab compile-mods
----------------------------------------

As with neurodamus (``neurodamus-compile-mods``), compiling MOD files is an explicit step run
before the simulation. BlueCelluLab never compiles on import. Run, in the directory you will run
the simulation from:

.. code-block:: shell

   # A SONATA circuit: compiles its mechanisms_dir plus BlueCelluLab's bundled files
   bluecellulab compile-mods --circuit-config /path/to/circuit_config.json

   # Or plain directories of MOD files
   bluecellulab compile-mods /path/to/mod

   # The same via the module form, e.g. when the command is not on PATH
   python -m bluecellulab compile-mods /path/to/mod

This builds into ``./<arch>`` (``arm64`` or ``x86_64``), the folder NEURON loads automatically when
it is imported. The MOD files used are copied to ``./mod_files`` and recorded in
``./modules.json``. Running the command again with unchanged files does nothing; after you edit a
MOD file it rebuilds. Use ``--output-dir`` to build elsewhere, and
``--no-with-internal-mods`` to leave out the bundled files. ``./mod_files`` is emptied on every
build, so it cannot also be one of the source directories; this is refused with an error.

Importing ``bluecellulab`` loads NEURON, which auto-loads an existing ``./<arch>`` build from the
current directory. That is fine: a stale build is unlinked before ``nrnivmodl`` replaces it, and
the NEURON version baked into the cache key is read from the compiler's ``nrniv`` (falling back to
pip metadata, then a warning), so a build made by a different NEURON is rebuilt rather than reused.
If NEURON itself was upgraded and the old library no longer loads, delete ``./<arch>`` first or use
``--output-dir``. On headless runs, set ``NEURON_MODULE_OPTIONS=-nogui`` to avoid the GUI import
error.

Compile once, before starting the simulation. Under MPI, run ``bluecellulab compile-mods`` before
``mpiexec`` rather than from every rank, since concurrent builds into the same directory collide.

From a notebook or script, call the function instead:

.. code-block:: python

   from bluecellulab.mod_compilation import compile_mods

   compile_mods(["/path/to/mod"])  # or compile_mods(circuit_config="circuit_config.json")

Because NEURON can only load mechanisms once per process, ``compile_mods`` loads the new build
into the running process only if its mechanisms are not loaded yet. If an older build was already
loaded (for example auto-loaded at import), it logs a warning asking you to restart the kernel.

BlueCelluLab also warns, when it first loads mechanisms, if any MOD file changed (or was added to a
source directory) since the build in the working directory was compiled, and when a SONATA circuit
declares mechanisms that are not loaded or a ``mechanisms_dir`` that does not exist. If the library
was rebuilt afterwards with plain ``nrnivmodl``, ``modules.json`` no longer describes it: the
warning is skipped and the next ``bluecellulab compile-mods`` rebuilds.

Technical MOD files supplied by BlueCelluLab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some mechanisms are infrastructure shared between simulators rather than part of any particular
circuit's biophysical model. BlueCelluLab bundles these and supplies them itself, so circuits do
not need to carry them in their ``mechanisms_dir``. This matches what neurodamus does, and follows
the "Class 1 / Class 2" classification in `prod-build-circuit#32
<https://github.com/openbraininstitute/prod-build-circuit/issues/32>`_.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Mechanism
     - File
     - Used by BlueCelluLab for
   * - ``VecStim``
     - ``vecevent.mod``
     - Spike replay
   * - ``InhPoissonStim``
     - ``InhPoissonStim.mod``
     - Spontaneous synaptic events (minis)
   * - ``TTXDynamicsSwitch``
     - ``TTXDynamicsSwitch.mod``
     - The ``ttx`` modification, i.e. ``Cell.enable_ttx``
   * - ``ConductanceSource``
     - ``ConductanceSource.mod``
     - Not used directly. Bundled to keep the set aligned with neurodamus, and so a
       circuit whose templates reference it still finds it.
   * - ``MembraneCurrentSource``
     - ``MembraneCurrentSource.mod``
     - Not used directly, as above.

BlueCelluLab generates Ornstein-Uhlenbeck and shot-noise stimuli in Python and injects them with
NEURON's own ``IClamp``/``SEClamp``, which is why the two source mechanisms are not needed for
those.

The first three are used by :class:`bluecellulab.Cell` itself, independently of any circuit, so
``bluecellulab compile-mods`` includes them for single-cell workflows as well as for circuits.

Circuits often ship these same mechanisms under different filenames: ``VecStim.mod`` for
``VecStim`` and ``netstim_inhpoisson.mod`` for ``InhPoissonStim``. They are recognised by the
mechanism they declare rather than by filename, so they are resolved against the bundled copies
correctly either way.

How MOD files are classified
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table above is the "Class 2" set. `prod-build-circuit#32
<https://github.com/openbraininstitute/prod-build-circuit/issues/32>`_ divides MOD files into
three groups, and only one of them belongs in a circuit directory:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Group
     - Who provides it
     - Files
   * - Class 1
     - Neurodamus, internally
     - ``SonataReports.mod``, ``SonataReportHelper.mod``,
       ``CoreNEURONArtificialCell.mod``, and the deprecated ``DetAMPANMDA.mod`` /
       ``DetGABAAB.mod``
   * - Class 2
     - The simulator: neurodamus **and** BlueCelluLab
     - ``vecevent.mod`` (or legacy ``VecStim.mod``), ``TTXDynamicsSwitch.mod``,
       ``netstim_inhpoisson.mod`` (bundled here as ``InhPoissonStim.mod``),
       ``ConductanceSource.mod``, ``MembraneCurrentSource.mod``
   * - Circuit specific
     - The circuit, in its own ``mod`` directory
     - Ion channels and synapse mechanisms: ``NaTg.mod``, ``SKv3_1.mod``,
       ``ProbAMPANMDA_EMS.mod``, ``ProbGABAAB_EMS.mod``, ``StochKv3.mod`` and so on

Only the circuit-specific group should live in a circuit's ``mechanisms_dir``. The ion channel and
synapse mechanisms are part of the model being simulated, vary between circuits, and BlueCelluLab
never supplies them: they are discovered from the circuit as described below.

Class 1 files are neurodamus internals. BlueCelluLab neither bundles nor needs them, because it
writes SONATA spike and compartment reports with h5py directly rather than through NEURON
mechanisms. They are tolerated if a circuit happens to ship them: ``SonataReports.mod`` and
``SonataReportHelper.mod`` include ``bbp/sonata/reports.h``, which is unavailable here and would
otherwise fail the whole compilation, so BlueCelluLab passes ``-DDISABLE_REPORTINGLIB`` to
``nrnivmodl`` by default. Those files then compile to inert stubs and the circuit still runs.

Circuit MOD files (SONATA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``--circuit-config``, the ``mechanisms_dir`` declared under ``components`` in the circuit's
``circuit_config.json`` (see the `SONATA config documentation
<https://sonata-extension.readthedocs.io/en/latest/sonata_config.html>`_) is compiled together with
the bundled technical files into a single library:

.. code-block:: json

   {
       "components": {
           "mechanisms_dir": "/path/to/mod/files"
       }
   }

A relative ``mechanisms_dir`` is resolved against the directory holding the circuit config. A
``mechanisms_dir`` set on an individual node or edge population is picked up too, and the same
directories are searched for synapse helper HOCs. A declared directory that does not exist is an
error when compiling.

A MOD file that declares several mechanisms is kept only if none of them is provided by a
higher-priority file; it never displaces a lower-priority file it loses to. If that leaves one of
its mechanisms without any provider, a warning names it. ``SUFFIX nothing``, the placeholder that
function-only MOD files conventionally declare, is not treated as a mechanism.

Pre-compiling for a service
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you run ``nrnivmodl`` yourself, for example in a container image, copy every file from
:func:`bluecellulab.mod_compilation.internal_mods_path` into the same directory first, or simply
use ``bluecellulab compile-mods``, which does this for you.

Duplicate mechanisms
^^^^^^^^^^^^^^^^^^^^

Two MOD files can define the same mechanism under different filenames. The clearest example is the
legacy ``VecStim.mod`` and its successor ``vecevent.mod``, which both declare
``ARTIFICIAL_CELL VecStim``. ``nrnivmodl`` compiles such a pair without complaint, and NEURON then
refuses to load the result with *"The user defined name already exists"*.
``bluecellulab compile-mods`` therefore resolves duplicates by the mechanism name each file
declares, not just by filename, and compiles only one of them.

By default BlueCelluLab's own copy wins, as in neurodamus, and a warning names the circuit file
that was skipped. To keep a circuit's own copy instead, for example to reproduce results from a
circuit shipping a customised technical MOD file, set:

.. code-block:: shell

   export BLUECELLULAB_MOD_PRECEDENCE=circuit

``BLUECELLULAB_MOD_PRECEDENCE`` takes one of two keywords, ``simulator`` (the default) or
``circuit``. Any other value is ignored with a warning, and ``simulator`` is used.

Manual Compilation
------------------

You can still run ``nrnivmodl`` yourself; BlueCelluLab loads the result the same way. You then
have to include the bundled technical files and avoid duplicate mechanisms yourself.

Importance of the Working Directory
-----------------------------------

Importing ``bluecellulab`` imports NEURON, which automatically loads the mechanism library in the current working directory.

Choose the working directory before importing NEURON or BlueCelluLab:

- Navigate (``cd``) into the directory containing your compiled ``arm64`` or ``x86_64`` folder.
- Do not change directories to switch libraries after NEURON has been imported; start a new process instead.

.. code-block:: python

   import bluecellulab  # imports NEURON and auto-loads ./<arch> if present

Compile Mechanisms Using nrnivmodl
----------------------------------

To compile the neuron mechanisms, utilize the ``nrnivmodl`` command. This command should generate a folder containing compiled mechanisms, and the name of this folder will vary depending on your machine's architecture. The typical folder names to expect include "i686", "x86_64", "powerpc", or "umac".

Usage:

.. code-block:: shell

   nrnivmodl <path_to_mod_files>

Ensure to:

1. Replace ``<path_to_mod_files>`` with the path to your ``.mod`` files.
2. Check for the creation of one of the aforementioned folders upon successful compilation.

Working with Compiled Mechanisms
--------------------------------

BlueCelluLab will automatically load the compiled mechanisms if one of the architecture-specific folders ("i686", "x86_64", "powerpc", or "umac") is present in the current working directory. It is worth noting that after utilizing ``nrnivmodl``, only one of these folders should be present.

Customizing Mechanism Path with Environment Variable
----------------------------------------------------

In scenarios where you desire to point BlueCelluLab to another directory containing the compiled mechanisms, utilize the ``BLUECELLULAB_MOD_LIBRARY_PATH`` environment variable. Set it to point to the desired folder containing the compiled mechanisms.

Example:

.. code-block:: shell

   export BLUECELLULAB_MOD_LIBRARY_PATH="YOUR/DIRECTORY/x86_64"

Replace ``"YOUR/DIRECTORY/x86_64"`` with the path to your specific compiled mechanism directory.

Important Note on Path Specification
------------------------------------

Use **either** a compiled mechanism directory in the working directory **or** ``BLUECELLULAB_MOD_LIBRARY_PATH``, not both. BlueCelluLab raises an error when it loads mechanisms if the environment variable is set and the working directory already contains a build for the current architecture.

May your simulations run smoothly with BlueCelluLab!
