Compiling Mechanisms
====================================

Welcome to a brief tutorial on compiling mechanisms in BlueCelluLab!

In order to facilitate smooth simulations with this tool, one must navigate through specific steps, especially given its dependency on the NEURON simulator. This guide aims to detail the necessary steps and considerations for compiling neuron mechanisms in BlueCelluLab.

Automatic Compilation
---------------------

BlueCelluLab compiles the MOD files it needs and loads them for you, so in most cases you do
**not** need to run ``nrnivmodl`` yourself. Two sets of files are involved.

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
     - The ``ttx`` modification, i.e. ``Cell.enable_ttx``. The only one of the five
       that is ion coupled, so it cannot be supplied separately from a library that
       already contains sodium channels; see `Ion coupling: one limit on what can be
       supplied separately`_.
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

The first three are used by :class:`bluecellulab.Cell` itself, independently of any circuit, so
they are supplied for bare single-cell workflows too.

Circuit MOD files (SONATA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

When you load a SONATA circuit via :class:`bluecellulab.CircuitSimulation`, BlueCelluLab also
reads the ``mechanisms_dir`` declared under ``components`` in the circuit's
``circuit_config.json`` (see the `SONATA config documentation
<https://sonata-extension.readthedocs.io/en/latest/sonata_config.html>`_):

.. code-block:: json

   {
       "components": {
           "mechanisms_dir": "/path/to/mod/files"
       }
   }

The ``.mod`` files found there are compiled together with the bundled technical files into a
single shared library, which is loaded before any cell is instantiated. A relative
``mechanisms_dir`` is resolved against the directory holding the circuit config, not the current
working directory.

If the circuit config does not declare a ``mechanisms_dir``, or declares it as an empty string,
BlueCelluLab has nothing to discover and compiles only its own bundled files. The circuit's ion
channel and synapse mechanisms then have to come from somewhere else, normally a directory you
compiled yourself with ``nrnivmodl`` as described under `Manual Compilation`_.

Caching and concurrency
^^^^^^^^^^^^^^^^^^^^^^^

The compiled library is cached under ``~/.cache/bluecellulab/mods/`` by default, keyed by the
names and contents of the files actually compiled plus the compilation options, so subsequent runs
reuse it instead of recompiling. Relocate the cache with the ``BLUECELLULAB_MOD_BUILD_DIR``
environment variable.

If two processes try to compile the same set of files at once (multiple MPI ranks, or parallel
workers), the first to start wins and the others wait for it and reuse the result, rather than
racing on ``nrnivmodl``.

Ion coupling: one limit on what can be supplied separately
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NEURON does not share an ion between separately compiled libraries. If a
mechanism that writes an ion sits in one library and a mechanism that reads it
sits in another, they do not see each other, and the simulation runs on with the
coupling simply absent rather than reporting an error.

Of the five bundled files only ``TTXDynamicsSwitch`` is affected: it writes the
custom ``ttx`` ion, which the sodium channels of a circuit read
(``USEION ttx READ ttxo, ttxi``). The other four are point processes and
artificial cells with no ion coupling, so BlueCelluLab can always supply them.

In practice this only matters if you pre-compile. When BlueCelluLab does the
compiling it puts the circuit's MOD files and the bundled ones into a single
library, so the coupling is intact. But if you compile a directory yourself and
it contains sodium channels without ``TTXDynamicsSwitch.mod``, then the ``ttx``
modification will not work, because BlueCelluLab can only add its copy in a
second library. Include ``TTXDynamicsSwitch.mod`` in your own ``nrnivmodl``
invocation in that case; :func:`bluecellulab.mod_compilation.internal_mods_path`
returns the directory to take it from.

BlueCelluLab does not let this pass quietly. It warns when it detects the split
while loading, and :meth:`bluecellulab.Cell.enable_ttx` and
:meth:`~bluecellulab.Cell.disable_ttx` raise
:class:`~bluecellulab.exceptions.BluecellulabError` rather than run a simulation
whose sodium channels are never actually blocked. Workflows that do not use the
``ttx`` modification are unaffected.

Mechanisms NEURON already has are left alone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NEURON cannot register a mechanism twice in one process, and it automatically loads an
architecture-specific folder found in the current working directory when it is imported. So
BlueCelluLab only compiles and loads what NEURON does not already provide.

This means a directory you compiled by hand with ``nrnivmodl`` keeps working exactly as before:
whatever it provides takes precedence, and BlueCelluLab fills in only what is missing.

Duplicate mechanisms
^^^^^^^^^^^^^^^^^^^^

Two MOD files can define the same mechanism under different filenames. The clearest example is the
legacy ``VecStim.mod`` and its successor ``vecevent.mod``, which both declare
``ARTIFICIAL_CELL VecStim``. ``nrnivmodl`` compiles such a pair without complaint, and NEURON then
refuses to load the result with *"The user defined name already exists"*. BlueCelluLab therefore
resolves duplicates by the mechanism name each file declares, not just by filename, and compiles
only one of them.

By default BlueCelluLab's own copy wins, as in neurodamus, and a warning names the circuit file
that was skipped. To keep a circuit's own copy instead, for example to reproduce results from a
circuit shipping a customised technical MOD file, set:

.. code-block:: shell

   export BLUECELLULAB_MOD_PRECEDENCE=circuit

BlueCelluLab still fills in any technical mechanism the circuit does not provide, so this only
changes which copy is used where both exist.

To leave out the bundled files altogether, call the compilation entry point directly:

.. code-block:: python

   from bluecellulab.mod_compilation import compile_mechanisms

   compile_mechanisms(["/path/to/mod"], include_internal_mods=False)

Be aware that this can reintroduce ``not a MECHANISM`` errors for any feature whose mechanism is
then missing, so prefer ``BLUECELLULAB_MOD_PRECEDENCE`` unless you are deliberately taking over
the whole set.

Note that, like any NEURON mechanism library, mod files can only be loaded once per Python
process. If your workflow instantiates more than one :class:`bluecellulab.CircuitSimulation` for
*different* circuits in the same process, only the mod files from whichever circuit is loaded
first will take effect; BlueCelluLab logs a warning if it detects this situation.

Manual Compilation
------------------

You can still compile mechanisms yourself, and anything you provide this way takes precedence over
BlueCelluLab's automatic compilation. The rest of this page describes that workflow.

Importance of the Working Directory
-----------------------------------

It's vital to underscore that, unlike other Python packages, the working directory you are in when you import BlueCelluLab significantly influences the output. This peculiar behavior is attributed to its utilization of the NEURON simulator.

When importing NEURON or when importing bluecellulab (that imports NEURON), ensure to:

- Navigate (``cd``) into the appropriate directory that is contaning one of the architecture-specific folders ("i686", "x86_64", "powerpc", or "umac") containing your compiled mechanisms.
- Confirm the potential impact on results that may stem from being in different directories when invoking NEURON.

.. code-block:: python

   import bluecellulab

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

Be mindful to adhere to the condition that **either** the current working directory should contain the compiled mechanisms **or** the ``BLUECELLULAB_MOD_LIBRARY_PATH`` environment variable should be set—**not both**. Setting the environment variable and importing BlueCelluLab from a directory containing (e.g.) an "x86_64" folder results in an error.

In summary, BlueCelluLab resolves mechanisms in the following order:

1. ``BLUECELLULAB_MOD_LIBRARY_PATH``, if set. The library it points at is loaded verbatim and
   nothing else happens: no discovery, no compilation. This is the full manual override.
2. Otherwise, whatever NEURON already provides is kept, including an architecture-specific folder
   ("x86_64", etc.) in the current working directory that NEURON auto-loads on import.
3. Anything still missing is compiled and loaded: the ``mechanisms_dir`` of a SONATA circuit if one
   is being used, plus BlueCelluLab's own bundled technical MOD files.

Steps 2 and 3 cooperate rather than conflict, so a hand-compiled directory and BlueCelluLab's
automatic compilation can coexist.

If a circuit declares mechanisms that cannot be compiled, that is an error: the simulation cannot
run without them. Failing to compile only BlueCelluLab's own bundled files is logged as a warning
instead, since a given script may never use the features that need them.

May your simulations run smoothly with BlueCelluLab!
