Compiling Mechanisms
====================================

Welcome to a brief tutorial on compiling mechanisms in BlueCelluLab!

In order to facilitate smooth simulations with this tool, one must navigate through specific steps, especially given its dependency on the NEURON simulator. This guide aims to detail the necessary steps and considerations for compiling neuron mechanisms in BlueCelluLab.

Automatic Compilation for SONATA Circuits
------------------------------------------

When you load a SONATA circuit via :class:`bluecellulab.CircuitSimulation`, BlueCelluLab will
automatically discover and compile the circuit's MOD files, so in most cases you do **not** need
to run ``nrnivmodl`` yourself.

This works by reading the ``mechanisms_dir`` declared under ``components`` in the circuit's
``circuit_config.json`` (see the `SONATA config documentation
<https://sonata-extension.readthedocs.io/en/latest/sonata_config.html>`_):

.. code-block:: json

   {
       "components": {
           "mechanisms_dir": "/path/to/mod/files"
       }
   }

If a ``mechanisms_dir`` is found, BlueCelluLab gathers the ``.mod`` files it contains, and compiles
them with ``nrnivmodl`` into a shared library that is loaded automatically before any cell is
instantiated. The compiled library is cached (keyed by the content of the mod files and the
compilation options) under ``~/.cache/bluecellulab/mods/`` by default, so subsequent runs against
the same circuit reuse the compiled library instead of recompiling. The cache directory can be
relocated with the ``BLUECELLULAB_MOD_BUILD_DIR`` environment variable. ``BLUECELLULAB_MOD_LIBRARY_PATH``
(described below) still takes precedence over this automatic behavior, if set.

If two processes attempt to compile the same circuit's mod files at the same time (e.g. multiple
MPI ranks, or parallel workers), the first one to start compiling wins and the others wait for it
to finish and reuse the result, rather than racing on ``nrnivmodl``.

This automatic behavior is scoped to SONATA circuits declaring a ``mechanisms_dir``. If the
circuit config does not declare one, or if you are working with a bare :class:`bluecellulab.Cell`
outside of a circuit (e.g. single-cell examples), the manual workflow described below still
applies.

Note that, like any NEURON mechanism library, mod files can only be loaded once per Python
process. If your workflow instantiates more than one :class:`bluecellulab.CircuitSimulation` for
*different* circuits in the same process, only the mod files from whichever circuit is loaded
first will take effect; BlueCelluLab logs a warning if it detects this situation.

Manual Compilation (non-SONATA / single-cell workflows)
---------------------------------------------------------

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

Be mindful to adhere to the condition that **either** the current working directory should contain the compiled mechanisms **or** the ``BLUECELLULAB_MOD_LIBRARY_PATH`` environment variable should be set—**not both**. Setting the environment variable and importing BlueCelluLab from a directory containing (e.g.) an "x86_64" folder results in an error. The same restriction applies when loading a SONATA circuit that declares a ``mechanisms_dir``: it cannot be combined with an ``x86_64`` folder in the current working directory either.

In summary, BlueCelluLab resolves mechanisms in the following order:

1. ``BLUECELLULAB_MOD_LIBRARY_PATH``, if set (explicit override, works for any workflow).
2. For SONATA circuits with a declared ``mechanisms_dir``: automatic discovery, compilation and
   caching, as described above.
3. An architecture-specific folder ("x86_64", etc.) in the current working directory, for manual
   ``nrnivmodl`` workflows.

Only one of these should apply at a time; combining them raises an error.

May your simulations run smoothly with BlueCelluLab!
