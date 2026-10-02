Synapse helper HOCs
====================

BlueCelluLab bundles the standard Neurodamus helper templates used when a
SONATA connection specifies ``modoverride``. The bundled files are:

* ``AMPANMDAHelper.hoc``
* ``Exp2SynHelper.hoc``
* ``GABAABHelper.hoc``
* ``GluSynapseHelper.hoc``

As in Neurodamus, the ``modoverride`` value is a *helper prefix*, not a
mechanism name: ``modoverride = "<Prefix>"`` always loads
``<Prefix>Helper.hoc`` and constructs the ``<Prefix>Helper`` template. There
are no aliases. For example, ``modoverride = "AMPANMDA"`` loads
``AMPANMDAHelper.hoc`` and constructs the ``AMPANMDAHelper`` template, and a
circuit shipping ``ProbFiltHelper.hoc`` uses ``modoverride = "ProbFilt"``.
The helper is constructed on the target section and its ``synapse`` object is
used as the point process.

.. note::

   Exception (differs from Neurodamus): ``modoverride = "GluSynapse"`` and
   ``modoverride = "Exp2Syn"`` do **not** go through ``GluSynapseHelper`` /
   ``Exp2SynHelper``. They keep BlueCelluLab's native ``GluSynapse`` and
   ``Exp2Syn`` classes, which are selected from the edge data (plasticity
   fields present, or Allen ``tau1``/``tau2``/``erev`` fields), so existing
   plasticity and Allen simulations are unchanged. The bundled
   ``GluSynapseHelper.hoc`` and ``Exp2SynHelper.hoc`` are shipped for
   Neurodamus parity but are not used for these two values.

* ``ProbAMPANMDA_EMS`` / ``ProbGABAAB_EMS`` are mechanism names, not helper
  prefixes. They are the default synapse path and need no override; setting
  ``modoverride = "ProbAMPANMDA_EMS"`` fails with a missing-helper error.
* ``AMPANMDA`` / ``GABAAB`` select the bundled ``AMPANMDAHelper`` /
  ``GABAABHelper`` for the overridden connections.

The configuration only checks that the value is a non-empty string; the
helper (and its compiled mechanism) is resolved when the synapse is built,
so mechanisms may be loaded after the configuration is parsed.

Needed attributes
-----------------

A helper may declare a semicolon-separated global string
``<Prefix>Helper_NeededAttributes`` listing SONATA edge attributes it reads
(e.g. ``GluSynapseHelper`` declares the plasticity fields). These attributes
are mandatory and have no defaults: they are extracted from edge populations
that provide them, and building an overridden synapse whose attributes are
missing (or NaN, which is how a column absent from that synapse's edge
population appears) raises ``BluecellulabError`` naming the helper, the
synapse and the missing fields. Only ``maskValue`` (``-1``) and ``location``
(``0.5``) are reserved and defaulted, as in Neurodamus.

Provenance
----------

The four bundled helper templates (``AMPANMDAHelper.hoc``, ``Exp2SynHelper.hoc``,
``GABAABHelper.hoc``, ``GluSynapseHelper.hoc``) are copied verbatim from
Neurodamus, under its open-source license, and are byte-for-byte identical to
the upstream files.

Each of these helpers loads ``RNGSettings.hoc`` via ``{load_file("RNGSettings.hoc")}``
and calls ``getSynapseSeed()`` on it. BlueCelluLab does **not** use Neurodamus's
``RNGSettings.hoc``: it keeps its own, pre-existing, and much simpler
``RNGSettings.hoc`` that only implements ``getSynapseSeed()`` and the
``rngMode``/``COMPATIBILITY``/``RANDOM123``/``UPMCELLRAN4`` constants used
elsewhere in BlueCelluLab. Neurodamus's version additionally parses a full
SONATA ``Run`` block and exposes ``getIonChannelSeed()``, ``getStimulusSeed()``,
and ``getMinisSeed()``, none of which BlueCelluLab currently needs. This
substitution is intentional and keeps the seed lookup consistent with the
rest of BlueCelluLab's RNG handling; it does not affect the ``getSynapseSeed()``
call site that the four helpers depend on.

External helpers
----------------

``<Prefix>Helper.hoc`` is resolved explicitly, first match wins:

1. the current working directory;
2. the user's ``HOC_LIBRARY_PATH`` entries, in order (the bundled directory
   is excluded here, since BlueCelluLab appends it for dependencies);
3. directories registered via ``register_helper_search_dirs``.
   :class:`SonataCircuitAccess` automatically registers each node
   population's ``mechanisms_dir`` and ``biophysical_neuron_models_dir``
   (plus the circuit-level ``components`` entries), so circuits that ship
   their own helper HOCs are found without setting ``HOC_LIBRARY_PATH``;
4. the bundled ``bluecellulab/hoc`` directory.

The resolved file is loaded by absolute path, once per prefix. If the
``<Prefix>Helper`` template is already defined in NEURON (e.g. loaded by the
user), it is reused and no file is loaded, because HOC cannot redefine a
template. The bundled HOC directory stays on ``HOC_LIBRARY_PATH`` only so
helper dependencies such as ``RNGSettings.hoc`` resolve, including for
circuit-provided helpers.

Compiled mechanisms are still required
--------------------------------------

The HOC templates do not provide the compiled NEURON mechanisms that they
instantiate. A simulation using these helpers must compile and load the
corresponding ``.mod`` mechanisms separately:

* ``AMPANMDAHelper`` requires ``ProbAMPANMDA_EMS``;
* ``GABAABHelper`` requires ``ProbGABAAB_EMS``;
* ``GluSynapseHelper`` requires ``GluSynapse``;
* ``Exp2SynHelper`` requires NEURON's ``Exp2Syn`` mechanism.

See :doc:`compiling-mechanisms` for compiling mechanisms and configuring
``BLUECELLULAB_MOD_LIBRARY_PATH``. A helper can be replaced through the cwd,
``HOC_LIBRARY_PATH`` or a registered directory, but a replacement must define
the expected ``<Prefix>Helper`` template and expose a ``synapse`` object. The replacement
also remains responsible for any compiled mechanisms it uses.
