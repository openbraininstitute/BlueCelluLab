Synapse helper HOCs
====================

BlueCelluLab bundles the standard Neurodamus helper templates used when a
SONATA connection specifies ``modoverride``. The bundled files are:

* ``AMPANMDAHelper.hoc``
* ``Exp2SynHelper.hoc``
* ``GABAABHelper.hoc``
* ``GluSynapseHelper.hoc``

For a mechanism suffix ``<Suffix>``, the synapse loader looks for
``<Suffix>Helper.hoc``. For example, ``modoverride = "GluSynapse"`` loads
``GluSynapseHelper.hoc`` and constructs the ``GluSynapseHelper`` template.
The helper is constructed on the target section and its ``synapse`` object is
used as the point process.

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

BlueCelluLab first searches for a helper by its filename using
``HOC_LIBRARY_PATH``. This preserves the ability to override a bundled helper
with a project-specific implementation. If no external helper is found, the
matching helper from the installed BlueCelluLab package is loaded. The
bundled HOC directory is added to the HOC search path so helper dependencies,
such as ``RNGSettings.hoc``, can also be resolved.

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
``BLUECELLULAB_MOD_LIBRARY_PATH``. A helper can be replaced through
``HOC_LIBRARY_PATH``, but a replacement must define the expected
``<Suffix>Helper`` template and expose a ``synapse`` object. The replacement
also remains responsible for any compiled mechanisms it uses.
