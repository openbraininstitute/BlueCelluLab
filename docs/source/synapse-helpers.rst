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

   ``modoverride = "GluSynapse"`` and ``modoverride = "Exp2Syn"`` build
   BlueCelluLab's native ``GluSynapse`` / ``Exp2Syn`` classes instead of
   ``GluSynapseHelper`` / ``Exp2SynHelper``. As in Neurodamus, the override
   forces that mechanism whatever the ``syn_type_id`` (an inhibitory edge in a
   ``GluSynapse`` block gets ``GluSynapse``). The edges must provide the
   plasticity fields (``GluSynapse``) or ``tau1``/``tau2``/``erev``
   (``Exp2Syn``); otherwise building the synapse raises
   ``BluecellulabError``. The native ``GluSynapse`` seeds its RNG with
   ``post_gid + 1``, like ``GluSynapseHelper``.

   Without ``modoverride``, BlueCelluLab keeps its data-driven selection for
   backward compatibility: edges with all plasticity fields get
   ``GluSynapse`` and Allen edges get ``Exp2Syn``. Neurodamus would build
   ``AMPANMDA`` / ``GABAAB`` synapses there.

* ``ProbAMPANMDA_EMS`` / ``ProbGABAAB_EMS`` are mechanism names, not helper
  prefixes. They are the default synapse path and need no override; setting
  ``modoverride = "ProbAMPANMDA_EMS"`` fails with a missing-helper error.
* ``AMPANMDA`` / ``GABAAB`` select the bundled ``AMPANMDAHelper`` /
  ``GABAABHelper`` for the overridden connections.

The configuration only checks that the value is a non-empty string; the
helper (and its compiled mechanism) is resolved when the synapse is built,
so mechanisms may be loaded after the configuration is parsed.

Helper parameters
-----------------

The helper receives a parameter object built as Neurodamus
``SynapseParameters`` does. Standard fields use Neurodamus names, mapped from
the SONATA edge attributes:

.. list-table::
   :header-rows: 1

   * - Helper field
     - SONATA attribute
   * - ``sgid``
     - ``@source_node``
   * - ``delay``
     - ``delay`` (rounded to ``dt``)
   * - ``isec``, ``ipt``, ``offset``
     - ``afferent_section_id``, ``-1``, ``afferent_section_pos`` (else the
       segment id and offset)
   * - ``weight``
     - ``conductance``
   * - ``U``
     - ``u_syn`` (scaled by the calcium Hill factor)
   * - ``D``, ``F``, ``DTC``
     - ``depression_time``, ``facilitation_time``, ``decay_time``
   * - ``synType``
     - ``syn_type_id``
   * - ``nrrp``
     - ``n_rrp_vesicles`` (default ``-1``)
   * - ``u_hill_coefficient``
     - ``u_hill_coefficient`` (default ``0``)
   * - ``conductance_ratio``
     - ``conductance_scale_factor`` (default ``-1``)
   * - ``maskValue``
     - reserved, always ``-1``
   * - ``location``
     - reserved, always ``0.5``

``maskValue`` and ``location`` are never read from the edges. ``nrrp`` is
mandatory in Neurodamus; BlueCelluLab defaults it to ``-1`` (the helpers then
keep the MOD default), as it treats ``n_rrp_vesicles`` as optional everywhere.

A helper may declare a semicolon-separated global string
``<Prefix>Helper_NeededAttributes`` listing extra SONATA edge attributes it
reads (e.g. ``w_corr;tau_corr``). They are passed under their raw SONATA
name, unmapped and unscaled, even when the name is also a standard attribute
(a helper declaring ``conductance`` gets ``conductance`` next to
``weight``). Needed attributes are mandatory: building an overridden synapse
whose attributes are missing (or NaN, which is how a column absent from that
synapse's edge population appears) raises ``BluecellulabError`` naming the
helper, the synapse and the missing fields.

A helper may also declare ``<Prefix>Helper_UHillScaleVariables``, a
semicolon-separated list of fields scaled by the same constrained Hill factor
as ``U`` when ``extracellular_calcium`` is set (e.g. ``GluSynapseHelper``
declares ``Use_d_TM;Use_p_TM``).

If the helper's point process exposes a ``conductance`` variable, it is set
to the synapse weight. As in Neurodamus, ``conductance_ratio`` is not applied
to overridden synapses: ``modoverride = "AMPANMDA"`` keeps the MOD default
``NMDA_ratio`` instead of the edge ``conductance_scale_factor``, unlike the
non-override path.

Override synapses are classified as excitatory or inhibitory by
``syn_type_id`` (``< 100`` is inhibitory) to pick the node-level spontaneous
minis rate, as in Neurodamus.

Random numbers
--------------

Helpers always seed Random123-style, from ``RNGSettings.getSynapseSeed()``,
which follows ``RNGSettings.synapse_seed`` at all times. Neurodamus supports
only Random123. In ``Compatibility`` or ``UpdatedMCell`` mode, native synapses
use MCellRan4 while helper synapses keep Random123 streams; BlueCelluLab logs
a warning once in that case.

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
   is excluded here);
3. the circuit's own directories: :class:`SonataCircuitAccess` collects each
   node population's ``mechanisms_dir`` and ``biophysical_neuron_models_dir``
   (plus the circuit-level ``components`` entries) in its ``helper_dirs``, so
   circuits that ship their own helper HOCs are found without setting
   ``HOC_LIBRARY_PATH``. These directories apply to that circuit only;
4. the bundled ``bluecellulab/hoc`` directory.

The resolved file is loaded by absolute path, once per prefix and process.
If the ``<Prefix>Helper`` template is already defined in NEURON (e.g. loaded
by the user or by another circuit), it is reused and no file is loaded,
because HOC cannot redefine a template; a warning is logged if the current
circuit would resolve the prefix to a different file. While a helper loads,
the bundled HOC directory is appended to ``HOC_LIBRARY_PATH`` so helper
dependencies such as ``RNGSettings.hoc`` resolve; the variable is restored
afterwards.

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
``HOC_LIBRARY_PATH`` or a circuit directory, but a replacement must define
the expected ``<Prefix>Helper`` template and expose a ``synapse`` object. The replacement
also remains responsible for any compiled mechanisms it uses.
