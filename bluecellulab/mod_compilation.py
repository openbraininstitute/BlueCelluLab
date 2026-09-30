# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Discover and compile the NEURON MOD files needed to run a simulation.

This mirrors neurodamus's ``neurodamus.utils.compile_mods``: MOD files are
gathered from one or more directories, fingerprinted, and compiled with
``nrnivmodl`` into a shared library that NEURON can load. Repeated calls with
the same inputs reuse the previous build instead of recompiling.

Like neurodamus, BlueCelluLab bundles the shared "technical" MOD files that
circuits should not have to carry themselves (``bluecellulab/data/mod``, see
`_internal_mods_path`), and includes them in every compilation. Unlike
neurodamus there is no CLI and no CoreNEURON support: compilation is driven
from `bluecellulab.importer` when a `Cell` or `CircuitSimulation` is created.

Two behaviours are deliberately stricter than neurodamus, both because
BlueCelluLab runs against circuits as it finds them rather than against
inputs curated by a launcher:

* Duplicate mechanisms are resolved by the mechanism name each MOD file
  *declares*, not just by filename. Two differently named files can define
  the same mechanism (the legacy ``VecStim.mod`` and the current
  ``vecevent.mod`` both declare ``ARTIFICIAL_CELL VecStim``). ``nrnivmodl``
  compiles such a pair without complaint and NEURON then refuses to load the
  result with "The user defined name already exists".
* MOD files whose mechanism NEURON has already registered are skipped. NEURON
  auto-loads a compiled directory found in the current working directory at
  import time, which is what the ``nrnivmodl`` step in our example notebooks
  produces, and mechanisms cannot be registered twice in one process.

Which copy wins when a circuit and BlueCelluLab both supply a mechanism is
controlled by ``BLUECELLULAB_MOD_PRECEDENCE`` (see `circuit_mods_take_precedence`).
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess  # noqa: S404
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path

import libsonata

logger = logging.getLogger(__name__)

VERSION = 1

# Path within the output/cache dir where gathered mod files are placed.
MOD_FILES_PATH = "mod_files"

# Env var to override the default cache location.
MOD_BUILD_DIR_ENV_VAR = "BLUECELLULAB_MOD_BUILD_DIR"

# Env var selecting which copy wins when a circuit and BlueCelluLab both
# provide the same mechanism. "simulator" (default) or "circuit".
MOD_PRECEDENCE_ENV_VAR = "BLUECELLULAB_MOD_PRECEDENCE"

# How long to wait for another process to finish compiling before giving up.
_LOCK_TIMEOUT_S = 600
_LOCK_POLL_INTERVAL_S = 0.5

# Mechanisms this process had to compile apart from others sharing one of their
# ions, and which are therefore present but inert. See
# `mechanisms_with_split_ion_coupling`.
_SPLIT_ION_MECHANISMS: set[str] = set()

# NMODL declares the name a file provides with exactly one of these keywords.
_MECH_DECL_RE = re.compile(
    r"^[ \t]*(?:SUFFIX|POINT_PROCESS|ARTIFICIAL_CELL)[ \t]+([A-Za-z_]\w*)",
    re.MULTILINE,
)

# Ions a file reads or writes. A mechanism sharing an ion with one in another
# compiled library does not see it, so this is needed to warn about that.
_USEION_RE = re.compile(r"^[ \t]*USEION[ \t]+([A-Za-z_]\w*)", re.MULTILINE)

# COMMENT ... ENDCOMMENT blocks must be stripped before looking for the above,
# or documentation that happens to quote a declaration is picked up as real.
_COMMENT_BLOCK_RE = re.compile(
    r"^[ \t]*COMMENT\b.*?^[ \t]*ENDCOMMENT\b", re.MULTILINE | re.DOTALL
)


class ModCompilationError(Exception):
    """Raised when gathering or compiling MOD files fails."""


# Passed to nrnivmodl by default. Neurodamus's "Class 1" reporting MOD files
# (``SonataReports.mod``, ``SonataReportHelper.mod``) include
# ``bbp/sonata/reports.h``, which is not available here and makes the whole
# compilation fail with "file not found" if one of them is present in a
# circuit's mechanisms directory. They guard that dependency behind
# ``DISABLE_REPORTINGLIB``, so defining it lets such a circuit still be
# simulated. BlueCelluLab does not need those mechanisms in any case: it writes
# SONATA spike and compartment reports with h5py directly.
DEFAULT_INCFLAGS = "-DDISABLE_REPORTINGLIB"


@dataclass
class Options:
    """Options that influence the compiled output (and thus the cache key)."""

    incflags: str = DEFAULT_INCFLAGS
    loadflags: str = ""


def circuit_mods_take_precedence() -> bool:
    """Whether a circuit's MOD file wins over BlueCelluLab's own copy.

    Defaults to ``False``, matching neurodamus, which appends its internal
    MOD directory last so that the simulator's copy overrides the circuit's.
    Set ``BLUECELLULAB_MOD_PRECEDENCE=circuit`` to invert this and keep the
    circuit's copy, for instance to reproduce results from a circuit that
    ships a customised technical MOD file.
    """
    value = os.environ.get(MOD_PRECEDENCE_ENV_VAR, "simulator").strip().lower()
    if value not in ("simulator", "circuit"):
        logger.warning(
            "Ignoring unknown %s=%r, expected 'simulator' or 'circuit'.",
            MOD_PRECEDENCE_ENV_VAR,
            value,
        )
        return False
    return value == "circuit"


def declared_mechanisms(path: str | Path) -> set[str]:
    """Return the mechanism name(s) a MOD file declares.

    A MOD file normally declares exactly one ``SUFFIX``, ``POINT_PROCESS`` or
    ``ARTIFICIAL_CELL``; a set is returned so an unusual file cannot silently
    lose a declaration. An empty set means nothing could be parsed, in which
    case the file is treated as having no mechanism to clash over.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        logger.warning("Could not read mod file %s: %s", path, e)
        return set()
    return set(_MECH_DECL_RE.findall(_COMMENT_BLOCK_RE.sub("", text)))


def declared_ions(path: str | Path) -> set[str]:
    """Return the ion name(s) a MOD file declares with ``USEION``.

    Used to detect the one case where compiling a file separately is not
    safe; see `_warn_on_split_ion_coupling`.
    """
    try:
        text = Path(path).read_text(errors="replace")
    except OSError as e:
        logger.warning("Could not read mod file %s: %s", path, e)
        return set()
    return set(_USEION_RE.findall(_COMMENT_BLOCK_RE.sub("", text)))


def mechanisms_with_split_ion_coupling() -> set[str]:
    """Mechanisms compiled apart from others sharing one of their ions.

    Such a mechanism is present in NEURON but inert: it cannot reach the
    mechanisms it is meant to influence. Callers whose feature depends on one
    of these should refuse to run rather than produce a result that looks fine
    (see ``Cell.enable_ttx``).
    """
    return set(_SPLIT_ION_MECHANISMS)


def _warn_on_split_ion_coupling(
    mod_files: Iterable[Path], already_registered: set[str]
) -> None:
    """Warn when a MOD file using an already-present ion must be compiled
    apart.

    NEURON does not share a custom ion between separately compiled libraries.
    If a mechanism writing the ion ends up in one library and a mechanism
    reading it in another, they simply do not see each other and the
    simulation runs on silently with the coupling absent. ``TTXDynamicsSwitch``
    writing ``ttx`` while ``NaTs2_t`` or ``NaTg`` read it is the case that
    matters in practice.

    NEURON exposes a used ion as a ``<name>_ion`` mechanism, so an ion already
    being present means some library we did not build is already using it.
    """
    for path in mod_files:
        for ion in sorted(declared_ions(path)):
            if f"{ion}_ion" not in already_registered:
                continue
            _SPLIT_ION_MECHANISMS.update(declared_mechanisms(path))
            logger.warning(
                "'%s' uses the '%s' ion, which mechanisms already loaded in NEURON"
                " also use. NEURON cannot share an ion between separately compiled"
                " libraries, so compiling this file on its own would leave the"
                " coupling silently inactive. Compile it together with those"
                " mechanisms instead, by adding %s to your own nrnivmodl"
                " invocation.",
                path.name,
                ion,
                path.parent,
            )


def registered_mechanisms() -> set[str]:
    """Return the mechanisms NEURON has already registered in this process.

    Covers both distributed mechanisms and point processes. This is the
    authoritative check: it reflects whatever NEURON actually loaded,
    including a compiled directory auto-loaded from the current working
    directory, without having to guess at architecture-specific directory
    names.
    """
    import neuron

    names: set[str] = set()
    for is_point_process in (0, 1):
        try:
            mech_type = neuron.h.MechanismType(is_point_process)
            buf = neuron.h.ref("")
            for i in range(int(mech_type.count())):
                mech_type.select(i)
                mech_type.selected(buf)
                names.add(buf[0])
        except Exception as e:  # noqa: BLE001 - NEURON raises bare RuntimeError
            logger.debug("Could not enumerate NEURON mechanisms (%s): %s", is_point_process, e)
    return names


def _metadata_path(output_dir: Path) -> Path:
    """Return the cache metadata path."""
    return output_dir / "modules.json"


def _md5sum(path: Path) -> str:
    """Get the md5sum of the contents of `path`."""
    h = hashlib.md5(usedforsecurity=False)  # noqa: S324
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _warn_conflict(
    winner: Path, loser: Path, internal_dir: Path | None, mechanism: str | None = None
) -> None:
    """Warn that two MOD files provide the same thing, and say which is used.

    When one of them is a technical MOD file BlueCelluLab bundles, the
    message tells circuit owners what to do about it; otherwise it just
    reports which file was chosen, as neurodamus does.
    """
    if internal_dir is None or internal_dir not in (winner.parent, loser.parent):
        # Two circuit files clashing with each other. BlueCelluLab is not
        # involved, so there is nothing to advise about technical mod files.
        logger.warning(
            "The mod files '%s' (%s) and '%s' (%s) both provide %s. Using '%s'.",
            loser.name,
            loser.parent,
            winner.name,
            winner.parent,
            f"mechanism '{mechanism}'" if mechanism else "the same mechanism",
            winner.name,
        )
        return

    if winner.parent == internal_dir:
        circuit_file, internal_file = loser, winner
        precedence_note = "BlueCelluLab's copy takes precedence."
        hint = f" To use the circuit's copy instead, set {MOD_PRECEDENCE_ENV_VAR}=circuit."
    else:
        circuit_file, internal_file = winner, loser
        precedence_note = "The circuit copy takes precedence."
        hint = ""

    if mechanism is None:
        mechanisms = declared_mechanisms(internal_file) or declared_mechanisms(circuit_file)
        mechanism = ", ".join(sorted(mechanisms)) if mechanisms else "(unknown)"

    # Avoid the redundant "supplies as 'X.mod'" when both files share a name.
    also_supplies = (
        "which BlueCelluLab also supplies"
        if circuit_file.name == internal_file.name
        else f"which BlueCelluLab also supplies as '{internal_file.name}'"
    )

    logger.warning(
        "The mod file '%s' in %s provides mechanism '%s', %s. %s Technical mod"
        " files should be removed from circuit mechanisms directories as per OBI"
        " and neurodamus (OBI's largescale circuit simulator) specifications.%s",
        circuit_file.name,
        circuit_file.parent,
        mechanism,
        also_supplies,
        precedence_note,
        hint,
    )


def select_mod_files(
    input_dirs: list[Path],
    internal_dir: Path | None = None,
    already_registered: Iterable[str] | None = None,
) -> dict[Path, str]:
    """Choose which MOD files to compile, with their md5sums.

    `input_dirs` is in increasing order of priority: where two files provide
    the same thing, the one from the later directory wins. Resolution happens
    at two levels, filename first (as neurodamus does) and then the mechanism
    each file declares, so that differently named files defining the same
    mechanism cannot both be compiled.

    Files whose mechanism appears in `already_registered` are dropped, since
    NEURON cannot register a mechanism twice in one process.

    `internal_dir` identifies BlueCelluLab's own bundled directory, and is
    used only to phrase conflict warnings usefully.
    """
    priority: dict[Path, int] = {}
    by_name: dict[str, Path] = {}
    for index, directory in enumerate(input_dirs):
        directory = Path(directory)
        if not directory.is_dir():
            # A circuit that declares a mechanisms_dir which is not there would
            # otherwise silently contribute nothing, and only fail much later
            # with "... is not a MECHANISM".
            logger.warning(
                "Mod file directory %s does not exist, no mod files taken from it.",
                directory,
            )
            continue
        for path in sorted(directory.glob("*.mod")):
            path = path.absolute()
            priority[path] = index
            previous = by_name.get(path.name)
            if previous is not None:
                # Directories are in increasing priority, so `path` wins.
                _warn_conflict(path, previous, internal_dir)
            by_name[path.name] = path

    # Resolve files that declare the same mechanism under different names.
    by_mechanism: dict[str, Path] = {}
    selected: dict[Path, set[str]] = {}
    for path in by_name.values():
        mechanisms = declared_mechanisms(path)
        if not mechanisms:
            logger.debug("No mechanism declaration found in %s, compiling it anyway", path)
            selected[path] = set()
            continue

        superseded = False
        for mechanism in sorted(mechanisms):
            incumbent = by_mechanism.get(mechanism)
            if incumbent is None or incumbent == path:
                continue
            if priority[path] >= priority[incumbent]:
                _warn_conflict(path, incumbent, internal_dir, mechanism)
                selected.pop(incumbent, None)
                for m in list(by_mechanism):
                    if by_mechanism[m] == incumbent:
                        del by_mechanism[m]
            else:
                _warn_conflict(incumbent, path, internal_dir, mechanism)
                superseded = True
        if superseded:
            continue
        for mechanism in mechanisms:
            by_mechanism[mechanism] = path
        selected[path] = mechanisms

    # Drop anything NEURON already has; it cannot be registered again.
    already = set(already_registered or ())
    result: dict[Path, str] = {}
    for path, mechanisms in selected.items():
        clash = mechanisms & already
        if clash:
            logger.debug(
                "Skipping %s: mechanism(s) %s already registered in NEURON",
                path,
                ", ".join(sorted(clash)),
            )
            continue
        result[path] = _md5sum(path)
    return result


def default_mod_build_dir(mod_files: dict[Path, str]) -> Path:
    """Return the cache directory for a given set of mod files.

    Keyed by the names and contents of the files actually being compiled, so
    that two runs compiling different subsets (for instance one that had to
    skip mechanisms NEURON already had) never share a directory and
    invalidate each other's cache. Can be overridden wholesale with the
    ``BLUECELLULAB_MOD_BUILD_DIR`` environment variable.
    """
    if MOD_BUILD_DIR_ENV_VAR in os.environ:
        return Path(os.environ[MOD_BUILD_DIR_ENV_VAR]).absolute()

    key = "|".join(f"{p.name}:{h}" for p, h in sorted(mod_files.items(), key=lambda kv: kv[0].name))
    digest = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324
    return Path.home() / ".cache" / "bluecellulab" / "mods" / digest


def _generate_mod_metadata(mod_files: dict[Path, str], options: Options) -> dict:
    """Create metadata about the compiled mod files, to track cache hits."""
    return {
        "version": VERSION,
        "hashes": sorted([p.name, hash_] for p, hash_ in mod_files.items()),
        **asdict(options),
    }


def _get_dynamic_file(output_dir: Path, name: str) -> Path:
    """Return the path of the compiled file for this machine and platform."""
    base = (output_dir / platform.machine()).absolute()
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    return base / f"{name}{ext}"


def _check_cache(mod_files: dict[Path, str], output_dir: Path, options: Options) -> bool:
    """See if we have a cache hit."""
    metadata = _metadata_path(output_dir)
    if not metadata.exists():
        return False

    with metadata.open() as fd:
        old = json.load(fd)

    new = _generate_mod_metadata(mod_files, options)
    if old != new:
        return False

    return _get_dynamic_file(output_dir, "libnrnmech").exists()


def _write_cache(mod_files: dict[Path, str], output_dir: Path, options: Options) -> None:
    """Write the cache metadata."""
    with _metadata_path(output_dir).open("w") as fd:
        json.dump(_generate_mod_metadata(mod_files, options), fd)


def _place_mod_files(output_dir: Path, mod_files: Iterable[Path]) -> Path:
    """Place mod files in the output directory."""
    mod_dir = (output_dir / MOD_FILES_PATH).absolute()
    mod_dir.mkdir(parents=True, exist_ok=True)
    for f in mod_dir.glob("*.mod"):
        if f.is_file():
            f.unlink()

    for path in mod_files:
        shutil.copy(path, mod_dir)

    return mod_dir


class _CompileLock:
    """A simple cross-process file lock guarding the compilation step.

    Unlike neurodamus (a single launcher process), BlueCelluLab is commonly
    used from multiple worker processes (``NestedPool``, MPI ranks) that may
    try to compile the same circuit's mod files concurrently. This lock makes
    concurrent callers wait for the first compilation to finish and reuse its
    result, rather than racing on ``nrnivmodl`` and corrupting the output.
    """

    def __init__(self, output_dir: Path, timeout: float = _LOCK_TIMEOUT_S):
        self._lock_path = output_dir / ".compile.lock"
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "_CompileLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                if time.monotonic() - start > self._timeout:
                    raise ModCompilationError(
                        f"Timed out waiting for lock {self._lock_path}. "
                        "If a previous compilation crashed, remove this file manually."
                    ) from None
                time.sleep(_LOCK_POLL_INTERVAL_S)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None:
            os.close(self._fd)
        try:
            self._lock_path.unlink()
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise


def _build_mod_files(
    mod_files: dict[Path, str],
    output_dir: Path,
    nrnivmodl_path: str | None,
    options: Options,
) -> Path:
    """Compile `mod_files`, reusing a cached build if they are unchanged.

    Returns the path to the compiled shared library (``libnrnmech``).
    """
    if not mod_files:
        raise ModCompilationError("No mod files found to be compiled")

    output_dir.mkdir(parents=True, exist_ok=True)

    with _CompileLock(output_dir):
        if _check_cache(mod_files, output_dir, options):
            logger.debug("Mod file cache hit in %s, skipping compilation", output_dir)
            return _get_dynamic_file(output_dir, "libnrnmech")

        nrnivmodl = nrnivmodl_path or shutil.which("nrnivmodl")
        if not nrnivmodl:
            raise ModCompilationError("nrnivmodl not found in PATH")

        cmd = [nrnivmodl]
        if options.incflags:
            cmd.extend(["-incflags", options.incflags])
        if options.loadflags:
            cmd.extend(["-loadflags", options.loadflags])

        mod_dir = _place_mod_files(output_dir, mod_files.keys())
        cmd.append(str(mod_dir))

        logger.info("Compiling %d mod file(s) with: %s", len(mod_files), " ".join(cmd))
        # Output is captured rather than inherited: under Jupyter, `sys.stdout`
        # and `sys.stderr` are ipykernel streams with no usable file
        # descriptor, and handing one to a subprocess raises. Capturing also
        # lets the compiler output be reported as part of the failure.
        res = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if res.returncode:
            raise ModCompilationError(
                f"Failed to compile mod files (exit code {res.returncode}):\n"
                f"{res.stdout}"
            )
        logger.debug("nrnivmodl output:\n%s", res.stdout)

        _write_cache(mod_files, output_dir, options)

        libnrnmech = _get_dynamic_file(output_dir, "libnrnmech")
        if not libnrnmech.exists():
            raise ModCompilationError(
                f"{libnrnmech} does not exist after running nrnivmodl, compilation may have failed"
            )
        return libnrnmech


def internal_mods_path() -> Path:
    """Public accessor for BlueCelluLab's bundled technical MOD directory.

    Useful to a caller that compiles mechanisms itself and wants to include
    these files in its own ``nrnivmodl`` invocation, which is the only way to
    get a working ``ttx`` modification when the mechanisms are pre-compiled.
    """
    return _internal_mods_path()


def _internal_mods_path() -> Path:
    """Get the path to BlueCelluLab's own bundled "technical" mod files.

    These are mechanisms that a circuit's own ``mechanisms_dir`` should not
    have to provide (e.g. ``vecevent.mod``), because they are infrastructure
    shared between simulators rather than part of any particular circuit's
    biophysical model. This mirrors neurodamus's
    ``_internal_mods_path``/``--with-internal-mods``: neurodamus bundles
    these same files under ``neurodamus/data/mod/`` and always includes them
    when compiling.

    See https://github.com/openbraininstitute/prod-build-circuit/issues/32
    for the "Class 1 / Class 2" mod file classification this follows.
    """
    spec = find_spec("bluecellulab")
    assert spec
    assert spec.origin
    return Path(spec.origin).parent / "data" / "mod"


def extract_mechanisms_dir(circuit_config_path: str | Path) -> list[Path]:
    """Get the `mechanisms_dir` path(s) declared in a SONATA circuit config."""
    cc = libsonata.CircuitConfig.from_file(str(circuit_config_path))
    paths = set()
    for name in cc.node_populations:
        properties = cc.node_population_properties(name)
        if d := properties.mechanisms_dir:
            paths.add(Path(d).absolute())
    return list(paths)


def compile_mechanisms(
    input_dirs: Iterable[str | Path],
    nrnivmodl_path: str | None = None,
    options: Options | None = None,
    include_internal_mods: bool = True,
    already_registered: Iterable[str] | None = None,
) -> Path | None:
    """Discover and compile the mod files needed on top of what NEURON has.

    `input_dirs` are a circuit's mechanisms directories (see
    `extract_mechanisms_dir`); pass an empty iterable when there is no
    circuit. BlueCelluLab's own bundled technical mod files are included
    unless `include_internal_mods` is ``False``, so circuits need not carry
    them.

    `already_registered` is the set of mechanisms NEURON already knows (see
    `registered_mechanisms`); those are skipped, because NEURON cannot
    register a mechanism twice in one process. Pass ``None`` to query NEURON.

    Returns the path to the compiled shared library, or ``None`` when there
    is nothing left to compile because NEURON already provides everything.
    """
    circuit_dirs = [Path(d) for d in input_dirs]
    internal_dir = _internal_mods_path() if include_internal_mods else None

    # Later directories win. neurodamus appends its internal mods last, so
    # the simulator's copy overrides the circuit's; inverted on request.
    if internal_dir is None:
        ordered = circuit_dirs
    elif circuit_mods_take_precedence():
        ordered = [internal_dir, *circuit_dirs]
    else:
        ordered = [*circuit_dirs, internal_dir]

    if not ordered:
        return None

    if already_registered is None:
        already_registered = registered_mechanisms()

    mod_files = select_mod_files(ordered, internal_dir, already_registered)
    if not mod_files:
        logger.debug("No mod files left to compile, NEURON already has everything needed")
        return None

    _warn_on_split_ion_coupling(mod_files, set(already_registered))

    options = options or Options()
    output_dir = default_mod_build_dir(mod_files)
    return _build_mod_files(mod_files, output_dir, nrnivmodl_path, options)
