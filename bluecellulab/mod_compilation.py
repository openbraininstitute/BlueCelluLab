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
"""Compile the NEURON MOD files needed to run a simulation.

This mirrors neurodamus's ``neurodamus-compile-mods``: compilation is an
explicit step run before the simulation, never on import. MOD files are
gathered from a circuit's ``mechanisms_dir`` and/or given directories plus
BlueCelluLab's bundled "technical" MOD files (``bluecellulab/data/mod``),
fingerprinted, and compiled with ``nrnivmodl`` into ``<output_dir>/<arch>``.
With the default output directory, the working directory, NEURON loads that
folder automatically on import. Re-running with unchanged files does nothing.

Use the ``bcl-compile-mods`` command or `compile_mods`.

One behaviour is deliberately stricter than neurodamus: duplicate mechanisms
are resolved by the mechanism name each MOD file *declares*, not just by
filename. Two differently named files can define the same mechanism (the
legacy ``VecStim.mod`` and the current ``vecevent.mod`` both declare
``ARTIFICIAL_CELL VecStim``). ``nrnivmodl`` compiles such a pair without
complaint and NEURON then refuses to load the result with "The user defined
name already exists".

Which copy wins when a circuit and BlueCelluLab both supply a mechanism is
controlled by ``BLUECELLULAB_MOD_PRECEDENCE`` (see `circuit_mods_take_precedence`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess  # noqa: S404
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path

import libsonata

logger = logging.getLogger(__name__)

VERSION = 1

# Path within the output/cache dir where gathered mod files are placed.
MOD_FILES_PATH = "mod_files"

# Env var selecting which copy wins when a circuit and BlueCelluLab both
# provide the same mechanism. "simulator" (default) or "circuit".
MOD_PRECEDENCE_ENV_VAR = "BLUECELLULAB_MOD_PRECEDENCE"

# NMODL declares the name a file provides with exactly one of these keywords.
_MECH_DECL_RE = re.compile(
    r"^[ \t]*(?:SUFFIX|POINT_PROCESS|ARTIFICIAL_CELL)[ \t]+([A-Za-z_]\w*)",
    re.MULTILINE,
)

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


def _generate_mod_metadata(mod_files: dict[Path, str], options: Options) -> dict:
    """Create metadata about the compiled mod files, to track cache hits."""
    return {
        "version": VERSION,
        "neuron_version": neuron_version(),
        "hashes": sorted([p.name, hash_] for p, hash_ in mod_files.items()),
        "sources": sorted(str(p) for p in mod_files),
        **asdict(options),
    }


def neuron_version() -> str:
    """Return the version of the NEURON installation in use."""
    import neuron

    return str(neuron.__version__)


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

    if _check_cache(mod_files, output_dir, options):
        logger.info("Mechanisms in %s are up to date, nothing to compile", output_dir)
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

    Useful to a caller that runs ``nrnivmodl`` itself and wants to include
    these files in that invocation.
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
    # Edge populations can override `mechanisms_dir` too (synapse mechanisms
    # often live there). Order matters, since later directories win duplicate
    # mechanisms in `select_mod_files`, so deduplicate while keeping a stable
    # order: node populations, then edge populations, each sorted by name
    # (libsonata returns population names as an unordered set).
    declared = [
        cc.node_population_properties(name).mechanisms_dir
        for name in sorted(cc.node_populations)
    ] + [
        cc.edge_population_properties(name).mechanisms_dir
        for name in sorted(cc.edge_populations)
    ]
    paths: dict[Path, None] = {}
    for d in declared:
        if d:
            paths.setdefault(Path(d).absolute(), None)
    return list(paths)


def _ordered_input_dirs(
    input_dirs: Iterable[str | Path], with_internal_mods: bool
) -> tuple[list[Path], Path | None]:
    """Return the directories to compile from, in increasing priority."""
    dirs = [Path(d).absolute() for d in input_dirs]
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        raise ModCompilationError(
            "Mod file directory does not exist: " + ", ".join(str(d) for d in missing)
        )
    internal_dir = _internal_mods_path().absolute() if with_internal_mods else None
    # Later directories win. neurodamus appends its internal mods last, so
    # the simulator's copy overrides the circuit's; inverted on request.
    if internal_dir is None:
        return dirs, None
    if circuit_mods_take_precedence():
        return [internal_dir, *dirs], internal_dir
    return [*dirs, internal_dir], internal_dir


def compile_mods(
    input_dirs: Iterable[str | Path] = (),
    circuit_config: str | Path | None = None,
    output_dir: str | Path = ".",
    with_internal_mods: bool = True,
    nrnivmodl_path: str | None = None,
    options: Options | None = None,
    load: bool = True,
) -> Path:
    """Compile MOD files into ``<output_dir>/<arch>``.

    Takes the MOD files in `input_dirs` and in the ``mechanisms_dir`` of
    `circuit_config`, plus BlueCelluLab's bundled ones unless
    `with_internal_mods` is ``False``. Nothing is compiled if the previous
    build in `output_dir` came from identical files and options.

    With `load` (the default), the library is also loaded into this process
    if NEURON does not have its mechanisms yet. This makes the function
    usable from a notebook after ``bluecellulab`` was imported. If an older
    build is already loaded, NEURON cannot replace it and a warning asks for
    a restart.

    Returns the path to the compiled library.
    """
    dirs = list(input_dirs)
    if circuit_config is not None:
        dirs.extend(extract_mechanisms_dir(circuit_config))
    ordered, internal_dir = _ordered_input_dirs(dirs, with_internal_mods)
    mod_files = select_mod_files(ordered, internal_dir)
    output_dir = Path(output_dir).absolute()
    options = options or Options()
    up_to_date = _check_cache(mod_files, output_dir, options)
    libnrnmech = _build_mod_files(mod_files, output_dir, nrnivmodl_path, options)
    if load:
        _load_into_running_neuron(libnrnmech, mod_files, rebuilt=not up_to_date)
    return libnrnmech


def _load_into_running_neuron(
    libnrnmech: Path, mod_files: dict[Path, str], rebuilt: bool
) -> None:
    """Load `libnrnmech` unless NEURON already has its mechanisms."""
    if "neuron" not in sys.modules:
        # NEURON loads the folder itself when it is imported.
        return
    mechanisms: set[str] = set()
    for path in mod_files:
        mechanisms |= declared_mechanisms(path)
    registered = registered_mechanisms() & mechanisms
    if not registered:
        sys.modules["neuron"].h.nrn_load_dll(str(libnrnmech))
        logger.info("Loaded %s", libnrnmech)
    elif rebuilt:
        # Loaded before this rebuild, e.g. auto-loaded from the working
        # directory at import: the process runs the old build.
        logger.warning(
            "NEURON already has mechanisms from the previous build loaded and"
            " cannot replace them in a running process. Restart the Python"
            " process or kernel to use the new build in %s.",
            libnrnmech,
        )


def changed_mod_files(output_dir: str | Path = ".") -> list[Path]:
    """Return the source MOD files changed since `output_dir` was compiled.

    Empty if there is no BlueCelluLab build in `output_dir`. A recorded
    source that was deleted counts as changed.
    """
    metadata = _metadata_path(Path(output_dir))
    if not metadata.exists():
        return []
    try:
        recorded = json.loads(metadata.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    hashes = dict(recorded.get("hashes", []))
    changed = []
    for source in map(Path, recorded.get("sources", [])):
        if not source.exists() or _md5sum(source) != hashes.get(source.name):
            changed.append(source)
    return changed


def build_parser() -> argparse.ArgumentParser:
    """Build the ``bcl-compile-mods`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="bcl-compile-mods",
        description="Compile MOD files for BlueCelluLab into <output-dir>/<arch>.",
    )
    parser.add_argument("dirs", nargs="*", help="Directories with MOD files, as for nrnivmodl")
    parser.add_argument("--circuit-config", help="Circuit config for `mechanisms_dir` discovery")
    parser.add_argument(
        "--input-dir", nargs="*", default=[], help="Same as DIRS (neurodamus-compatible)"
    )
    parser.add_argument("--output-dir", default=".", help="Output path (default: .)")
    parser.add_argument(
        "--with-internal-mods",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BlueCelluLab's bundled mods (default: yes)",
    )
    parser.add_argument("--incflags", default=DEFAULT_INCFLAGS, help="`-incflags` for nrnivmodl")
    parser.add_argument("--loadflags", default="", help="`-loadflags` for nrnivmodl")
    parser.add_argument("--nrnivmodl", help="Optional path to nrnivmodl")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``bcl-compile-mods``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    input_dirs = [*args.dirs, *args.input_dir]
    if not input_dirs and not args.circuit_config and not args.with_internal_mods:
        build_parser().error("nothing to compile")
    try:
        libnrnmech = compile_mods(
            input_dirs,
            circuit_config=args.circuit_config,
            output_dir=args.output_dir,
            with_internal_mods=args.with_internal_mods,
            nrnivmodl_path=args.nrnivmodl,
            options=Options(incflags=args.incflags, loadflags=args.loadflags),
            load=False,
        )
    except ModCompilationError as e:
        logger.error("%s", e)
        return 1
    print(libnrnmech)  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
