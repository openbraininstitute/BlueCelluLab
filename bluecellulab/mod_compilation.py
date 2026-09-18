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
"""Discover and compile NEURON MOD files declared by a SONATA circuit.

This mirrors the approach used by neurodamus's
``neurodamus.utils.compile_mods`` module: MOD files referenced by a
circuit's ``mechanisms_dir`` are gathered, fingerprinted, and compiled with
``nrnivmodl`` into a shared library that NEURON can load directly. Repeated
calls with the same inputs and options reuse the previously compiled
library instead of recompiling.

This module is intentionally scoped to SONATA circuits: unlike neurodamus,
BlueCelluLab has no natural "install directory" to build into and does not
bundle its own MOD files, so there is no equivalent of neurodamus's
``--with-internal-mods`` and no CoreNEURON support.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import platform
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

# How long to wait for another process to finish compiling before giving up.
_LOCK_TIMEOUT_S = 600
_LOCK_POLL_INTERVAL_S = 0.5


class ModCompilationError(Exception):
    """Raised when gathering or compiling MOD files fails."""


@dataclass
class Options:
    """Options that influence the compiled output (and thus the cache key)."""

    incflags: str = ""
    loadflags: str = ""


def default_mod_build_dir(input_dirs: Iterable[Path]) -> Path:
    """Return the default cache directory for a given set of mod dirs.

    Keyed by a hash of the (sorted, absolute) input directories so that
    different circuits get independent caches. Can be overridden wholesale
    with the ``BLUECELLULAB_MOD_BUILD_DIR`` environment variable.
    """
    if MOD_BUILD_DIR_ENV_VAR in os.environ:
        return Path(os.environ[MOD_BUILD_DIR_ENV_VAR]).absolute()

    key = "|".join(sorted(str(Path(d).absolute()) for d in input_dirs))
    digest = hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]  # noqa: S324
    return Path.home() / ".cache" / "bluecellulab" / "mods" / digest


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


def _get_mod_files(input_dirs: list[Path]) -> dict[Path, str]:
    """Get all the mod files, with their md5sum.

    Note: files with the same name; the last one "wins".
    """
    files: dict[str, Path] = {}
    for d in input_dirs:
        if not Path(d).is_dir():
            continue
        for f in sorted(Path(d).glob("*.mod")):
            if f.name in files:
                logger.warning(
                    "Already seen `%s` (%s), overriding with `%s`",
                    f.name,
                    files[f.name],
                    f.absolute(),
                )
            files[f.name] = f.absolute()

    hashed_files: dict[Path, str] = {}
    seen_hashes: set[str] = set()
    for p in files.values():
        hashed_files[p] = _md5sum(p)
        if hashed_files[p] in seen_hashes:
            logger.warning("Already added a file with the same contents: %s", hashed_files[p])
        seen_hashes.add(hashed_files[p])
    return hashed_files


def _generate_mod_metadata(mod_files: dict[Path, str], options: Options) -> dict:
    """Create metadata about the compiled mod files, used to track cache hits."""
    return {
        "version": VERSION,
        "hashes": sorted([p.name, hash_] for p, hash_ in mod_files.items()),
        **asdict(options),
    }


def _get_dynamic_file(output_dir: Path, name: str) -> Path:
    """Return the path of the compiled file for the current machine/platform."""
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
    input_dirs: list[Path], output_dir: Path, nrnivmodl_path: str | None, options: Options
) -> Path:
    """Compile the mod files, reusing a cached build if inputs are unchanged.

    Returns the path to the compiled shared library (``libnrnmech``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mod_files = _get_mod_files(input_dirs)
    if not mod_files:
        raise ModCompilationError("No mod files found to be compiled")

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
        res = subprocess.run(  # noqa: S603
            cmd, cwd=str(output_dir), stdout=sys.stderr, check=False
        )
        if res.returncode:
            raise ModCompilationError(f"Failed to compile mod files (exit code {res.returncode})")

        _write_cache(mod_files, output_dir, options)

        libnrnmech = _get_dynamic_file(output_dir, "libnrnmech")
        if not libnrnmech.exists():
            raise ModCompilationError(
                f"{libnrnmech} does not exist after running nrnivmodl, compilation may have failed"
            )
        return libnrnmech


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
) -> Path | None:
    """Discover and compile the mod files found in `input_dirs`.

    This is the generic entry point used by the importer once a set of mod
    directories has been resolved (e.g. via `extract_mechanisms_dir` for a
    SONATA circuit). BlueCelluLab's own bundled "technical" mod files (see
    `_internal_mods_path`) are always included unless `include_internal_mods`
    is set to ``False``, so circuits do not need to carry them in their own
    `mechanisms_dir`.

    Returns the path to the compiled shared library, or ``None`` if there
    are no mod files to compile at all (`input_dirs` is empty and internal
    mods are excluded).
    """
    dirs = [Path(d) for d in input_dirs]
    if include_internal_mods:
        dirs.append(_internal_mods_path())

    if not dirs:
        return None

    options = options or Options()
    output_dir = default_mod_build_dir(dirs)
    return _build_mod_files(dirs, output_dir, nrnivmodl_path, options)
