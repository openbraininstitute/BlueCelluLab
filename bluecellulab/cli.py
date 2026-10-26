# Copyright 2025 Open Brain Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BlueCelluLab command line interface.

Subcommands are grouped under the single ``bluecellulab`` command so that the
name on PATH always makes the package it belongs to clear. Subcommands
should import only what they need; importing the package itself already
loads NEURON, as in neurodamus.
"""

import argparse
import contextlib
import importlib.metadata
import logging
import sys

import libsonata

from bluecellulab import mod_compilation


@contextlib.contextmanager
def _cli_logging(logger: logging.Logger):
    """Isolated stderr logging for a subcommand, restored afterwards.

    The ``bluecellulab`` logger may already be at ERROR/CRITICAL from
    ``set_verbose_from_env`` (e.g. ``BLUECELLULAB_VERBOSE_LEVEL=0`` in batch
    jobs), which would hide both progress and errors here.
    """
    saved = (logger.handlers[:], logger.level, logger.propagate)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers[:], logger.level, logger.propagate = saved


def _compile_mods(args: argparse.Namespace) -> int:
    """Run the ``compile-mods`` subcommand."""
    input_dirs = [*args.dirs, *args.input_dir]
    if not input_dirs and not args.circuit_config and not args.with_internal_mods:
        args.command_parser.error("nothing to compile")
    try:
        libnrnmech = mod_compilation.compile_mods(
            input_dirs,
            circuit_config=args.circuit_config,
            output_dir=args.output_dir,
            with_internal_mods=args.with_internal_mods,
            nrnivmodl_path=args.nrnivmodl,
            options=mod_compilation.Options(
                incflags=args.incflags, loadflags=args.loadflags
            ),
            load=False,
        )
    except (
        mod_compilation.ModCompilationError,
        libsonata.SonataError,
        OSError,
        RuntimeError,
    ) as e:
        mod_compilation.logger.error("%s", e)
        return 1
    print(libnrnmech)  # noqa: T201
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``bluecellulab`` argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="bluecellulab",
        description="BlueCelluLab command line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('bluecellulab')}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    compile_parser = subparsers.add_parser(
        "compile-mods",
        help="Compile MOD files for BlueCelluLab into <output-dir>/<arch>.",
        description="Compile MOD files for BlueCelluLab into <output-dir>/<arch>.",
    )
    mod_compilation.add_compile_mods_arguments(compile_parser)
    compile_parser.set_defaults(
        handler=_compile_mods,
        command_parser=compile_parser,
        logger=mod_compilation.logger,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``bluecellulab`` command."""
    args = build_parser().parse_args(argv)
    with _cli_logging(args.logger):
        return args.handler(args)
