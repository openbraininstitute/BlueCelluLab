# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Config models for gap junctions and continuous (graded) connections.

Both blocks share the same NEURON wiring path
(``ParallelContext.source_var`` / ``target_var``); they differ only in
whether the coupling is symmetric (gap junctions: True) or directional
(graded chemical synapses: False).
"""
from __future__ import annotations

from typing import Optional

from pydantic import field_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=dict(extra="forbid"))
class GapJunctionConfig:
    """Configuration for a gap-junction edge population.

    Mirrors the neurodamus ``GapJunctionManager`` block.
    """

    edge_population: str
    node_set: str = "All"
    weight_factor: float = 1.0
    mod: str = "Gap"
    pointer_name: str = "vgap"
    name: Optional[str] = None
    # When True (default), each unordered pair is wired once but materializes
    # bidirectional pointer transfer (each cell broadcasts its v and receives
    # the other's v). For non-symmetric pointer connections, prefer
    # ``ContinuousConnectionConfig`` instead.
    symmetric: bool = True


@dataclass(frozen=True, config=dict(extra="forbid"))
class ContinuousConnectionConfig:
    """Configuration for a continuous (graded) chemical-synapse population.

    Uses the same ``ParallelContext.source_var`` / ``target_var`` pointer
    transfer as gap junctions, but coupling is one-way (source -> target).
    """

    name: str
    edge_population: str
    source: str = "All"
    target: str = "All"
    mod: str = "neuron_to_neuron_exc_syn"
    pointer_name: str = "vpre"
    weight_factor: float = 1.0
    symmetric: bool = False

    @field_validator("symmetric")
    @classmethod
    def _no_symmetric(cls, value):
        if value:
            raise ValueError(
                "ContinuousConnectionConfig.symmetric must be False; use "
                "GapJunctionConfig for symmetric coupling."
            )
        return value
