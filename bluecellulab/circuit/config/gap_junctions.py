# Copyright 2024 Blue Brain Project / EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Config models for gap junctions and continuous (graded) connections.

Gap junctions follow neurodamus: every SONATA edge population declared
with ``"type": "electrical"`` in the circuit config is wired automatically,
one junction mechanism on the target cell per directed edge. The optional
``gap_junctions`` field in the simulation config holds per-population
*overrides* (non-default mechanism/pointer names, conductance scaling);
populations without an override use the neurodamus defaults.

``continuous_connections`` is a BlueCelluLab-only extension for graded
chemical synapses (``POINTER``-driven, no spike threshold) that share the
same pointer-transfer wiring path.
"""
from __future__ import annotations

from typing import Optional

from pydantic import field_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True, config=dict(extra="forbid"))
class GapJunctionConfig:
    """Optional per-population override for a gap-junction edge population.

    Every ``"type": "electrical"`` edge population is wired even without a
    matching override; this block only customizes the mechanism name,
    POINTER name and a conductance scaling factor, in the spirit of the
    neurodamus ``GapJunctionManager``.
    """

    edge_population: str
    mod: str = "Gap"
    pointer_name: str = "vgap"
    weight_factor: float = 1.0
    name: Optional[str] = None


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
