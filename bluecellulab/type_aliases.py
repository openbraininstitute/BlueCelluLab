"""Type aliases used within the package."""
from __future__ import annotations

from typing import Any, Dict

from neuron import h as hoc_type
from typing_extensions import TypeAlias

HocObjectType: TypeAlias = hoc_type   # until NEURON is typed, most NEURON types are this
NeuronRNG: TypeAlias = hoc_type
NeuronVector: TypeAlias = hoc_type
NeuronSection: TypeAlias = hoc_type
TStim: TypeAlias = hoc_type

SectionMapping = Dict[str, NeuronSection]
SiteEntry: TypeAlias = dict[str, Any]
