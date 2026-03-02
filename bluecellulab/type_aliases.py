"""Type aliases used within the package."""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, TypedDict

from neuron import h as hoc_type
from typing_extensions import TypeAlias

HocObjectType: TypeAlias = hoc_type   # until NEURON is typed, most NEURON types are this
NeuronRNG: TypeAlias = hoc_type
NeuronVector: TypeAlias = hoc_type
NeuronSection: TypeAlias = hoc_type
TStim: TypeAlias = hoc_type

SectionMapping = Dict[str, NeuronSection]
class SiteEntry(TypedDict):
    report: str
    rec_name: str
    section: str
    segx: float

class ReportSite(NamedTuple):
    section: Optional[NeuronSection]
    section_name: str
    segx: float