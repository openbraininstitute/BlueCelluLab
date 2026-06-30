from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PointProcessConnParameters:
    """Point-neuron connection parameters (Allen-style / Neurodamus mirror)."""

    sgid: int        # source gid
    delay: float     # ms
    weight: float    # NetCon weight
