"""Spine information for morph-spines H5 files.

Provides :class:`SpineInfo`, which loads per-spine surface area data and
skeleton geometry from a morph-spines HDF5 file using the ``morph-spines``
library (optional dependency).  The skeleton geometry enables explicit
spine compartment creation in NEURON, while the surface area data supports
f-factor capacitance adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpineInfo:
    """Spine information for morph-spines H5 files.

    Attributes:
        section_spine_areas: Mapping from morphology section ID to total
            spine surface area (µm²) on that section.
        spine_count: Total number of spines on the morphology.
        spine_table: Raw spine table from morph-spines, containing at
            least ``afferent_section_id``, ``afferent_segment_id``,
            ``afferent_segment_offset``, and ``afferent_section_pos``.
        spine_areas: Per-spine surface areas (neck + head) in µm².
        spine_skeletons: List of NeuroM Neurite objects (one per spine)
            containing the 3D skeleton geometry in global coordinates.
            Used for explicit spine compartment creation in NEURON.
    """

    section_spine_areas: dict[int, float]
    spine_count: int
    spine_table: pd.DataFrame
    spine_areas: pd.DataFrame = field(repr=False)
    spine_skeletons: list[Any] = field(default_factory=list, repr=False)

    @classmethod
    def from_morphology_file(
        cls,
        filepath: str,
        morphology_name: Optional[str] = None,
        load_meshes: bool = True,
    ) -> SpineInfo:
        """Load spine info from a morph-spines H5 file.

        Uses ``morph_spines.load_morphology_with_spines`` to load spine
        skeletons (always) and optionally meshes for surface area
        computation.  Skeletons are used for explicit spine compartment
        creation; mesh-derived areas support f-factor capacitance
        adjustment.

        Args:
            filepath: Path to the morph-spines ``.h5`` file.
            morphology_name: Name of the morphology inside the file.
                If ``None``, the file must contain exactly one morphology.
            load_meshes: If True, load spine meshes and compute per-spine
                surface areas.  If False, only skeletons are loaded
                (much faster) and ``spine_areas`` / ``section_spine_areas``
                will be empty.

        Returns:
            A populated :class:`SpineInfo` instance.

        Raises:
            ImportError: If the ``morph-spines`` package is not installed.
        """
        try:
            from morph_spines.utils.morph_spine_loader import (
                load_morphology_with_spines,
            )
        except ImportError as exc:
            raise ImportError(
                "morph-spines is required to load spine information. "
                "Install it with: pip install morph-spines  "
                "or: pip install bluecellulab[spines]"
            ) from exc

        morph_with_spines = load_morphology_with_spines(
            filepath, morphology_name, load_meshes=load_meshes
        )
        spines = morph_with_spines.spines

        # Load spine skeletons (always available, in global coordinates)
        spine_skeletons = list(spines.spine_skeletons)

        # Build spine table with section position for compartment connection
        table_cols = [
            "afferent_section_id",
            "afferent_segment_id",
            "afferent_segment_offset",
        ]
        if "afferent_section_pos" in spines.spine_table.columns:
            table_cols.append("afferent_section_pos")
        spine_table = spines.spine_table[table_cols].reset_index(drop=True)

        # Compute per-spine surface areas if meshes are available
        if load_meshes:
            spine_area_records = []
            for spine_id in range(spines.spine_count):
                try:
                    neck_area = spines.centered_spine_mesh(
                        spine_id, include_head=False
                    ).area
                except Exception:
                    neck_area = 0.0
                try:
                    head_area = spines.centered_spine_mesh(
                        spine_id, include_neck=False
                    ).area
                except Exception:
                    head_area = 0.0
                spine_area_records.append((neck_area, head_area))

            spine_areas = pd.DataFrame(
                spine_area_records, columns=["neck_area", "head_area"]
            )
            combined = pd.concat([spine_table, spine_areas], axis=1)
            combined["total_area"] = combined["neck_area"] + combined["head_area"]

            section_spine_areas: dict[int, float] = (
                combined.groupby("afferent_section_id")["total_area"]
                .sum()
                .to_dict()
            )
        else:
            spine_areas = pd.DataFrame(
                columns=["neck_area", "head_area", "total_area"]
            )
            combined = spine_table.copy()
            combined["total_area"] = 0.0
            section_spine_areas = {}

        logger.info(
            "Loaded %d spines from %s, %d sections have spine area, "
            "%d skeleton neurites",
            spines.spine_count,
            filepath,
            len(section_spine_areas),
            len(spine_skeletons),
        )

        return cls(
            section_spine_areas=section_spine_areas,
            spine_count=spines.spine_count,
            spine_table=spine_table,
            spine_areas=combined,
            spine_skeletons=spine_skeletons,
        )

    def section_f_factor(self, section_id: int, section_area_um2: float) -> float:
        """Compute the f-factor for a section.

        The f-factor is the ratio of total surface area (dendrite +
        spines) to dendrite-only surface area:

            f = (section_area + spine_area) / section_area

        Args:
            section_id: Morphology section ID.
            section_area_um2: Surface area of the dendritic section
                without spines (µm²).

        Returns:
            f-factor (≥ 1.0).  Returns 1.0 if the section has no spines
            or if ``section_area_um2`` is non-positive.
        """
        spine_area = self.section_spine_areas.get(section_id, 0.0)
        if section_area_um2 <= 0:
            return 1.0
        return (section_area_um2 + spine_area) / section_area_um2
