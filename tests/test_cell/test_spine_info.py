"""Tests for SpineInfo class."""

import pandas as pd
import pytest

from bluecellulab.cell.spine_info import SpineInfo


class TestSpineInfoFActor:
    """Tests for section_f_factor computation (pure math, no file needed)."""

    def test_f_factor_no_spines(self):
        """f-factor is 1.0 when section has no spines."""
        spine_areas = {2: 10.0}
        info = SpineInfo(
            section_spine_areas=spine_areas,
            spine_count=1,
            spine_table=pd.DataFrame(),
            spine_areas=pd.DataFrame(),
        )
        assert info.section_f_factor(section_id=1, section_area_um2=100.0) == 1.0

    def test_f_factor_with_spines(self):
        """f-factor = (area + spine_area) / area."""
        spine_areas = {1: 50.0}
        info = SpineInfo(
            section_spine_areas=spine_areas,
            spine_count=1,
            spine_table=pd.DataFrame(),
            spine_areas=pd.DataFrame(),
        )
        f = info.section_f_factor(section_id=1, section_area_um2=100.0)
        assert f == pytest.approx(1.5)

    def test_f_factor_zero_area(self):
        """f-factor is 1.0 when section area is zero (avoids division by zero)."""
        info = SpineInfo(
            section_spine_areas={1: 10.0},
            spine_count=1,
            spine_table=pd.DataFrame(),
            spine_areas=pd.DataFrame(),
        )
        assert info.section_f_factor(section_id=1, section_area_um2=0.0) == 1.0

    def test_f_factor_negative_area(self):
        """f-factor is 1.0 when section area is negative."""
        info = SpineInfo(
            section_spine_areas={1: 10.0},
            spine_count=1,
            spine_table=pd.DataFrame(),
            spine_areas=pd.DataFrame(),
        )
        assert info.section_f_factor(section_id=1, section_area_um2=-5.0) == 1.0


class TestSpineInfoFromMorphologyFile:
    """Tests for from_morphology_file (requires morph-spines)."""

    def test_import_error_when_morph_spines_missing(self):
        """Should raise ImportError with helpful message when morph-spines is not installed."""
        # Patch the import to simulate morph-spines not being available
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "morph_spines.utils.morph_spine_loader":
                raise ImportError("No module named 'morph_spines'")
            return original_import(name, *args, **kwargs)

        with pytest.MonkeyPatch().context() as m:
            m.setattr(builtins, "__import__", mock_import)
            with pytest.raises(ImportError, match="morph-spines is required"):
                SpineInfo.from_morphology_file("dummy.h5", "dummy")
