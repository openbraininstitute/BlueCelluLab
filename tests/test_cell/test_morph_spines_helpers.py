"""Tests for morph-spines H5 detection and path helpers."""

import h5py
import numpy as np

from bluecellulab.cell.morphio_wrapper import (
    MorphIOWrapper,
    is_morph_spines_file,
    morph_spines_morphology_path,
)


class TestIsMorphSpinesFile:
    """Tests for is_morph_spines_file()."""

    def test_synthetic_morph_spines_file(self, synthetic_morph_spines_h5):
        """Should detect a morph-spines H5 file."""
        assert is_morph_spines_file(synthetic_morph_spines_h5) is True

    def test_standard_h5_not_morph_spines(self, tmp_path):
        """A standard H5v1 morphology (root points/structure) is not morph-spines."""
        filepath = tmp_path / "standard.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("points", data=np.zeros((3, 4), dtype=np.float32))
            f.create_dataset("structure", data=np.array([[0, -1, 1]], dtype=np.int32))
        assert is_morph_spines_file(str(filepath)) is False

    def test_nonexistent_file(self):
        """Non-existent file returns False, does not raise."""
        assert is_morph_spines_file("/nonexistent/path/file.h5") is False


class TestMorphSpinesMorphologyPath:
    """Tests for morph_spines_morphology_path()."""

    def test_path_construction(self):
        """Should construct the correct nested path."""
        path = morph_spines_morphology_path("/data/cell.h5", "my_neuron")
        assert path == "/data/cell.h5/morphology/my_neuron"

    def test_path_with_complex_name(self):
        """Should handle morphology names with special characters."""
        path = morph_spines_morphology_path("/data/merged.h5", "864691135839950227")
        assert path == "/data/merged.h5/morphology/864691135839950227"


class TestMorphIOWrapperWithMorphSpines:
    """Tests for MorphIOWrapper loading from a morph-spines H5 file."""

    def test_load_from_synthetic_morph_spines(self, synthetic_morph_spines_h5):
        """MorphIOWrapper should load the skeleton from a morph-spines H5."""
        morph_name = "test_cell"
        nested_path = morph_spines_morphology_path(synthetic_morph_spines_h5, morph_name)
        wrapper = MorphIOWrapper(nested_path)
        cmds = wrapper.morph_as_hoc()
        assert len(cmds) > 0
        assert any("create" in cmd for cmd in cmds)
        assert any("pt3dadd" in cmd for cmd in cmds)

    def test_load_with_h5_suffix(self, synthetic_morph_spines_h5):
        """MorphIOWrapper should also work with .h5 suffix on the path."""
        morph_name = "test_cell"
        nested_path = morph_spines_morphology_path(synthetic_morph_spines_h5, morph_name) + ".h5"
        wrapper = MorphIOWrapper(nested_path)
        cmds = wrapper.morph_as_hoc()
        assert len(cmds) > 0
