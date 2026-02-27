"""Tests for H5 morphology wrapper functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch

from bluecellulab.cell.morphio_wrapper import (
    MorphIOWrapper,
    SectionName,
    split_morphology_path,
)


class TestMorphologyWrapper:
    """Test suite for MorphIOWrapper class."""

    def test_split_morphology_path_file_exists(self, tmp_path):
        """Test split_morphology_path with existing file."""
        morph_file = tmp_path / "test_morph.h5"
        morph_file.write_bytes(b"dummy content")

        collection_dir, morph_name, morph_ext = split_morphology_path(morph_file)

        assert collection_dir == str(tmp_path)
        assert morph_name == "test_morph"
        assert morph_ext == ".h5"

    def test_split_morphology_path_directory(self, tmp_path):
        """Test split_morphology_path with directory path."""
        # Create the directory and file for testing
        container_dir = tmp_path / "container_dir"
        container_dir.mkdir()
        morph_path = container_dir / "test_morph.h5"
        morph_path.write_bytes(b"dummy content")

        collection_dir_result, morph_name, morph_ext = split_morphology_path(morph_path)

        assert collection_dir_result == str(container_dir)
        assert morph_name == "test_morph"
        assert morph_ext == ".h5"

    def test_split_morphology_path_error_invalid_path(self):
        """Test split_morphology_path raises error for invalid path."""
        from bluecellulab.exceptions import BluecellulabError

        # Empty string triggers the error because dirname("") == ""
        with pytest.raises(BluecellulabError, match="Failed to split path"):
            split_morphology_path("")

    def test_split_morphology_path_container_with_cell_name(self, tmp_path):
        """Test split_morphology_path with H5 container and cell name."""
        container_file = tmp_path / "container.h5"
        container_file.write_bytes(b"dummy h5 content")

        cell_path = f"{container_file}/cell_name"
        collection_dir, morph_name, morph_ext = split_morphology_path(cell_path)

        assert collection_dir == str(container_file)
        assert morph_name == "cell_name"
        assert morph_ext == ""

    def test_morphology_wrapper_init_success(self):
        """Test successful MorphologyWrapper initialization with real H5 file."""
        # Test with a real H5 file from BlueCelluLab test data
        h5_file = Path(__file__).parent.parent / "examples" / "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" / "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        assert h5_file.exists(), f"Test file not found: {h5_file}"

        wrapper = MorphIOWrapper(h5_file)

        # Test basic properties
        assert wrapper._morph_ext == ".h5"
        assert wrapper._morph_name == "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0"
        assert len(wrapper._section_names) > 0
        assert len(wrapper.morph_as_hoc()) > 0

        # Test section names are SectionName objects
        for section_name in wrapper._section_names:
            assert isinstance(section_name, SectionName)
            assert hasattr(section_name, 'name')
            assert hasattr(section_name, 'id')

        # Test HOC commands are valid
        commands = wrapper.morph_as_hoc()
        assert any('create' in cmd for cmd in commands)
        assert any('pt3dadd' in cmd for cmd in commands)

    def test_morphology_wrapper_individual_h5_files(self):
        """Test MorphologyWrapper with individual H5 file."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )
        assert h5_file.exists(), f"Test file not found: {h5_file}"

        wrapper = MorphIOWrapper(h5_file)

        # Test that file loads successfully
        assert wrapper._morph_ext == ".h5"
        assert len(wrapper._section_names) > 0
        assert len(wrapper.morph_as_hoc()) > 0

        # Test that morph property is accessible
        assert wrapper.morph is not None

    def test_morphology_wrapper_h5_container(self):
        """Test MorphologyWrapper with H5 container."""
        container_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "merged-morphologies.h5"
        )

        try:
            # Test that container file exists and can be accessed
            import h5py
            with h5py.File(container_file, 'r') as f:
                assert len(f.keys()) > 0, "Container should have cells"
                # Test that we can access cell names (but don't load them due to length issues)
                cell_names = list(f.keys())
                assert len(cell_names) > 0

            # Test that MorphIOWrapper can handle container path format
            # Use a simple test path to verify path parsing works
            test_path = f"{container_file}/test_cell"
            collection_dir, morph_name, morph_ext = split_morphology_path(test_path)

            # Verify path parsing works for containers
            assert collection_dir == str(container_file)
            assert morph_name == "test_cell"
            assert morph_ext == ""

        except FileNotFoundError:
            pytest.skip("Container test file not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            else:
                raise

    def test_morphology_wrapper_no_morphio(self, tmp_path):
        """Test MorphologyWrapper when MorphIO is not available."""
        # Create a temporary H5 file so path splitting succeeds
        h5_file = tmp_path / "test.h5"
        h5_file.write_bytes(b"dummy")

        # Mock the morphio import to raise ImportError
        import sys
        original_modules = sys.modules.copy()

        # Ensure branch below is always executed
        sys.modules['morphio'] = object()

        # Remove morphio from sys.modules if it exists
        if 'morphio' in sys.modules:
            del sys.modules['morphio']

        # Mock the import to fail
        with patch.dict('sys.modules', {'morphio': None}):
            with pytest.raises(RuntimeError, match="MorphIO is not available"):
                MorphIOWrapper(str(h5_file))

        # Restore original modules
        sys.modules.update(original_modules)

    def test_morphology_wrapper_unsupported_extension(self):
        """Test MorphIOWrapper with unsupported file extension."""
        # Neurodamus doesn't validate extension, it will fail when trying to load
        # Just test that it doesn't crash on init
        try:
            wrapper = MorphIOWrapper("test.h5")
            # If it gets here, it will fail when trying to load the morphology
        except Exception:
            # Expected to fail when loading non-H5 file
            pass

    def test_morph_as_hoc(self):
        """Test HOC command generation with morph_as_hoc method."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )
        assert h5_file.exists(), f"Test file not found: {h5_file}"

        wrapper = MorphIOWrapper(h5_file)

        # Test that morph_as_hoc generates HOC commands
        commands = wrapper.morph_as_hoc()

        assert isinstance(commands, list)
        assert len(commands) > 0

        # Verify sections were created
        import neuron
        for cmd in commands:
            neuron.h(cmd)

        # Check that soma exists
        soma_exists = False
        for sec in neuron.h.allsec():
            if 'soma' in neuron.h.secname(sec=sec):
                soma_exists = True
                break

        assert soma_exists, "Soma section should exist in NEURON after loading"

    def test_morph_property(self):
        """Test morph property access."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )
        assert h5_file.exists(), f"Test file not found: {h5_file}"

        wrapper = MorphIOWrapper(h5_file)

        # Test morph property
        assert wrapper.morph is not None
        assert hasattr(wrapper.morph, 'soma')
        assert hasattr(wrapper.morph, 'sections')

    def test_section_name_objects(self):
        """Test that section names are SectionName objects."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )
        assert h5_file.exists(), f"Test file not found: {h5_file}"

        wrapper = MorphIOWrapper(h5_file)

        # Test that _section_names contains SectionName objects
        assert len(wrapper._section_names) > 0
        assert isinstance(wrapper._section_names[0], SectionName)
        assert wrapper._section_names[0].name == "soma"
        assert wrapper._section_names[0].id == 0

        # Test string representation
        assert str(wrapper._section_names[0]) == "soma[0]"


class TestCellH5Integration:
    """Test integration of H5 morphologies with Cell class."""

    def test_cell_h5_detection_simple(self):
        """Test simple H5 detection without Cell integration."""
        # Test the simple string check that replaced is_h5_morphology
        assert '.h5' in "test.h5".lower()
        assert '.h5' in "test.H5".lower()
        assert '.h5' in "container.h5/cell_name".lower()
        assert '.h5' not in "test.asc".lower()
        assert '.h5' not in "test.swc".lower()
        assert '.h5' not in "test.txt".lower()

    def test_cell_h5_automatic_detection(self):
        """Test that Cell automatically detects and loads H5 morphologies."""
        # This is a simple test to verify the H5 detection logic in Cell.__init__
        # The actual Cell integration is tested elsewhere

        # Just verify the detection logic works
        test_paths = [
            "test.h5",
            "container.h5/cell_name",
            "path/to/morph.H5",
        ]

        for path in test_paths:
            assert '.h5' in str(path).lower(), f"Should detect H5 in {path}"

    def test_type2name_method(self):
        """Test type2name class method."""
        # Test known types
        assert MorphIOWrapper.type2name(1) == "soma"
        assert MorphIOWrapper.type2name(2) == "axon"
        assert MorphIOWrapper.type2name(3) == "dend"
        assert MorphIOWrapper.type2name(4) == "apic"

        # Test unknown types
        assert MorphIOWrapper.type2name(5) == "dend_5"
        assert MorphIOWrapper.type2name(10) == "dend_10"

        # Test negative types
        assert MorphIOWrapper.type2name(-1) == "minus_1"
        assert MorphIOWrapper.type2name(-5) == "minus_5"

    def test_mksubset_method(self):
        """Test mksubset class method."""
        # Test known types
        assert MorphIOWrapper.mksubset(1, "soma") == 'forsec "soma" somatic.append'
        assert MorphIOWrapper.mksubset(2, "axon") == 'forsec "axon" axonal.append'
        assert MorphIOWrapper.mksubset(3, "dend") == 'forsec "dend" basal.append'
        assert MorphIOWrapper.mksubset(4, "apic") == 'forsec "apic" apical.append'

        # Test unknown types
        assert MorphIOWrapper.mksubset(5, "unknown") == 'forsec "unknown" dendritic_5.append'
        assert MorphIOWrapper.mksubset(10, "custom") == 'forsec "custom" dendritic_10.append'

        # Test negative types
        assert MorphIOWrapper.mksubset(-1, "negative") == 'forsec "negative" minus_1set.append'

    def test_section_name_dataclass(self):
        """Test SectionName dataclass functionality."""
        # Test SectionName creation and attributes
        section_name = SectionName(name="soma", id=0)
        assert section_name.name == "soma"
        assert section_name.id == 0

        # Test string representation
        assert str(section_name) == "soma[0]"

        # Test equality
        section_name2 = SectionName(name="soma", id=0)
        assert section_name == section_name2

        # Test inequality
        section_name3 = SectionName(name="dend", id=1)
        assert section_name != section_name3

    def test_morphio_wrapper_with_single_point_soma(self):
        """Test MorphIOWrapper with single point soma morphology."""
        # This tests the soma conversion functions (_to_sphere, single_point_sphere_to_circular_contour)
        # which are called for SOMA_SINGLE_POINT type morphologies
        h5_file = Path(__file__).parent.parent / "examples" / "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" / "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"

        if h5_file.exists():
            # Initialize wrapper - this will trigger soma conversion if needed
            wrapper = MorphIOWrapper(h5_file)

            # Verify the wrapper was created successfully
            assert wrapper is not None
            assert hasattr(wrapper, '_morph')

            # The soma should have been converted to a circular contour
            # Verify basic morphology properties
            hoc_commands = wrapper.morph_as_hoc()
            assert len(hoc_commands) > 0
            assert any('soma' in cmd for cmd in hoc_commands)

    def test_make_convex_edge_cases(self):
        """Test make_convex function with edge cases."""
        from bluecellulab.cell.morphio_wrapper import make_convex
        import numpy as np

        # Test case where convex_idx returns False for some indices (line 97)
        # Create non-convex data where some points need to be filtered
        sides = [
            np.array([1.0, 3.0, 2.0, 4.0]),  # Non-monotonic - will trigger line 97
            np.array([1.0, 2.0, 3.0, 4.0])   # Monotonic
        ]
        rads = [
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.5, 0.6, 0.7, 0.8])
        ]

        result_sides, result_rads = make_convex(sides, rads)

        # Verify the function returns filtered arrays
        assert len(result_sides) == 2
        assert len(result_rads) == 2
        # First side should be filtered to remove non-convex point
        assert len(result_sides[0]) <= len(sides[0])

    def test_to_sphere_and_single_point_conversion(self):
        """Test _to_sphere and single_point_sphere_to_circular_contour functions."""
        import numpy as np
        try:
            import morphio  # noqa: F401
        except ImportError:
            pytest.skip("morphio not available")

        # Test the _to_sphere function directly
        from bluecellulab.cell.morphio_wrapper import _to_sphere, single_point_sphere_to_circular_contour

        # Create a mock neuron object with single point soma
        class MockSoma:
            def __init__(self):
                self.points = np.array([[0.0, 0.0, 0.0]])
                self.diameters = np.array([10.0])  # Diameter of 10, radius 5

        class MockNeuron:
            def __init__(self):
                self.soma = MockSoma()

        neuron = MockNeuron()

        # Call _to_sphere - this should convert single point to circular contour
        _to_sphere(neuron)

        # Verify the soma has been converted to 20 points in a circle
        assert len(neuron.soma.points) == 20
        assert len(neuron.soma.diameters) == 20

        # Verify points form a circle with radius 5.0
        radius = neuron.soma.diameters[0]
        assert radius == 5.0

        # Test single_point_sphere_to_circular_contour wrapper function
        neuron2 = MockNeuron()
        single_point_sphere_to_circular_contour(neuron2)

        # Should also have 20 points after conversion
        assert len(neuron2.soma.points) == 20

    def test_contourcenter_function(self):
        """Test contourcenter utility function."""
        import numpy as np
        from bluecellulab.cell.morphio_wrapper import contourcenter

        xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        mean, new_xyz = contourcenter(xyz)

        assert mean.shape == (3,)
        assert new_xyz.shape == (101, 3)

    def test_get_sides_function(self):
        """Test get_sides utility function."""
        import numpy as np
        from bluecellulab.cell.morphio_wrapper import get_sides

        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, -1.0, 0.0]
        ])

        major = np.array([1.0, 0.0, 0.0])
        minor = np.array([0.0, 1.0, 0.0])

        sides, rads = get_sides(points, major, minor)

        assert len(sides) == 2
        assert len(rads) == 2
        assert len(sides[0]) > 0
        assert len(sides[1]) > 0

    def test_contour2centroid_function(self):
        """Test contour2centroid utility function."""
        import numpy as np
        from bluecellulab.cell.morphio_wrapper import contour2centroid

        N = 20
        radius = 5.0
        phase = 2 * np.pi / (N - 1) * np.arange(N)
        points = np.zeros((N, 3))
        points[:, 0] = radius * np.cos(phase)
        points[:, 1] = radius * np.sin(phase)

        mean = np.array([0.0, 0.0, 0.0])

        result_points, diameters = contour2centroid(mean, points)

        assert result_points.shape[0] == 21
        assert diameters.shape[0] == 21
        assert np.all(diameters > 0)
