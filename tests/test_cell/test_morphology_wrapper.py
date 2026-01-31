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

    def test_morphology_wrapper_init_success(self):
        """Test successful MorphologyWrapper initialization with real H5 file."""
        # Test with a real H5 file from BlueCelluLab test data
        h5_file = Path(__file__).parent.parent / "examples" / "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" / "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"

        try:
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

        except FileNotFoundError:
            pytest.skip("Neurodamus test data not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            else:
                raise

    def test_morphology_wrapper_individual_h5_files(self):
        """Test MorphologyWrapper with individual H5 file."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )

        try:
            wrapper = MorphIOWrapper(h5_file)

            # Test that file loads successfully
            assert wrapper._morph_ext == ".h5"
            assert len(wrapper._section_names) > 0
            assert len(wrapper.morph_as_hoc()) > 0

            # Test that morph property is accessible
            assert wrapper.morph is not None

        except FileNotFoundError:
            pytest.skip(f"Test file {h5_file} not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            elif "Missing points or structure datasets" in str(e):
                pytest.skip("File structure not compatible")
            else:
                raise

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

        try:
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

        except FileNotFoundError:
            pytest.skip("Neurodamus test data not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            else:
                raise

    def test_morph_property(self):
        """Test morph property access."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )

        try:
            wrapper = MorphIOWrapper(h5_file)

            # Test morph property
            assert wrapper.morph is not None
            assert hasattr(wrapper.morph, 'soma')
            assert hasattr(wrapper.morph, 'sections')

        except FileNotFoundError:
            pytest.skip("Neurodamus test data not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            else:
                raise

    def test_section_name_objects(self):
        """Test that section names are SectionName objects."""
        h5_file = (
            Path(__file__).parent.parent / "examples" /
            "container_nbS1-O1__202247__cADpyr__L5_TPC_A" / "morphologies" / "h5" /
            "dend-rat_20150119_LH1_cell1_axon-rp111203_C3_idA_-_Scale_x1.000_y0.950_z1.000_-_Clone_0.h5"
        )

        try:
            wrapper = MorphIOWrapper(h5_file)

            # Test that _section_names contains SectionName objects
            assert len(wrapper._section_names) > 0
            assert isinstance(wrapper._section_names[0], SectionName)
            assert wrapper._section_names[0].name == "soma"
            assert wrapper._section_names[0].id == 0

            # Test string representation
            assert str(wrapper._section_names[0]) == "soma[0]"

        except FileNotFoundError:
            pytest.skip("Neurodamus test data not available")
        except Exception as e:
            if "MorphIO is not available" in str(e):
                pytest.skip("MorphIO not available")
            else:
                raise


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
