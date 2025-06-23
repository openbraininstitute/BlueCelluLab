"""Tests for analysis plotting functions."""
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import pytest
import numpy as np
from bluecellulab.analysis.plotting import generate_cell_thumbnail
from bluecellulab.circuit.circuit_access.definition import EmodelProperties

# Test data paths
TEST_DATA_DIR = (Path(__file__).parent / "examples" 
                 / "circuit_sonata_quick_scx" / "components")
HOC_FILE = TEST_DATA_DIR / "hoc" / "cADpyr_L2TPC.hoc"
MORPH_FILE = TEST_DATA_DIR / "morphologies" / "asc" / "rr110330_C3_idA.asc"

# Fixtures
@pytest.fixture(scope="module")
def emodel_properties() -> EmodelProperties:
    """Fixture providing default emodel properties for testing."""
    return EmodelProperties(threshold_current=1.1433533430099487,
                            holding_current=1.4146618843078613,
                            AIS_scaler=1.4561502933502197,
                            soma_scaler=1.0)

@pytest.fixture
def output_dir():
    """Fixture providing a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

class TestGenerateCellThumbnail:
    """Test cases for generate_cell_thumbnail function."""

    def test_with_explicit_output_path(self, output_dir: Path, emodel_properties: EmodelProperties):
        """Test thumbnail generation with explicit output path."""
        # Skip test if required files don't exist
        if not HOC_FILE.exists() or not MORPH_FILE.exists():
            pytest.skip("Required test files not found")
            
        # Arrange
        output_path = output_dir / "cell_thumbnail.png"
        
        # Act
        time, voltage = generate_cell_thumbnail(
            template_path=str(HOC_FILE),
            morphology_path=str(MORPH_FILE),
            template_format="v6",
            emodel_properties=emodel_properties,
            output_path=output_path
        )

        # Assert
        assert output_path.exists(), "Thumbnail file was not created"
        assert output_path.stat().st_size > 0, "Thumbnail file is empty"
        assert len(time) == len(voltage) > 0, "Time and voltage arrays should have matching lengths"
        assert isinstance(time, np.ndarray), "Time should be a numpy array"
        assert isinstance(voltage, np.ndarray), "Voltage should be a numpy array"

    def test_with_default_output_path(self, tmp_path, emodel_properties: EmodelProperties):
        """Test thumbnail generation with default output path."""
        # Skip test if required files don't exist
        if not HOC_FILE.exists() or not MORPH_FILE.exists():
            pytest.skip("Required test files not found")
            
        # Arrange
        default_path = Path("cell_thumbnail.png")
        
        try:
            # Act
            time, voltage = generate_cell_thumbnail(
                template_path=str(HOC_FILE),
                morphology_path=str(MORPH_FILE),
                template_format="v6",
                emodel_properties=emodel_properties
            )

            # Assert
            assert default_path.exists(), "Default thumbnail file was not created"
            assert default_path.stat().st_size > 0, "Default thumbnail file is empty"
            assert len(time) > 0 and len(voltage) > 0, "Output arrays should not be empty"
        finally:
            # Cleanup
            if default_path.exists():
                default_path.unlink()

    @pytest.mark.parametrize("threshold_value", [0, None])
    def test_without_threshold(self, output_dir: Path, emodel_properties: EmodelProperties, threshold_value: Optional[float]):
        """Test thumbnail generation when threshold needs to be calculated."""
        # Skip test if required files don't exist
        if not HOC_FILE.exists() or not MORPH_FILE.exists():
            pytest.skip("Required test files not found")
            
        # Arrange
        output_path = output_dir / "cell_no_threshold.png"
        
        # Create a copy of emodel_properties with no threshold
        test_props = EmodelProperties(
            threshold_current=0.0 if threshold_value == 0 else 0.1,  # Will trigger calculate_rheobase
            holding_current=emodel_properties.holding_current,
            AIS_scaler=emodel_properties.AIS_scaler,
            soma_scaler=emodel_properties.soma_scaler
        )

        # Act
        time, voltage = generate_cell_thumbnail(
            template_path=str(HOC_FILE),
            morphology_path=str(MORPH_FILE),
            template_format="v6",
            emodel_properties=test_props,
            output_path=output_path
        )

        # Assert
        assert output_path.exists(), "Thumbnail file was not created"
        assert output_path.stat().st_size > 0, "Thumbnail file is empty"
        assert len(time) > 0 and len(voltage) > 0, "Output arrays should not be empty"

    @pytest.mark.parametrize("show_figure", [True, False])
    def test_show_figure_parameter(self, output_dir: Path, emodel_properties: EmodelProperties, show_figure: bool):
        """Test that show_figure parameter works as expected."""
        # Skip test if required files don't exist
        if not HOC_FILE.exists() or not MORPH_FILE.exists():
            pytest.skip("Required test files not found")
            
        # Arrange
        output_path = output_dir / f"show_figure_{show_figure}.png"
        
        # Act
        time, voltage = generate_cell_thumbnail(
            template_path=str(HOC_FILE),
            morphology_path=str(MORPH_FILE),
            template_format="v6",
            emodel_properties=emodel_properties,
            output_path=output_path,
            show_figure=show_figure
        )

        # Assert
        assert output_path.exists(), "Thumbnail file was not created"
        assert output_path.stat().st_size > 0, "Thumbnail file is empty"
