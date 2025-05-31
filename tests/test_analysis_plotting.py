"""Tests for analysis plotting functions."""
import tempfile
from pathlib import Path
from typing import Optional
import matplotlib
import pytest
from bluecellulab.cell import create_ball_stick
from bluecellulab.analysis.plotting import generate_cell_thumbnail

matplotlib.use('Agg')


def test_generate_cell_thumbnail() -> None:
    """Test the generate_cell_thumbnail function with a simple ball-stick model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "cell_thumbnail.png"
        cell = create_ball_stick()

        time, voltage = generate_cell_thumbnail(cell, output_path=output_path)

        # Verify file was created and has content
        assert output_path.exists(), "Thumbnail file was not created"
        assert output_path.stat().st_size > 0, "Thumbnail file is empty"

        # Verify return values
        assert len(time) > 0, "Time array is empty"
        assert len(voltage) > 0, "Voltage array is empty"
        assert len(time) == len(voltage), "Time and voltage arrays have different lengths"

def test_generate_cell_thumbnail_default_output() -> None:
    """Test that the function works with default output path."""
    default_path = Path("cell_thumbnail.png")
    try:
        cell = create_ball_stick()
        time, voltage = generate_cell_thumbnail(cell)

        # Verify file was created and has content
        assert default_path.exists(), "Default thumbnail file was not created"
        assert default_path.stat().st_size > 0, "Default thumbnail file is empty"

        # Verify return values
        assert len(time) > 0, "Time array is empty"
        assert len(voltage) > 0, "Voltage array is empty"
    finally:
        # Clean up
        if default_path.exists():
            default_path.unlink()

@pytest.mark.parametrize("threshold_value", [0, None])
def test_generate_cell_thumbnail_without_threshold(threshold_value: Optional[float]) -> None:
    """Test the function when cell threshold needs to be calculated."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "cell_no_threshold.png"
        cell = create_ball_stick()

        # Handle threshold setup for different cell types
        has_emodel = (hasattr(cell, 'template_params') and
                     hasattr(cell.template_params, 'emodel_properties') and
                     cell.template_params.emodel_properties is not None)

        if not has_emodel:
            cell.threshold = 0.1 # nA
        else:
            if hasattr(cell, 'threshold'):
                delattr(cell, 'threshold')
            if threshold_value is not None:
                cell.threshold = threshold_value

        # Test the thumbnail generation
        time, voltage = generate_cell_thumbnail(cell, output_path=output_path)

        # Verify results
        assert output_path.exists(), "Thumbnail file was not created"
        assert output_path.stat().st_size > 0, "Thumbnail file is empty"
        assert len(time) > 0, "Time array is empty"
        assert len(voltage) > 0, "Voltage array is empty"
