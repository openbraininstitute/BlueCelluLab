"""Shared fixtures for morph-spines tests."""

import h5py
import numpy as np
import pytest


def create_synthetic_morph_spines_h5(filepath, morphology_name="test_cell"):
    """Create a minimal morph-spines H5 file for testing.

    The file contains:
    - /morphology/{name}/points and /structure (MorphIO-loadable skeleton)
    - /edges/{name}/ with a small spine table
    - /spines/skeletons/{name}/ with minimal spine skeleton data

    This avoids needing the large real morph-spines file in the repo.
    """
    with h5py.File(filepath, "w") as f:
        # Morphology skeleton: a simple soma (3-point contour) + one dendrite
        morph_grp = f.create_group(f"morphology/{morphology_name}")
        # Points: [x, y, z, diameter] — soma (3-point non-collinear contour) + dendrite (2 points)
        points = np.array([
            [5.0, 0.0, 0.0, 10.0],    # soma point 1
            [-5.0, 0.0, 0.0, 10.0],   # soma point 2
            [0.0, 5.0, 0.0, 10.0],    # soma point 3 (non-collinear)
            [0.0, 0.0, 0.0, 2.0],     # dendrite start
            [0.0, 0.0, 50.0, 2.0],    # dendrite end
        ], dtype=np.float32)
        morph_grp.create_dataset("points", data=points)
        # Structure: [start_offset, type, parent_id]
        # type 1 = soma, type 3 = basal dendrite
        structure = np.array([
            [0, 1, -1],  # soma: starts at point 0, type=soma, root
            [3, 3, 0],   # dendrite: starts at point 3, type=basal, parent=soma
        ], dtype=np.int32)
        morph_grp.create_dataset("structure", data=structure)
        # Required metadata attributes for MorphIO H5v1
        meta = morph_grp.create_group("metadata")
        meta.attrs["version"] = np.array([1, 3], dtype=np.uint32)
        meta.attrs["cell_family"] = np.int64(0)

        # Edges (spine table): 2 spines on section 1
        edges_grp = f.create_group(f"edges/{morphology_name}")
        n_spines = 2
        edges_grp.create_dataset("afferent_section_id", data=np.array([1, 1], dtype=np.int64))
        edges_grp.create_dataset("afferent_segment_id", data=np.array([0, 0], dtype=np.int64))
        edges_grp.create_dataset("afferent_segment_offset", data=np.array([0.5, 0.8], dtype=np.float64))
        edges_grp.create_dataset("afferent_section_pos", data=np.array([0.3, 0.7], dtype=np.float64))
        edges_grp.create_dataset("afferent_center_x", data=np.zeros(n_spines))
        edges_grp.create_dataset("afferent_center_y", data=np.zeros(n_spines))
        edges_grp.create_dataset("afferent_center_z", data=np.array([10.0, 30.0]))
        edges_grp.create_dataset("afferent_surface_x", data=np.zeros(n_spines))
        edges_grp.create_dataset("afferent_surface_y", data=np.zeros(n_spines))
        edges_grp.create_dataset("afferent_surface_z", data=np.array([10.0, 30.0]))
        edges_grp.create_dataset("spine_id", data=np.array([0, 1], dtype=np.uint32))
        edges_grp.create_dataset("spine_length", data=np.array([2.0, 3.0]))
        edges_grp.create_dataset("spine_morphology", data=np.array(
            [f"{morphology_name}", f"{morphology_name}"], dtype=object))
        edges_grp.create_dataset("spine_orientation_vector_x", data=np.zeros(n_spines))
        edges_grp.create_dataset("spine_orientation_vector_y", data=np.zeros(n_spines))
        edges_grp.create_dataset("spine_orientation_vector_z", data=np.ones(n_spines))
        edges_grp.create_dataset("spine_rotation_x", data=np.zeros(n_spines))
        edges_grp.create_dataset("spine_rotation_y", data=np.zeros(n_spines))
        edges_grp.create_dataset("spine_rotation_z", data=np.zeros(n_spines))
        edges_grp.create_dataset("spine_rotation_w", data=np.ones(n_spines))
        edges_grp.create_group("metadata")

        # Soma mesh (minimal)
        soma_grp = f.create_group(f"soma/meshes/{morphology_name}")
        soma_grp.create_dataset("vertices", data=np.zeros((4, 3), dtype=np.float32))
        soma_grp.create_dataset("triangles", data=np.array([[0, 1, 2]], dtype=np.int64))

        # Spines group (minimal — just skeletons to pass validation)
        skel_grp = f.create_group(f"spines/skeletons/{morphology_name}")
        skel_grp.create_dataset("points", data=np.zeros((4, 4), dtype=np.float32))
        skel_grp.create_dataset("structure", data=np.array([[0, -1, 3], [1, 0, 3]], dtype=np.int32))
        skel_grp.create_group("metadata")

        f.create_group("spines/meshes")


@pytest.fixture
def synthetic_morph_spines_h5(tmp_path):
    """Provide a synthetic morph-spines H5 file for testing."""
    filepath = tmp_path / "test_with_spines.h5"
    create_synthetic_morph_spines_h5(str(filepath))
    return str(filepath)
