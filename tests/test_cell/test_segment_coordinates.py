"""Tests for cell coordinate helpers."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import neuron
from bluecellulab.cell.core import Cell
from bluecellulab.circuit.node_id import CellId


def _patched_public_hoc_cell(sections_cache):
    """Return a mock public_hoc_cell that exposes .all."""
    def _mock(cell):
        m = MagicMock()
        m.all = list(sections_cache.values())
        return m
    return _mock


@pytest.fixture
def simple_cell():
    soma = neuron.h.Section(name="soma")
    soma.nseg = 1
    soma.L = 10.0
    soma.diam = 10.0
    neuron.h.pt3dadd(0, 0, 0, 10, sec=soma)
    neuron.h.pt3dadd(10, 0, 0, 10, sec=soma)
    dend = neuron.h.Section(name="dend[0]")
    dend.nseg = 3
    dend.L = 100.0
    dend.diam = 2.0
    dend.connect(soma(1), 0)
    neuron.h.pt3dadd(10, 0, 0, 2, sec=dend)
    neuron.h.pt3dadd(110, 0, 0, 2, sec=dend)
    cell = Cell.__new__(Cell)
    cell.cell_id = CellId("", 0)
    cell.cell = neuron.h.Cell()
    cell.soma = soma
    cell._sections_cache = {"soma": soma, "dend[0]": dend}
    cell._local_to_global_matrix = None
    return cell


def test_local_coords(simple_cell):
    with patch("bluecellulab.cell.core.public_hoc_cell", _patched_public_hoc_cell(simple_cell._sections_cache)):
        coords = simple_cell.compute_segment_local_coordinates()
    assert "soma" in coords
    assert "dend[0]" in coords
    assert coords["soma"].shape == (2, 3)
    assert coords["dend[0]"].shape == (4, 3)


def test_global_coords_identity():
    soma = neuron.h.Section(name="soma")
    soma.nseg = 1
    soma.L = 10.0
    neuron.h.pt3dadd(5, 5, 5, 10, sec=soma)
    neuron.h.pt3dadd(15, 5, 5, 10, sec=soma)
    cell = Cell.__new__(Cell)
    cell.cell_id = CellId("", 1)
    cell.cell = neuron.h.Cell()
    cell.soma = soma
    cell._sections_cache = {"soma": soma}
    cell._local_to_global_matrix = None
    with patch("bluecellulab.cell.core.public_hoc_cell", _patched_public_hoc_cell(cell._sections_cache)):
        local_c = cell.compute_segment_local_coordinates()
        global_c = cell.compute_segment_global_coordinates()
    np.testing.assert_array_equal(local_c["soma"], global_c["soma"])


def test_l2g_identity():
    cell = Cell.__new__(Cell)
    cell.cell_id = CellId("", 2)
    cell.cell = neuron.h.Cell()
    cell.soma = neuron.h.Section(name="soma")
    cell.set_local_to_global_matrix(np.array([10.0, 20.0, 30.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    pts = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(cell.local_to_global_coord_mapping(pts), [[11.0, 20.0, 30.0]])


def test_l2g_90z():
    cell = Cell.__new__(Cell)
    cell.cell_id = CellId("", 3)
    cell.cell = neuron.h.Cell()
    cell.soma = neuron.h.Section(name="soma")
    angle = np.pi / 4
    cell.set_local_to_global_matrix(np.zeros(3), np.array([np.cos(angle), 0.0, 0.0, np.sin(angle)]))
    pts = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(cell.local_to_global_coord_mapping(pts)[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_l2g_translation_only():
    cell = Cell.__new__(Cell)
    cell.cell_id = CellId("", 4)
    cell.cell = neuron.h.Cell()
    cell.soma = neuron.h.Section(name="soma")
    cell.set_local_to_global_matrix(np.array([5.0, -3.0, 2.0]), None)
    pts = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(cell.local_to_global_coord_mapping(pts), [[6.0, -1.0, 5.0]])


def test_get_seg_pos_3d():
    sec = neuron.h.Section(name="testsec")
    sec.nseg = 2
    sec.L = 100.0
    neuron.h.pt3dadd(0, 0, 0, 2, sec=sec)
    neuron.h.pt3dadd(100, 0, 0, 2, sec=sec)
    pts = np.array([[0, 0, 0], [50, 0, 0], [100, 0, 0]])
    np.testing.assert_allclose(Cell.get_segment_position(pts, np.zeros(3), sec, 0.0), [0, 0, 0])
    np.testing.assert_allclose(Cell.get_segment_position(pts, np.zeros(3), sec, 0.5), [50, 0, 0])
    np.testing.assert_allclose(Cell.get_segment_position(pts, np.zeros(3), sec, 1.0), [100, 0, 0])


def test_get_seg_pos_axon():
    axon = neuron.h.Section(name="axon[0]")
    axon.nseg = 1
    axon.L = 30.0
    soma_pos = np.array([10.0, 20.0, 30.0])
    pos = Cell.get_segment_position(np.array([]), soma_pos, axon, 0.5, func_loc2glob=None)
    np.testing.assert_allclose(pos, [10.0, 5.0, 30.0])


def test_get_seg_pos_axon_l2g():
    axon = neuron.h.Section(name="axon[0]")
    axon.nseg = 1
    axon.L = 30.0
    soma_pos = np.array([1.0, 0.0, 0.0])
    pos = Cell.get_segment_position(np.array([]), soma_pos, axon, 0.5, func_loc2glob=lambda p: p + 10.0)
    np.testing.assert_allclose(pos, [11.0, -5.0, 10.0])


def test_get_seg_pos_value_error():
    sec = neuron.h.Section(name="dend_sec")
    sec.nseg = 1
    sec.L = 10.0
    with pytest.raises(ValueError, match="has no 3d points defined"):
        Cell.get_segment_position(np.array([]), np.zeros(3), sec, 0.5)


def test_no_3d_warning(simple_cell, caplog):
    axon = neuron.h.Section(name="axon[0]")
    axon.nseg = 1
    axon.L = 30.0
    axon.connect(simple_cell._sections_cache["dend[0]"](1), 0)
    simple_cell._sections_cache["axon[0]"] = axon
    with patch("bluecellulab.cell.core.public_hoc_cell", _patched_public_hoc_cell(simple_cell._sections_cache)):
        with caplog.at_level("WARNING", logger="bluecellulab.cell.core"):
            coords = simple_cell.compute_segment_local_coordinates()
    assert coords["axon[0]"].size == 0
    assert "has no 3D points" in caplog.text
