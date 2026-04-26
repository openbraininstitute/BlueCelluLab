# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from bluecellulab.stimulus.extracellular import ElectrodeSource


def test_electrode_source_init():
    """Test ElectrodeSource initialization."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )
    assert es.duration == 10
    assert es.dt == 1.0
    assert len(es.fields) == 1


def test_apply_ramp_case1():
    """Test ramp application when ramp_up/down_time > dt and is multiple of dt."""
    import neuron

    dt = 0.5
    ramp_up_time = 2.0
    ramp_down_time = 1.5

    es = ElectrodeSource(0, 0, 100, [], ramp_up_time, ramp_down_time, dt)
    stim_vec = neuron.h.Vector(list(range(1, 11)))

    assert np.isclose(es.ramp_up_time, ramp_up_time)
    assert np.isclose(es.ramp_down_time, ramp_down_time)
    assert np.isclose(es.dt, dt)

    es.apply_ramp(stim_vec, es.dt)

    expected = [0, 2 / 3, 2, 4, 5, 6, 7, 8, 4.5, 0]
    np.testing.assert_allclose(stim_vec.as_numpy(), expected)


def test_apply_ramp_case2():
    """Test ramp when ramp_up/down_time > dt but not multiple of dt."""
    import neuron

    dt = 0.5
    ramp_up_time = 2.4
    ramp_down_time = 1.7

    es = ElectrodeSource(0, 0, 100, [], ramp_up_time, ramp_down_time, dt)
    stim_vec = neuron.h.Vector(list(range(1, 11)))

    es.apply_ramp(stim_vec, es.dt)

    expected = [0, 2 / 3, 2, 4, 5, 6, 7, 8, 4.5, 0]
    np.testing.assert_allclose(stim_vec.as_numpy(), expected)


def test_apply_ramp_case3():
    """Test ramp when ramp_up/down_time < dt (no ramp applied)."""
    import neuron

    dt = 0.5
    ramp_up_time = 0.3
    ramp_down_time = 0.4

    es = ElectrodeSource(0, 0, 100, [], ramp_up_time, ramp_down_time, dt)
    stim_vec = neuron.h.Vector(list(range(1, 11)))

    es.apply_ramp(stim_vec, es.dt)

    expected = list(range(1, 11))
    np.testing.assert_allclose(stim_vec.as_numpy(), expected)


def test_dc_field():
    """Test constant (DC) field when frequency=0."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    assert es.efields.shape[0] == 3

    for i in range(3):
        field_values = es.efields[i]
        assert np.all(field_values[1:-1] != 0)


def test_ac_field():
    """Test AC (cosine) field with non-zero frequency."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 100, "Ey": 0, "Ez": 0, "frequency": 10}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    assert es.efields.shape[0] == 3


def test_multi_field_summation():
    """Test that multiple fields sum correctly."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[
            {"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 100},
            {"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0},
        ],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    assert len(es.fields) == 2
    assert es.efields.shape[0] == 3


def test_compute_potentials():
    """Test potential calculation from displacement vector."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    displacement = np.array([1e-6, 2e-6, 3e-6])
    potentials = es.compute_potentials(displacement)

    assert len(potentials) == len(es.time_vec)


def test_iadd_combining_sources():
    """Test combining two ElectrodeSource objects."""
    es1 = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es2 = ElectrodeSource(
        base_amp=0,
        delay=5,
        duration=10,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    original_len = len(es1.time_vec)
    es1 += es2

    assert len(es1.time_vec) >= original_len


def test_iadd_different_dt_raises():
    """Test that combining sources with different dt raises assertion error."""
    es1 = ElectrodeSource(
        base_amp=0, delay=0, duration=10, fields=[],
        ramp_up_time=0, ramp_down_time=0, dt=1.0
    )

    es2 = ElectrodeSource(
        base_amp=0, delay=0, duration=10, fields=[],
        ramp_up_time=0, ramp_down_time=0, dt=0.5
    )

    with pytest.raises(AssertionError, match="multiple extracellular stimuli must have common dt"):
        es1 += es2


def test_iadd_with_delay():
    """Test combining sources with delay."""
    es1 = ElectrodeSource(
        base_amp=0,
        delay=2,
        duration=10,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es2 = ElectrodeSource(
        base_amp=0,
        delay=5,
        duration=10,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es1 += es2

    assert len(es1.time_vec) > 0
    assert es1.time_vec[0] == 0


def test_iadd_non_overlapping():
    """Test combining sources with non-overlapping time ranges."""
    es1 = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=5,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es2 = ElectrodeSource(
        base_amp=0,
        delay=10,
        duration=5,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es1 += es2

    assert len(es1.time_vec) > 0


def test_iadd_time_concatenation():
    """Test combining sources with time vector concatenation edge case."""
    es1 = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=5,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es2 = ElectrodeSource(
        base_amp=0,
        delay=5,
        duration=5,
        fields=[{"Ex": 100, "Ey": -50, "Ez": 50, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    es1 += es2

    assert len(es1.time_vec) > 0
    assert es1.time_vec[0] == 0


def test_cleanup():
    """Test that cleanup() clears references."""
    es = ElectrodeSource(
        base_amp=0,
        delay=0,
        duration=10,
        fields=[{"Ex": 50, "Ey": -25, "Ez": 75, "frequency": 0}],
        ramp_up_time=0,
        ramp_down_time=0,
        dt=1.0,
    )

    assert es.efields is not None
    assert es.segment_displacements is not None

    es.cleanup()

    assert es.efields is None
    assert es.segment_displacements is None


def test_interp_axon_positions():
    """Test axon position interpolation."""
    from bluecellulab.cell import Cell

    soma_position = np.array([0.0, 0.0, 0.0])

    # Test axon[0]
    pos = Cell.interp_axon_positions(0.0, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0])

    pos = Cell.interp_axon_positions(0.5, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, -15.0, 0.0])

    pos = Cell.interp_axon_positions(1.0, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, -30.0, 0.0])

    # Test axon[1]
    pos = Cell.interp_axon_positions(0.0, 1, soma_position)
    np.testing.assert_allclose(pos, [0.0, -30.0, 0.0])

    pos = Cell.interp_axon_positions(0.5, 1, soma_position)
    np.testing.assert_allclose(pos, [0.0, -45.0, 0.0])

    pos = Cell.interp_axon_positions(1.0, 1, soma_position)
    np.testing.assert_allclose(pos, [0.0, -60.0, 0.0])


def test_interp_axon_positions_error():
    """Test that more than 2 axon sections raises error."""
    from bluecellulab.cell import Cell

    soma_position = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="More than 2 axon sections exist"):
        Cell.interp_axon_positions(0.5, 2, soma_position)


def test_interp_myelin_positions():
    """Test myelin position interpolation."""
    from bluecellulab.cell import Cell

    soma_position = np.array([0.0, 0.0, 0.0])

    # Test myelin[0]
    pos = Cell.interp_myelin_positions(0.0, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, -60.0, 0.0])

    pos = Cell.interp_myelin_positions(0.5, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, -560.0, 0.0])

    pos = Cell.interp_myelin_positions(1.0, 0, soma_position)
    np.testing.assert_allclose(pos, [0.0, -1060.0, 0.0])


def test_interp_myelin_positions_error():
    """Test that more than 1 myelin section raises error."""
    from bluecellulab.cell import Cell

    soma_position = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="More than 1 myelin section exist"):
        Cell.interp_myelin_positions(0.5, 1, soma_position)
