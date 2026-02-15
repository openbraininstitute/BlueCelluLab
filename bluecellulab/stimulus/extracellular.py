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
"""Extracellular stimulus signal synthesis and application."""

from __future__ import annotations

import numpy as np
import neuron


class ElectrodeSource:
    """Constructs an extracellular potential field as the sum of multiple user-defined e-fields.
    
    Applies the resulting signal to segment's e_extracellular reference.
    Adapted from Neurodamus stimuli.py for BlueCelluLab.
    
    Args:
        base_amp: baseline amplitude when signal is inactive
        delay: start time delay in ms
        duration: duration of the signal, not including ramp up and ramp down
        fields: list of user-defined electric field components (dicts with Ex, Ey, Ez, frequency, phase)
        ramp_up_time: duration during which the signal amplitude ramps up linearly from 0, in ms
        ramp_down_time: duration during which the signal amplitude ramps down linearly to 0, in ms
        dt: time step in ms
    """

    def __init__(self, base_amp, delay, duration, fields, ramp_up_time, ramp_down_time, dt):
        self.time_vec = neuron.h.Vector()
        self._cur_t = 0
        self._base_amp = base_amp
        self._delay = delay
        self.fields = fields
        self.duration = duration
        self.dt = dt
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        
        if delay > 0:
            self.time_vec.append(self._cur_t)
            self._cur_t = delay
        
        self.efields = self.add_cosines()
        self.segment_displacements = {}
        self.segment_potentials = []

    def delay_time(self, duration):
        """Increments the ref time so that the next created signal is delayed."""
        self._cur_t += duration
        return self

    def add_cosines(self):
        """Add multiple cosinusoidal signals.
        
        Returns:
            numpy array of shape (3, n_timepoints) for Ex, Ey, Ez field components
        """
        total_duration = self.duration + self.ramp_up_time + self.ramp_down_time
        tvec = neuron.h.Vector()
        tvec.indgen(self._cur_t, self._cur_t + total_duration, self.dt)
        self.time_vec.append(tvec)
        self.delay_time(total_duration)
        
        self.time_vec.append(self._cur_t + self.dt)
        
        res_x = neuron.h.Vector(len(self.time_vec))
        res_y = neuron.h.Vector(len(self.time_vec))
        res_z = neuron.h.Vector(len(self.time_vec))
        
        for field in self.fields:
            vec = neuron.h.Vector(len(tvec))
            freq = field.get("frequency", 0)
            phase = field.get("phase", 0)
            Ex = field["Ex"]
            Ey = field["Ey"]
            Ez = field["Ez"]
            
            vec.sin(freq, phase + np.pi / 2, self.dt)
            self.apply_ramp(vec, self.dt)
            
            if self._delay > 0:
                vec.insrt(0, self._base_amp)
            vec.append(self._base_amp)
            
            res_x.add(vec.c().mul(Ex))
            res_y.add(vec.c().mul(Ey))
            res_z.add(vec.c().mul(Ez))

        return np.array([res_x, res_y, res_z])

    def compute_potentials(self, displacement_vec):
        """Compute potential at a segment given displacement from reference point.
        
        Args:
            displacement_vec: 3D displacement vector in meters [dx, dy, dz]
            
        Returns:
            numpy array of potentials in mV over time
        """
        return np.dot(displacement_vec, self.efields) * 1e3

    def apply_ramp(self, signal_vec, step):
        """Apply linear ramp up and down to signal.
        
        Args:
            signal_vec: hoc Vector to apply ramp to
            step: time step in ms
        """
        ramp_up_number = int(self.ramp_up_time / step)
        ramp_down_number = int(self.ramp_down_time / step)

        if ramp_up_number > 0:
            ramp_up = np.linspace(0, 1, ramp_up_number)
            for i in range(ramp_up_number):
                signal_vec[i] *= ramp_up[i]
        if ramp_down_number > 0:
            ramp_down = np.linspace(1, 0, ramp_down_number)
            vec_len = len(signal_vec)
            for i in range(ramp_down_number):
                signal_vec[vec_len - ramp_down_number + i] *= ramp_down[i]

    def apply_segment_potentials(self):
        """Apply potentials to segment.extracellular._ref_e for all segments."""
        for segment, displacement in self.segment_displacements.items():
            section = segment.sec
            e_ext_vec = neuron.h.Vector(self.compute_potentials(displacement))
            
            if not section.has_membrane("extracellular"):
                section.insert("extracellular")
            
            e_ext_vec.play(segment.extracellular._ref_e, self.time_vec, 0)
            self.segment_potentials.append(e_ext_vec)

        self.cleanup()

    def cleanup(self):
        """Clear unused variables to free memory."""
        self.efields = None
        self.segment_displacements = None

    def __iadd__(self, other):
        """Combine with another ElectrodeSource object.
        
        Merges time vectors and sums overlapping e-fields.
        
        Args:
            other: another ElectrodeSource instance
            
        Returns:
            self with combined time vector and e-fields
        """
        assert np.isclose(self.dt, other.dt), "multiple extracellular stimuli must have common dt"
        
        combined_time_vec, self.efields = self._combine_time_efields(
            self.time_vec.as_numpy(),
            self.efields,
            other.time_vec.as_numpy(),
            other.efields,
            self._delay > 0,
            other._delay > 0,
            self.dt,
        )
        self.time_vec = neuron.h.Vector(combined_time_vec)
        return self

    @staticmethod
    def _combine_time_efields(t1_vec, efields1, t2_vec, efields2, is_delay1, is_delay2, dt):
        """Combine time and efields vectors from 2 ElectrodeSource objects.
        
        Args:
            t1_vec, t2_vec: numpy arrays of time points
            efields1, efields2: arrays of shape (3, n_timepoints) for Ex, Ey, Ez
            is_delay1, is_delay2: whether stimuli have delays
            dt: time step
            
        Returns:
            tuple of (combined_time_vec, combined_efields)
        """
        if is_delay1:
            t1_vec = t1_vec[1:]
            efields1 = efields1[:, 1:]
        if is_delay2:
            t2_vec = t2_vec[1:]
            efields2 = efields2[:, 1:]

        t1_ticks = np.round(t1_vec / dt).astype(np.int64)
        t2_ticks = np.round(t2_vec / dt).astype(np.int64)

        if not (t1_ticks[-1] < t2_ticks[0] or t2_ticks[-1] < t1_ticks[0]):
            combined_time_ticks = np.union1d(t1_ticks, t2_ticks)
            
            idx1_left = np.searchsorted(t1_ticks, combined_time_ticks, side="right") - 1
            idx1_right = np.searchsorted(t1_ticks, combined_time_ticks, side="left")
            mask1 = idx1_left == idx1_right
            
            idx2_left = np.searchsorted(t2_ticks, combined_time_ticks, side="right") - 1
            idx2_right = np.searchsorted(t2_ticks, combined_time_ticks, side="left")
            mask2 = idx2_left == idx2_right

            combined_efields = np.zeros((len(efields1), len(combined_time_ticks)), dtype=float)
            combined_efields[:, mask1] += efields1[:, idx1_left[mask1]]
            combined_efields[:, mask2] += efields2[:, idx2_left[mask2]]

        elif t1_ticks[-1] < t2_ticks[0]:
            combined_time_ticks = np.concatenate((t1_ticks, t2_ticks))
            combined_efields = np.concatenate((efields1, efields2), axis=1)
        else:
            combined_time_ticks = np.concatenate((t2_ticks, t1_ticks))
            combined_efields = np.concatenate((efields2, efields1), axis=1)

        combined_time_vec = combined_time_ticks.astype(float) * dt

        if combined_time_vec[0] > 0:
            combined_time_vec = np.concatenate([[0.0], combined_time_vec])
            combined_efields = np.concatenate(
                [np.zeros((combined_efields.shape[0], 1)), combined_efields], axis=1
            )

        assert combined_efields.shape[1] == len(combined_time_vec), (
            "Time and efield length mismatch"
        )

        return combined_time_vec, combined_efields
