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
"""Defines the expected data structures associated with the stimulus defined in
simulation configs.

Run-time validates the data via Pydantic.
"""
from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Optional
import warnings

from pydantic import field_validator, NonNegativeFloat, PositiveFloat
from pydantic.dataclasses import dataclass


# create an enum for StimulusMode with Current and Conductance values
class ClampMode(Enum):
    """Current clamp or conductance (dynamic) clamp."""
    CURRENT = "current_clamp"
    CONDUCTANCE = "conductance"


class Pattern(Enum):
    """Enum that defaults to SONATA values.

    Has blueconfig overload.
    """
    NOISE = "noise"
    HYPERPOLARIZING = "hyperpolarizing"
    PULSE = "pulse"
    LINEAR = "linear"
    RELATIVE_LINEAR = "relative_linear"
    SYNAPSE_REPLAY = "synapse_replay"
    SHOT_NOISE = "shot_noise"
    RELATIVE_SHOT_NOISE = "relative_shot_noise"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    RELATIVE_ORNSTEIN_UHLENBECK = "relative_ornstein_uhlenbeck"
    SINUSOIDAL = "sinusoidal"
    SPATIALLY_UNIFORM_E_FIELD = "spatially_uniform_e_field"

    @classmethod
    def from_blueconfig(cls, pattern: str) -> Pattern:
        if pattern == "Noise":
            return Pattern.NOISE
        elif pattern == "Hyperpolarizing":
            return Pattern.HYPERPOLARIZING
        elif pattern == "Pulse":
            return Pattern.PULSE
        elif pattern == "RelativeLinear":
            return Pattern.RELATIVE_LINEAR
        elif pattern == "SynapseReplay":
            return Pattern.SYNAPSE_REPLAY
        elif pattern == "ShotNoise":
            return Pattern.SHOT_NOISE
        elif pattern == "RelativeShotNoise":
            return Pattern.RELATIVE_SHOT_NOISE
        elif pattern == "OrnsteinUhlenbeck":
            return Pattern.ORNSTEIN_UHLENBECK
        elif pattern == "RelativeOrnsteinUhlenbeck":
            return Pattern.RELATIVE_ORNSTEIN_UHLENBECK
        else:
            raise ValueError(f"Unknown pattern {pattern}")

    @classmethod
    def from_sonata(cls, pattern: str) -> Pattern:
        if pattern == "noise":
            return Pattern.NOISE
        elif pattern == "hyperpolarizing":
            return Pattern.HYPERPOLARIZING
        elif pattern == "pulse":
            return Pattern.PULSE
        elif pattern == "linear":
            return Pattern.LINEAR
        elif pattern == "relative_linear":
            return Pattern.RELATIVE_LINEAR
        elif pattern == "synapse_replay":
            return Pattern.SYNAPSE_REPLAY
        elif pattern == "shot_noise":
            return Pattern.SHOT_NOISE
        elif pattern == "relative_shot_noise":
            return Pattern.RELATIVE_SHOT_NOISE
        elif pattern == "ornstein_uhlenbeck":
            return Pattern.ORNSTEIN_UHLENBECK
        elif pattern == "relative_ornstein_uhlenbeck":
            return Pattern.RELATIVE_ORNSTEIN_UHLENBECK
        elif pattern == "sinusoidal":
            return Pattern.SINUSOIDAL
        elif pattern == "spatially_uniform_e_field":
            return Pattern.SPATIALLY_UNIFORM_E_FIELD
        else:
            raise ValueError(f"Unknown pattern {pattern}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class Stimulus:
    target: str
    delay: NonNegativeFloat
    duration: NonNegativeFloat

    node_set: Optional[str] = field(default=None, kw_only=True)
    compartment_set: Optional[str] = field(default=None, kw_only=True)

    @classmethod
    def from_blueconfig(cls, stimulus_entry: dict) -> Optional[Stimulus]:
        pattern = Pattern.from_blueconfig(stimulus_entry["Pattern"])
        mode_str = stimulus_entry.get("Mode", "Current").lower()
        if mode_str == "current":
            mode = ClampMode.CURRENT
        elif mode_str == "conductance":
            mode = ClampMode.CONDUCTANCE
        else:
            raise ValueError(f"Unknown clamp mode {mode_str}")
        if pattern == Pattern.NOISE:
            return Noise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                mean_percent=stimulus_entry["MeanPercent"],
                variance=stimulus_entry["Variance"],
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.HYPERPOLARIZING:
            return Hyperpolarizing(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.PULSE:
            return Pulse(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                amp_start=stimulus_entry["AmpStart"],
                width=stimulus_entry["Width"],
                frequency=stimulus_entry["Frequency"],
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.RELATIVE_LINEAR:
            return RelativeLinear(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                percent_start=stimulus_entry["PercentStart"],
                percent_end=stimulus_entry["PercentEnd"],
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.SYNAPSE_REPLAY:
            warnings.warn("Ignoring syanpse replay stimulus as it is not supported")
            return None
        elif pattern == Pattern.SHOT_NOISE:
            return ShotNoise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                rise_time=stimulus_entry["RiseTime"],
                decay_time=stimulus_entry["DecayTime"],
                rate=stimulus_entry["Rate"],
                amp_mean=stimulus_entry["AmpMean"],
                amp_var=stimulus_entry["AmpVar"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0),
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.RELATIVE_SHOT_NOISE:
            return RelativeShotNoise(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                rise_time=stimulus_entry["RiseTime"],
                decay_time=stimulus_entry["DecayTime"],
                mean_percent=stimulus_entry["MeanPercent"],
                sd_percent=stimulus_entry["SDPercent"],
                relative_skew=stimulus_entry.get("RelativeSkew", 0.5),
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0),
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.ORNSTEIN_UHLENBECK:
            return OrnsteinUhlenbeck(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                tau=stimulus_entry["Tau"],
                sigma=stimulus_entry["Sigma"],
                mean=stimulus_entry["Mean"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0),
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        elif pattern == Pattern.RELATIVE_ORNSTEIN_UHLENBECK:
            return RelativeOrnsteinUhlenbeck(
                target=stimulus_entry["Target"],
                delay=stimulus_entry["Delay"],
                duration=stimulus_entry["Duration"],
                dt=stimulus_entry.get("Dt", 0.25),
                tau=stimulus_entry["Tau"],
                mean_percent=stimulus_entry["MeanPercent"],
                sd_percent=stimulus_entry["SDPercent"],
                seed=stimulus_entry.get("Seed", None),
                mode=mode,
                reversal=stimulus_entry.get("Reversal", 0.0),
                node_set=stimulus_entry["Target"],
                compartment_set=None,
            )
        else:
            raise ValueError(f"Unknown pattern {pattern}")

    @classmethod
    def from_sonata(cls, stimulus_entry: dict, config_dir: Optional[str] = None) -> Optional[Stimulus]:
        pattern = Pattern.from_sonata(stimulus_entry["module"])
        node_set = stimulus_entry.get("node_set")
        compartment_set = stimulus_entry.get("compartment_set")
        if node_set is not None and compartment_set is not None:
            raise ValueError("Stimulus entry must not contain both 'node_set' and 'compartment_set'.")
        target_name: str | None = compartment_set if compartment_set is not None else node_set
        if target_name is None:
            raise ValueError("Stimulus entry must contain either 'node_set' or 'compartment_set'.")

        if pattern == Pattern.NOISE:
            has_mean = "mean" in stimulus_entry
            has_mean_percent = "mean_percent" in stimulus_entry
            if has_mean == has_mean_percent:
                raise ValueError("Noise input must contain exactly one of 'mean' or 'mean_percent'.")

            return Noise(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                mean=stimulus_entry.get("mean"),
                mean_percent=stimulus_entry.get("mean_percent"),
                variance=stimulus_entry["variance"],
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.HYPERPOLARIZING:
            return Hyperpolarizing(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.PULSE:
            return Pulse(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                amp_start=stimulus_entry["amp_start"],
                width=stimulus_entry["width"],
                frequency=stimulus_entry["frequency"],
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.LINEAR:
            return Linear(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                amp_start=stimulus_entry["amp_start"],
                amp_end=stimulus_entry.get("amp_end", stimulus_entry["amp_start"]),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.RELATIVE_LINEAR:
            return RelativeLinear(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                percent_start=stimulus_entry["percent_start"],
                percent_end=stimulus_entry.get("percent_end", stimulus_entry["percent_start"]),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.SYNAPSE_REPLAY:
            return SynapseReplay(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                spike_file=stimulus_entry["spike_file"],
                config_dir=config_dir,
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.SHOT_NOISE:
            return ShotNoise(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                rise_time=stimulus_entry["rise_time"],
                decay_time=stimulus_entry["decay_time"],
                rate=stimulus_entry["rate"],
                amp_mean=stimulus_entry["amp_mean"],
                amp_var=stimulus_entry["amp_var"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.RELATIVE_SHOT_NOISE:
            return RelativeShotNoise(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                rise_time=stimulus_entry["rise_time"],
                decay_time=stimulus_entry["decay_time"],
                mean_percent=stimulus_entry["mean_percent"],
                sd_percent=stimulus_entry["sd_percent"],
                relative_skew=stimulus_entry.get("relative_skew", 0.5),
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.ORNSTEIN_UHLENBECK:
            return OrnsteinUhlenbeck(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                tau=stimulus_entry["tau"],
                sigma=stimulus_entry["sigma"],
                mean=stimulus_entry["mean"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.RELATIVE_ORNSTEIN_UHLENBECK:
            return RelativeOrnsteinUhlenbeck(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                dt=stimulus_entry.get("dt", 0.25),
                tau=stimulus_entry["tau"],
                mean_percent=stimulus_entry["mean_percent"],
                sd_percent=stimulus_entry["sd_percent"],
                seed=stimulus_entry.get("random_seed", None),
                mode=ClampMode(stimulus_entry.get("input_type", "current_clamp").lower()),
                reversal=stimulus_entry.get("reversal", 0.0),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.SINUSOIDAL:
            return Sinusoidal(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                amp_start=stimulus_entry["amp_start"],
                frequency=stimulus_entry["frequency"],
                node_set=node_set,
                compartment_set=compartment_set,
            )
        elif pattern == Pattern.SPATIALLY_UNIFORM_E_FIELD:
            return SpatiallyUniformEField(
                target=target_name,
                delay=stimulus_entry["delay"],
                duration=stimulus_entry["duration"],
                fields=stimulus_entry["fields"],
                ramp_up_time=stimulus_entry.get("ramp_up_time", 0.0),
                ramp_down_time=stimulus_entry.get("ramp_down_time", 0.0),
                node_set=node_set,
                compartment_set=compartment_set,
            )
        else:
            raise ValueError(f"Unknown pattern {pattern}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class Noise(Stimulus):
    variance: float
    mean: Optional[float] = None          # nA
    mean_percent: Optional[float] = None  # % of threshold

    def __post_init__(self):
        # exactly one of mean / mean_percent must be provided
        if (self.mean is None) == (self.mean_percent is None):
            raise ValueError("Noise stimulus must define exactly one of 'mean' or 'mean_percent'.")
        if self.variance < 0:
            raise ValueError("'variance' must be >= 0.")


@dataclass(frozen=True, config=dict(extra="forbid"))
class Hyperpolarizing(Stimulus):
    ...


@dataclass(frozen=True, config=dict(extra="forbid"))
class Pulse(Stimulus):
    amp_start: float
    width: float
    frequency: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class Linear(Stimulus):
    amp_start: float
    amp_end: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeLinear(Stimulus):
    percent_start: float
    percent_end: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class SynapseReplay(Stimulus):
    spike_file: str
    config_dir: Optional[str] = None


@dataclass(frozen=True, config=dict(extra="forbid"))
class ShotNoise(Stimulus):
    rise_time: float
    decay_time: float
    rate: float
    amp_mean: float
    amp_var: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("decay_time")
    @classmethod
    def decay_time_gt_rise_time(cls, v, values):
        if v <= values.data["rise_time"]:
            raise ValueError("decay_time must be greater than rise_time")
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeShotNoise(Stimulus):
    rise_time: float
    decay_time: float
    mean_percent: float
    sd_percent: float
    relative_skew: float = 0.5
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("decay_time")
    @classmethod
    def decay_time_gt_rise_time(cls, v, values):
        if v <= values.data["rise_time"]:
            raise ValueError("decay_time must be greater than rise_time")
        return v

    @field_validator("relative_skew")
    @classmethod
    def relative_skew_in_range(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("relative skewness must be in [0,1]")
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class OrnsteinUhlenbeck(Stimulus):
    tau: float
    sigma: PositiveFloat
    mean: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0

    @field_validator("mean")
    @classmethod
    def mean_in_range(cls, v, values):
        if v < 0 and abs(v) > 2 * values.data["sigma"]:
            warnings.warn(
                "mean is outside of range [0, 2*sigma],",
                " ornstein uhlenbeck signal is mostly zero.",
            )
        return v


@dataclass(frozen=True, config=dict(extra="forbid"))
class RelativeOrnsteinUhlenbeck(Stimulus):
    tau: float
    mean_percent: float
    sd_percent: float
    dt: float = 0.25
    seed: Optional[int] = None
    mode: ClampMode = ClampMode.CURRENT
    reversal: float = 0.0


@dataclass(frozen=True, config=dict(extra="forbid"))
class Sinusoidal(Stimulus):
    amp_start: float
    frequency: float


@dataclass(frozen=True, config=dict(extra="forbid"))
class SpatiallyUniformEField(Stimulus):
    fields: list[dict[str, float]]
    ramp_up_time: NonNegativeFloat = 0.0
    ramp_down_time: NonNegativeFloat = 0.0

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v):
        if not v:
            raise ValueError("fields list cannot be empty")
        
        for i, field_dict in enumerate(v):
            if "Ex" not in field_dict or "Ey" not in field_dict or "Ez" not in field_dict:
                raise ValueError(
                    f"Field {i} must contain Ex, Ey, and Ez components"
                )
            
            frequency = field_dict.get("frequency", 0.0)
            if frequency < 0:
                raise ValueError(f"Field {i} frequency must be non-negative")
            
            phase = field_dict.get("phase", 0.0)
            if not isinstance(phase, (int, float)):
                raise ValueError(f"Field {i} phase must be a number")
        
        return v
