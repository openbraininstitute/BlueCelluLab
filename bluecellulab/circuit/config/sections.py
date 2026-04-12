# Copyright 2023-2024 Blue Brain Project / EPFL
# Copyright 2025-2026 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes to represent config sections."""

from __future__ import annotations
from typing import Any, Literal, Optional

from pydantic import field_validator, Field
from pydantic.dataclasses import dataclass

import neuron

import bluecellulab

# libsonata reorganized it's module layout; maintain compatibility with both:
# https://github.com/BlueBrain/libsonata/pull/345
try:
    from libsonata._libsonata import Conditions as LibSonataConditions
except ImportError:
    from libsonata._libsonata import SimulationConfig

    LibSonataConditions = SimulationConfig.Conditions


def string_to_bool(value: str) -> bool:
    """Convert a string to a boolean."""
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean value {value}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class ConditionEntry:
    """For mechanism specific conditions."""

    minis_single_vesicle: Optional[int] = Field(None, ge=0, le=1)
    init_depleted: Optional[int] = Field(None, ge=0, le=1)


@dataclass(frozen=True, config=dict(extra="forbid"))
class MechanismConditions:
    """For mechanism specific conditions."""

    ampanmda: Optional[ConditionEntry] = None
    gabaab: Optional[ConditionEntry] = None
    glusynapse: Optional[ConditionEntry] = None


@dataclass(frozen=True, config=dict(extra="forbid"))
class Conditions:
    mech_conditions: Optional[MechanismConditions] = None
    mechanisms: Optional[dict[str, dict[str, Any]]] = None
    celsius: Optional[float] = None
    v_init: Optional[float] = None
    extracellular_calcium: Optional[float] = None
    randomize_gaba_rise_time: Optional[bool] = None

    @classmethod
    def from_blueconfig(cls, condition_entries: dict) -> Conditions:
        """Create a Conditions instance from a blueconfig section."""
        randomize_gaba_risetime = condition_entries.get("randomize_Gaba_risetime")
        if randomize_gaba_risetime is not None:
            randomize_gaba_risetime = string_to_bool(randomize_gaba_risetime)

        msv_val = condition_entries.get("SYNAPSES__minis_single_vesicle", None)
        init_dep_val = condition_entries.get("SYNAPSES__init_depleted", None)
        mech_conditions = MechanismConditions(
            ampanmda=ConditionEntry(msv_val, init_dep_val),
            gabaab=ConditionEntry(msv_val, init_dep_val),
            glusynapse=ConditionEntry(msv_val, init_dep_val),
        )
        return cls(
            mech_conditions=mech_conditions,
            mechanisms=None,
            extracellular_calcium=condition_entries.get("cao_CR_GluSynapse", None),
            randomize_gaba_rise_time=randomize_gaba_risetime,
        )

    @classmethod
    def from_sonata(cls, condition_entries: LibSonataConditions) -> Conditions:
        """Create a Conditions instance from a SONATA section."""
        # sonata stores it as bool, hoc needs it as int
        msv_ampanmda = msv_gabaab = msv_glusynapse = None
        init_dep_ampanmda = init_dep_gabaab = init_dep_glusynapse = None
        mech_dict = condition_entries.mechanisms
        if mech_dict is not None:
            ampanmda = mech_dict.get("ProbAMPANMDA_EMS", None)
            if ampanmda is not None:
                msv_ampanmda = ampanmda.get("minis_single_vesicle", None)
                init_dep_ampanmda = ampanmda.get("init_depleted", None)
            gabaab = mech_dict.get("ProbGABAAB_EMS", None)
            if gabaab is not None:
                msv_gabaab = gabaab.get("minis_single_vesicle", None)
                init_dep_gabaab = gabaab.get("init_depleted", None)
            glusynapse = mech_dict.get("GluSynapse", None)
            if glusynapse is not None:
                msv_glusynapse = glusynapse.get("minis_single_vesicle", None)
                init_dep_glusynapse = glusynapse.get("init_depleted", None)

        mech_conditions = MechanismConditions(
            ampanmda=ConditionEntry(msv_ampanmda, init_dep_ampanmda),
            gabaab=ConditionEntry(msv_gabaab, init_dep_gabaab),
            glusynapse=ConditionEntry(msv_glusynapse, init_dep_glusynapse),
        )

        # Store the full generic mechanisms dict from libsonata
        generic_mechanisms = None
        if mech_dict is not None:
            generic_mechanisms = dict(mech_dict)

        return cls(
            mech_conditions=mech_conditions,
            mechanisms=generic_mechanisms,
            celsius=condition_entries.celsius,
            v_init=condition_entries.v_init,
            extracellular_calcium=condition_entries.extracellular_calcium,
            randomize_gaba_rise_time=condition_entries.randomize_gaba_rise_time,
        )

    @classmethod
    def init_empty(cls) -> Conditions:
        """Create an empty conditions object to be used when no condition is
        specified."""
        return cls(
            mech_conditions=None,
            mechanisms=None,
            celsius=None,
            v_init=None,
            extracellular_calcium=None,
            randomize_gaba_rise_time=None,
        )


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationBase:
    """Base class for all modification types."""

    name: str


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationNodeSet(ModificationBase):
    """Modification that targets a node_set."""

    node_set: str


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationTTX(ModificationNodeSet):
    """TTX modification — blocks Na channels on all sections of target
    cells."""

    type: Literal["ttx"] = "ttx"


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationConfigureAllSections(ModificationNodeSet):
    """Applies section_configure to all sections of target cells."""

    section_configure: str
    type: Literal["configure_all_sections"] = "configure_all_sections"


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationSectionList(ModificationNodeSet):
    """Applies section_configure to a named section list of target cells."""

    section_configure: str
    type: Literal["section_list"] = "section_list"


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationSection(ModificationNodeSet):
    """Applies section_configure to specific named sections of target cells."""

    section_configure: str
    type: Literal["section"] = "section"


@dataclass(frozen=True, config=dict(extra="forbid"))
class ModificationCompartmentSet(ModificationBase):
    """Applies section_configure to segments in a compartment set."""

    compartment_set: str
    section_configure: str
    type: Literal["compartment_set"] = "compartment_set"


def modification_from_libsonata(mod) -> ModificationBase:
    """Convert a libsonata modification object to a BlueCelluLab dataclass."""
    type_name = mod.type.name.lower()  # e.g. "ttx", "configure_all_sections", etc.
    if type_name == "ttx":
        return ModificationTTX(name=mod.name, node_set=mod.node_set)
    elif type_name == "configure_all_sections":
        return ModificationConfigureAllSections(
            name=mod.name,
            node_set=mod.node_set,
            section_configure=mod.section_configure,
        )
    elif type_name == "section_list":
        return ModificationSectionList(
            name=mod.name,
            node_set=mod.node_set,
            section_configure=mod.section_configure,
        )
    elif type_name == "section":
        return ModificationSection(
            name=mod.name,
            node_set=mod.node_set,
            section_configure=mod.section_configure,
        )
    elif type_name == "compartment_set":
        return ModificationCompartmentSet(
            name=mod.name,
            compartment_set=mod.compartment_set,
            section_configure=mod.section_configure,
        )
    else:
        raise ValueError(f"Unknown modification type: {type_name}")


@dataclass(frozen=True, config=dict(extra="forbid"))
class ConnectionOverrides:
    source: str
    target: str
    delay: Optional[float] = None
    weight: Optional[float] = None
    synapse_delay_override: Optional[float] = None
    spont_minis: Optional[float] = None
    synapse_configure: Optional[str] = None
    mod_override: Optional[Literal["GluSynapse"]] = None

    @field_validator("mod_override")
    @classmethod
    def validate_mod_override(cls, value):
        """Make sure the mod file to override is present."""
        if isinstance(value, str) and not hasattr(neuron.h, value):
            raise bluecellulab.ConfigError(f"Mod file for {value} is not found.")
        return value

    @classmethod
    def from_blueconfig(cls, conn_entry: dict) -> ConnectionOverrides:
        """Create a ConnectionOverrides instance from a blueconfig section."""
        return cls(
            source=conn_entry["Source"],
            target=conn_entry["Destination"],
            delay=conn_entry.get("Delay", None),
            weight=conn_entry.get("Weight", None),
            spont_minis=conn_entry.get("SpontMinis", None),
            synapse_configure=conn_entry.get("SynapseConfigure", None),
            mod_override=conn_entry.get("ModOverride", None),
        )

    @classmethod
    def from_sonata(cls, conn_entry: dict) -> ConnectionOverrides:
        """Create a ConnectionOverrides instance from a SONATA section."""
        return cls(
            source=conn_entry["source"],
            target=conn_entry["target"],
            delay=conn_entry.get("delay", None),
            weight=conn_entry.get("weight", None),
            synapse_delay_override=conn_entry.get("synapse_delay_override", None),
            spont_minis=conn_entry.get("spont_minis", None),
            synapse_configure=conn_entry.get("synapse_configure", None),
            mod_override=conn_entry.get("mod_override", None),
        )
