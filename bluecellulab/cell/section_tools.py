# Copyright 2025 Open Brain Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NEURON section helpers for BlueCelluLab."""


def currents_vars(section) -> dict:
    """Return recordable currents (with units) at a given section.

    - Ionic currents from the ions block (e.g. ``ina``, ``ik``, ``ica``), in mA/cm².
    - Mechanism currents for variables ``i`` or ``ihcn``, reported as ``<var>_<mech>``
      (e.g. ``i_pas``, ``ihcn_Ih``), in mA/cm².

    Args:
        section: NEURON Section object.

    Returns:
        dict mapping variable names to {"units": str, "kind": str}.
    """
    psec = section.psection()
    out = {}

    ions = psec.get("ions", {}) or {}
    for ion, vars_dict in ions.items():
        if ion == "ttx":
            continue
        name = f"i{ion}"
        if name in vars_dict:
            out[name] = {"units": "mA/cm²", "kind": "ionic_current"}

    special_currents = {
        ("pas", "i"): "i_pas",
        ("Ih", "ihcn"): "ihcn_Ih",
        ("hd", "i"): "i_hd",
    }

    for (mech, var), out_name in special_currents.items():
        if var in (psec.get("density_mechs") or {}).get(mech, {}):
            out[out_name] = {"units": "mA/cm²", "kind": "nonspecific_current"}

    return dict(sorted(out.items()))


def mechs_vars(section, include_point_mechs: bool = False) -> dict:
    """Return mechanism-scoped variables at a given section.

    Args:
        section: NEURON Section object.
        include_point_mechs: Whether to include point processes.
    """
    psec = section.psection()
    dens = psec.get("density_mechs", {}) or {}
    points = psec.get("point_mechs", {}) or {}
    mech_map = {
        mech: sorted(vars_dict.keys())
        for mech, vars_dict in dens.items() if vars_dict
    }
    entry = {"mech": mech_map}
    if include_point_mechs:
        point_map = {
            pp: sorted(vars_dict.keys())
            for pp, vars_dict in points.items() if vars_dict
        }
        entry["point"] = point_map
    return entry


def section_to_variable_recording_str(section, segx: float, variable: str) -> str:
    """Build an evaluable NEURON pointer string for `add_recording`.

    Accepts:
      - top-level vars: "v", "ina", "ik", ...
      - mechanism-scoped vars: "kca.gkca", "na3.m", "na3.h", ...

    Returns examples:
      neuron.h.soma[0](0.5)._ref_v
      neuron.h.soma[0](0.5)._ref_ina
      neuron.h.soma[0](0.5).kca._ref_gkca
      neuron.h.dend[3](0.7).na3._ref_m
    """
    sec_name = section.name()
    if "." in variable:
        mech, var = variable.split(".", 1)
        return f"neuron.h.{sec_name}({segx}).{mech}._ref_{var}"
    else:
        return f"neuron.h.{sec_name}({segx})._ref_{variable}"
