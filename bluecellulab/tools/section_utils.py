"""NEURON section/segment/mechanism/ion helpers for BlueCelluLab."""

def currents_vars(section) -> dict:
    """Return ionic and nonspecific currents (with units) at a given section/segment."""
    psec = section.psection()
    out = {}
    ions = psec.get("ions", {}) or {}
    for ion, vars_dict in ions.items():
        if ion == "ttx":
            continue
        if f"i{ion}" in vars_dict:
            out[f"i{ion}"] = {"units": "mA/cm²", "kind": "ionic_current"}
    for mech_name, vars_dict in (psec.get("density_mechs") or {}).items():
        if "i" in vars_dict:
            out[f"{mech_name}.i"] = {"units": "mA/cm²", "kind": "nonspecific_current"}
    for pp_name, vars_dict in (psec.get("point_mechs") or {}).items():
        if "i" in vars_dict:
            out[f"{pp_name}.i"] = {"units": "nA", "kind": "point_process_current"}
    return dict(sorted(out.items()))

def mechs_vars(section, include_point_mechs: bool = False) -> dict:
    """Return mechanism-scoped variables at a given section/segment."""
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
