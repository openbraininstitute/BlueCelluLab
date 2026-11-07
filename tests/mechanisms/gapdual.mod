TITLE Dual-mode gap-junction-like point process for tests

COMMENT
Dual-mode peer-voltage input (neurodamus Gap convention):
- POINTER vpre : used in serial runs (setpointer from the source section)
- RANGE vgap + use_vgap : used under MPI (pc.target_var writes vgap each
  step; NEURON cannot transfer into a POINTER)
ENDCOMMENT

NEURON {
    POINT_PROCESS gapdual
    POINTER vpre
    RANGE vgap
    RANGE use_vgap
    RANGE g
    RANGE i
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    g = 1
    use_vgap = 0
}

ASSIGNED {
    v (millivolt)
    vpre (millivolt)
    vgap (millivolt)
    i (nanoamp)
}

INITIAL {
    vgap = -70
    vpre = -70
}

BREAKPOINT {
    if (use_vgap) {
        i = (v - vgap) * g * 1e-3
    } else {
        i = (v - vpre) * g * 1e-3
    }
}
