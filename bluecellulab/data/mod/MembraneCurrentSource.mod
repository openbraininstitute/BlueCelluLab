COMMENT
This mechanism is identical to h.IClamp, but is a membrane current instead of an electrode current
Since this is not an electrode current, NEGATIVE values of i depolarize the cell
For backwards compatibility with IClamp, the membrane current is -1 times the amplitude
ENDCOMMENT

NEURON {
	POINT_PROCESS MembraneCurrentSource
	RANGE del, dur, amp, i
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
}

PARAMETER {
	del (ms)
	dur (ms)	<0,1e9>
	amp (nA)
}
ASSIGNED { i (nA) }

INITIAL {
	i = 0
}

BREAKPOINT {
	at_time(del)
	at_time(del+dur)

	if (t < del + dur && t >= del) {
		i = -1 * amp
	}else{
		i = 0
	}
}
