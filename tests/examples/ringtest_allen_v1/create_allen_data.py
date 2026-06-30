#!/bin/env python
# /// script
# dependencies = ['h5py', 'libsonata', 'numpy']
# ///
# the above allows one to run `uv run create_data.py` without a virtualenv
import itertools as it
import sys
from pathlib import Path

import h5py
import numpy as np

# Add path for local imports
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import Edges, make_edges, make_nodes


def make_v1_nodes():
    wanted = {
        "node_type_id": -1,
        "model_template": "hoc:TestCell",
        "model_type": "biophysical",
        "mtype": ["MTYPE0", "MTYPE1", "MTYPE2"],
        "etype": ["ETYPE0", "ETYPE1", "ETYPE2"],
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
        "morphology": "cell_small",
    }
    make_nodes(filename="nodes_A.h5", name="RingA", count=3, wanted_attributes=wanted)

    wanted = {
        "node_type_id": -1,
        "model_template": ["hoc:IntFire1_exc_1", "hoc:IntFire1_inh_1", "hoc:IntFire1_exc_1"],
        "model_type": "point_process",
        "morphology": "None",
        "mtype": ["MTYPE0", "MTYPE1", "MTYPE2"],
        # "etype": ["ETYPE0", "ETYPE1", "ETYPE2"],
        "x": it.count(0),
        "y": it.count(1),
        "z": it.count(2),
    }
    make_nodes(filename="nodes_B.h5", name="RingB", count=3, wanted_attributes=wanted)


def make_v1_edges():
    edges = Edges("RingA", "RingA", "Exp2Syn_synapse", [(0, 1), (1, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(11.0),
        "delay": it.count(13.0),
        "tau1": it.count(14.0),
        "tau2": it.count(15.0),
        "erev": it.count(16.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
    }
    make_edges(filename="edges_A.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingB", "RingB", "point_process", [(0, 1), (1, 2), (2, 0)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": it.count(21.0),
        "delay": it.count(23.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
    }
    make_edges(filename="edges_B.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingA", "RingB", "point_process", [(0, 0), (1, 1), (2, 2)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": 100.0,
        "delay": 3.0,
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
    }
    make_edges(filename="edges_AB.h5", edges=edges, wanted_attributes=wanted_attributes)

    edges = Edges("RingB", "RingA", "Exp2Syn_synapse", [(0, 0), (1, 1), (2, 2)])
    wanted_attributes = {
        "edge_type_id": -1,
        "conductance": 100.0,
        "delay": it.count(13.0),
        "tau1": it.count(14.0),
        "tau2": it.count(15.0),
        "erev": it.count(16.0),
        "afferent_section_id": 1,
        "afferent_section_pos": 0.75,
    }
    make_edges(filename="edges_BA.h5", edges=edges, wanted_attributes=wanted_attributes)


def make_replay_input_file(population):
    node_ids = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    timestamps = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    output_file = f"replay_{population}.h5"
    with h5py.File(output_file, "w") as f:
        spikes_group = f.create_group("spikes")
        population_group = spikes_group.create_group(population)
        sorting_dtype = h5py.enum_dtype({"none": 0, "by_id": 1, "by_time": 2}, basetype="uint8")
        population_group.attrs["sorting"] = np.array(2, dtype=sorting_dtype)  # by_time
        population_group.create_dataset("node_ids", data=node_ids, dtype="uint64")
        timestamps_dataset = population_group.create_dataset(
            "timestamps", data=timestamps, dtype="float64"
        )
        timestamps_dataset.attrs["units"] = "ms"


make_v1_nodes()
make_v1_edges()
make_replay_input_file("RingA")
make_replay_input_file("RingB")
