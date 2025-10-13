# Copyright 2025 Open Brain Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import h5py
from unittest.mock import patch

from bluecellulab.reports.writers.compartment import CompartmentReportWriter


def make_trace(length, value):
    """Create a trace filled with a fixed value."""
    return (np.ones(length) * value).astype(np.float32)


def test_parallel_compartment_set_merge_order_and_alignment(tmp_path):
    """Test that compartment set reports from multiple ranks are merged correctly,
    preserving node ID order and data alignment.
    """
    tlen = 10
    time = np.linspace(0, 1, tlen).tolist()

    rank0 = {
        "NodeA_0": {"time": time, "voltage": make_trace(tlen, 10.0)},
        "NodeA_2": {"time": time, "voltage": make_trace(tlen, 30.0)},
    }
    rank1 = {
        "NodeA_1": {"time": time, "voltage": make_trace(tlen, 20.0)},
    }

    # Merged dictionary simulating pc.py_gather across ranks (unordered)
    merged = {**rank0, **rank1}

    out_file = tmp_path / "comp_set.h5"

    report_cfg = {
        "name": "comp_set_report",
        "type": "compartment_set",
        "compartment_set": "NodeA",
        "variable_name": "v",
        "start_time": 0.0,
        "end_time": 1.0,
        "dt": 0.1,
        "_source_sets": {
            "NodeA": {
                "population": "NodeA" ,
                "compartment_set": [
                [1, 0, 0.5],
                [0, 0, 0.5],
                [2, 0, 0.5]
                ]
            },
        }
    }

    with patch("bluecellulab.reports.utils.resolve_source_nodes") as mock_resolve, \
         patch("bluecellulab.reports.utils.build_recording_sites") as mock_build:

        mock_resolve.return_value = (
            [0, 1, 2],
            [
                [0, "soma[0]", 0.5],
                [1, "dend[0]", 0.5],
                [2, "axon[0]", 0.5]
            ]
        )

        mock_build.return_value = {
            0: [(None, "soma[0]", 0.5)],
            1: [(None, "dend[0]", 0.5)],
            2: [(None, "axon[0]", 0.5)],
        }

        writer = CompartmentReportWriter(
            report_cfg=report_cfg,
            output_path=out_file,
            sim_dt=0.1
        )
        writer.write(cells=merged, tstart=0.0)

    assert out_file.exists(), "Report file was not created"

    with h5py.File(out_file, "r") as f:
        grp = f["/report/NodeA"]
        data = np.array(grp["data"])  # shape: (time_steps, nodes)
        node_ids = np.array(grp["mapping"]["node_ids"])

        assert node_ids.tolist() == [0, 1, 2], f"Unexpected node_id order: {node_ids.tolist()}"

        col_means = data.mean(axis=0)
        assert np.isclose(col_means[0], 10.0, atol=1e-6), f"Column 0 mean {col_means[0]} != 10.0"
        assert np.isclose(col_means[1], 20.0, atol=1e-6), f"Column 1 mean {col_means[1]} != 20.0"
        assert np.isclose(col_means[2], 30.0, atol=1e-6), f"Column 2 mean {col_means[2]} != 30.0"
