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

import matplotlib.pyplot as plt
import pathlib

from bluecellulab.analysis.inject_sequence import run_stimulus
from bluecellulab.stimulus.factory import StimulusFactory
from bluecellulab.tools import calculate_rheobase


def spiking_test(cell, rheobase, out_dir):
    """Run the spiking test on the cell."""
    stim_factory = StimulusFactory(dt=1.0)
    step_stimulus = stim_factory.idrest(threshold_current=rheobase, threshold_percentage=200)
    recording = run_stimulus(
        cell.template_params,
        step_stimulus,
        "soma[0]",
        0.5,
        add_hypamp=False,
        enable_spike_detection=True,
    )
    passed = recording.spike is not None and len(recording.spike) > 0

    # plotting
    outpath = out_dir / "spiking_test.png"
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.plot(recording.time, recording.voltage, color='black')
    current_axis = ax1.twinx()
    current_axis.plot(recording.time, recording.current, color="gray", alpha=0.6)
    current_axis.set_ylabel("Stimulus Current [nA]")
    fig.suptitle("Spiking Test - Step at 200% of Rheobase")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Voltage [mV]")
    fig.tight_layout()
    plt.show()
    fig.savefig(outpath, dpi=400)
    return {
        "run": True,
        "skipped": False,
        "passed": passed,
        "figures": [outpath],
    }


def run_validations(cell, cell_name):
    """Run all the validations on the cell.
    
    Args:
        cell (Cell): The cell to validate.
        cell_name (str): The name of the cell, used in the output directory.
    """
    out_dir = pathlib.Path("memodel_validation_figures") / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # cell = Cell.from_template_parameters(template_params)
    # do we already have rheobase in the cell?
    rheobase = calculate_rheobase(cell=cell, section="soma[0]", segx=0.5)

    # Validation 1: Spiking Test
    spiking_test_result = spiking_test(cell, rheobase, out_dir)

    return {
        "spiking_test": spiking_test_result,
    }

