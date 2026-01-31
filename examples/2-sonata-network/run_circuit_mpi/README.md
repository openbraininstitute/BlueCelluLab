# SONATA Circuit with H5 Container Morphologies

## Introduction

This directory contains an example of running a SONATA circuit simulation using BlueCelluLab with morphologies stored in H5 containers. See example 1 singlecell_H5_morph_H5_container.ipynb for more details of morphology formats.

## Circuit Details
For this example, we will use a `cADpyr_L5TPC` single cell [SONATA](https://sonata-extension.readthedocs.io/en/latest/) model used in the circuit with all the intrinsic and extrinsic synapses. This cell was programmatically extracted from the Somatosensory cortex(SSCx) circuit publish in [Modeling and simulation of neocortical micro- and mesocircuitry (Part II, Physiology and experimentation)](https://doi.org/10.7554/eLife.99693.3)

We also call such single cells with all its synapses, a **Synaptome**.

A synaptome named **nbS1-O1__202247__cADpyr__L5_TPC_A** is used in this example. For more details:

**[Open the synaptome in Open Brain Institute's Virtual Labs ](https://www.openbraininstitute.org/app/entity/82bae430-aff9-46b1-8ce7-e616bc733ed2)**.

Once you login, you can view all the details, subcircuits and circuit analysis on the detailed circuit page.

Locally, it is present in this folder: 
`BlueCelluLab/tests/examples/container_nbS1-O1__202247__cADpyr__L5_TPC_A`

The cell is stored in SONATA format with all the components including the H5 morphology. We also have the morphology in a H5 container containing a single morphology. Other 

## Key Features
- **H5 Container Morphologies**: Uses morphologies stored in H5 containers for efficient storage and access
- **MPI Support**: Configurable parallel execution using MPI
- **NWB Export**: Optional Neurodata Without Borders format output
- **Flexible Stimuli**: Multiple current clamp stimulus configurations
- **Comprehensive Reporting**: Soma voltage reporting with configurable sampling


## Core Configuration Files
- **`run_sonata.sh`** - Bash script to execute the simulation
- **`run_bluecellulab_simulation.py`** - Python simulation runner script
- **`simulation_config_container.json`** - Main simulation configuration file
- **`node_sets.json`** - Node set definitions for the circuit
- The circuit is defined in the "../../../tests/examples/container_nbS1-O1__202247__cADpyr__L5_TPC_A" folder.

The folder also contains `simulation_config_swc.json`, `simulation_config_asc.json` and `simulation_config_h5.json` files for running the simulation with swc, asc and h5 individual morphology formats respectively. Change the `simulation_config` variable in `run_sonata.sh` to run different simulation configs  to simulate with different morphology formats.

## Prerequisites
- Use a virtual environment with BlueCelluLab installed
- MPI installation: mpi4py or openmpi

## Running the Simulation
**Execute the simulation script:**
   Update the virtual environment path in the run_sonata.sh script if needed.
   The script supports parallel execution via MPI. Modify the `num_cores` variable in `run_sonata.sh`:
   It is set to 1 by default.
   ```bash
   ./run_sonata.sh
   ```

** Or run directly with Python:**
   ```bash
   python run_bluecellulab_simulation.py --simulation_config simulation_config_container.json --save-nwb
   ```

NB: The `num_cores=1` is set the `run_sonata.sh` script on a single core. `num_cores` is the number of cores to use for parallel execution and should be less or equal to the number of cells to be be simualted. 

## Output
The outputs are stored in the `output_container` directory. The `output_swc`, `output_asc`, `output_h5` directories are used for storing the outputs of the swc, asc and individual h5 morphology file formats respectively. These 

- **Spikes file**: `spikes.h5`
- **Soma report**: `soma_report.h5`
- **NWB file**: soma recordings from all neurons of the node set: `simulation_YYYYMMDD_HHMMSS.nwb`

Logs are stored in the `logs` directory: `simulation_YYYYMMDD_HHMMSS.log`
