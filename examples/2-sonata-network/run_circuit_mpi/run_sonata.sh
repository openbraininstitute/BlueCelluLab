#!/bin/bash
## This example runs the SONATA circuit with morphologies in an h5 container
## The circuit is defined in the container_nbS1-O1__202247__cADpyr__L5_TPC_A folder
## The simulation config and nodes file are in the container_configs_nbS1-O1__202247__cADpyr__L5_TPC_A folder
## The scipt calls the run_bluecellulab_simulation.py script to run the simulation 
## with the specified simulation config and uses num_cores cores to run the simulation
## in parallel using mpiexec.

source ../../../.venv/bin/activate

# Remove old compiled mod files
rm -r arm64/ # for mac-based systems
rm -r x86_64/ # for linux-based systems
 
circuit_folder="../../../tests/examples/container_nbS1-O1__202247__cADpyr__L5_TPC_A"
simulation_config="simulation_config_container.json"

## Change the simulation_config to run different simulation configs 
## to test with different morphology formats

# simulation_config="simulation_config_h5.json"
# simulation_config="simulation_config_swc.json"
# simulation_config="simulation_config_asc.json"

# Compile mod files
# flag DISABLE_REPORTINGLIB to skip SonataReportHelper.mod and SonataReport.mod from compilation.
nrnivmodl -incflags "-DDISABLE_REPORTINGLIB" $circuit_folder/mod  # Replace with the actual path to your mod files

echo "Running circuit simulation"
# NB: num_cores should be less than or equal to number of cells to be simulated
num_cores=1
mpiexec -n $num_cores python run_bluecellulab_simulation.py --simulation_config $simulation_config --save-nwb