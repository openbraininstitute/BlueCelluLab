from pathlib import Path

import json

SIM_DIR = Path(__file__).parent.parent.absolute() / "examples" / "ringtest_allen_v1"


def test_cell_create(capsys):
    from bluecellulab import CircuitSimulation

    sim_conf = str(SIM_DIR / "simulation_config.json")
    bcl = CircuitSimulation(simulation_config=sim_conf)

    # Load configuration using json
    with open(sim_conf) as f:
        simulation_config_data = json.load(f)

    # Get the directory of the simulation config
    sim_config_base_dir = Path(sim_conf).parent

    # Get manifest path
    OUTPUT_DIR = simulation_config_data.get("manifest", {}).get("$OUTPUT_DIR", "./")

    # Get the node_set
    node_set_name = simulation_config_data.get("node_set", "All")

    node_sets_file = sim_config_base_dir / simulation_config_data["node_sets_file"]

    with open(node_sets_file) as f:
        node_set_data = json.load(f)

    # Get population and node IDs
    if node_set_name not in node_set_data:
        raise KeyError(f"Node set '{node_set_name}' not found in node sets file")

    population = node_set_data[node_set_name]["population"]
    all_node_ids = node_set_data[node_set_name]["node_id"]

    cell_ids_for_this_rank = [(population, i) for i in all_node_ids]

    bcl.instantiate_gids(cell_ids_for_this_rank)

    tau_vals = {0: 24.0, 1: 7.0, 2: 24.0}

    # verify that a point neuron has been created with IntFire and parameters set according to what was in the nodes.h5 file
    for (cell_id, cell) in bcl.cells.items():
        assert (tau_vals[cell_id.id] == cell.pointcell.pointcell.tau)
