# PAUSE

This is the code repository for the PAUSE (Privacy-aware Active
User SElection) algorihm introcuded in the paper "Active User Selection for Low Latency and Privacy
Enhanced Federated Learning" by Peleg et al. (2025).

In the following you can find the instructions for setting up the environment and running the simulations as presented in the paper.

## Installation

To set up the environment for this project, follow these steps:

1. Create a Python 3.11 Conda environment named PAUSE:
   ```bash
   conda create -n PAUSE python=3.11
   ```

2. Activate the Conda environment:
   ```bash
   conda activate PAUSE
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running Simulations

The simulation bash for the full experiments scripts are located in the `simulations_runs` directory along with the according hyperparameters.

The full settings options are availabale and described in the configuration file `configurations.py`.

 To run the simulations:

1. Ensure your PAUSE environment is activated:
   ```bash
   conda activate PAUSE
   ```

2. Make the simulation scripts executable (if on macOS/Linux):
   ```bash
   chmod +x simulations_runs/*.sh
   ```

3. Run a simulation script:
   ```bash
   # On macOS/Linux
   ./simulations_runs/simulation_name.sh
   
   # On Windows
   bash simulations_runs/simulation_name.sh
   ```

Replace `simulation_name.sh` with the name of the specific simulation script you want to run.


Note: Besides the full experiments simulations, once can aso run specific simulations by putting the "--full_exp" flag off.