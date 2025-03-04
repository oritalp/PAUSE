#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run the Python script with arguments
python main.py --full_exp True --data cifar10 --model cnn3 --num_users 300 --num_users_per_round 15\
         --i_i_d False --delta_f 0.004 --epsilon_bar 10 --global_epochs 500 \
        --dirichlet_coeff 2 --choosing_users_verbose --seed 3 --max_iterations_sa_pause 4000 --beta 2 --alpha 10 --gamma 10 --accel_ucb_coeff 4 \
        --epsilon_sum_deascent_coeff 0.04