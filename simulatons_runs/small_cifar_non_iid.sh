#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT"

python main.py --full_exp True --data cifar10 --model cnn3 --num_users 30 --num_users_per_round 5\
         --i_i_d False --delta_f 0.012 --epsilon_bar 100 --global_epochs 300 \
        --dirichlet_coeff 2 --choosing_users_verbose --seed 1 --max_iterations_sa_pause 1000 --beta 1 --alpha 5 --gamma 20 --accel_ucb_coeff 1 \
        --epsilon_sum_deascent_coeff 0.04