#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT"

# Run the Python script with arguments
python main.py --full_exp True --data mnist --i_i_d True --model mlp --num_users 300 --num_users_per_round 15 --delta_f 0.003 \
            --epsilon_bar 20 --choosing_users_verbose --seed 0 --global_epochs 300 --accel_ucb_coeff 3 \
            --beta 1 --alpha 20 --gamma 10 --epsilon_sum_deascent_coeff 0.04