#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT"

python main.py --full_exp True --data cifar10 --model cnn3 --num_users 30 --num_users_per_round 5 \
         --i_i_d True --wandb True \
         --global_epochs 300 --max_seconds 600 --epsilon_bar 40.0 --epsilon_sum_deascent_coeff 0.04 \
         --delta_f 0.012 --alpha 100.0 --beta 2.0 --gamma 5.0 \
         --alternative_privacy_reward False --choosing_users_verbose \
         --dirichlet_coeff 1.0 --label_dominance 0.25 --bar_plot_interval 30 \
         --max_iterations_sa_pause 10000 --sa_pause_simulation False \
         --max_time_sa_pause 600 --pre_sa_pause_rounds 1 \
         --beta_max_reduction 10 --accel_ucb_coeff 1.0 \
         --tau_min 0.05 --privacy_noise laplace --privacy True \
         --seed 1.0 --norm_mean 0.5 --norm_std 0.5 \
         --train_batch_size 20 --test_batch_size 1000 \
         --local_epochs 1 --local_iterations 100 --lr 0.01 --momentum 0.5 \
         --optimizer adam