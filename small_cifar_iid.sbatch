#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name small_cifar			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
#SBATCH --gpus=rtx_4090:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
#SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=48G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate PAUSE				### activate a conda environment, replace my_env with your conda environment
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
