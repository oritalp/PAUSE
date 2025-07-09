#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-10:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name tiny_imagenet			### name of the job
#SBATCH --output job-%J.out			### output log for running job - %J for job number
#SBATCH --gpus=rtx_4090:1				### number of GPUs, allocating more than 1 requires IT team's permission. Example to request 3090 gpu: #SBATCH --gpus=rtx_3090:1

# Note: the following 4 lines are commented out
#SBATCH --mail-user=user@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=60G				### ammount of RAM memory, allocating more than 60G requires IT team's permission

################  Following lines will be executed by a compute node    #######################

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate PAUSE				### activate a conda environment, replace my_env with your conda environment
python main.py --data tiny_imagenet --model mobilenetv2 --num_users 1 --num_users_per_round 5 \
                --mixed_precision True --train_batch_size 128 --test_batch_size 512 \
                --local_iterations 100 \
                --full_exp False --i_i_d True --method_choosing_users "all users" \
                --delta_f 0.2 --epsilon_bar 100 \
                --epsilon_sum_deascent_coeff 0.02 --global_epochs 300 --dirichlet_coeff 3  \
                --privacy False \
                --lr 0.005 \
                # --num_classes_subset 100 \
                # --data_truncation None \
                # --snr_verbose \ 


# --full_exp True --data cifar10 --model cnn3 --num_users 30 --num_users_per_round 5\
#          --i_i_d True --delta_f 0.012 --epsilon_bar 10 --global_epochs 300 \
#         --dirichlet_coeff 1 --choosing_users_verbose --seed 1 --max_iterations_sa_pause 1000 --beta 2 --alpha 100 --gamma 5 --accel_ucb_coeff 1 \
#         --epsilon_sum_deascent_coeff 0.04
