#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition main			### specify partition name where to run a job. change only if you have a matching qos!! main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-16:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name cifar_cnn5_all_users			### name of the job
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
python main.py --full_exp False --data cifar10 --i_i_d False --method_choosing_users "random" --model cnn3 --num_users 30 \
             --num_users_per_round 5 --delta_f 0.012 --accel_ucb_coeff 1 \
            --epsilon_bar 100 --choosing_users_verbose --seed 2 --global_epochs 400 --max_iterations_sa_pause 500\
            --beta 2 --alpha 20 --gamma 40 --dirichlet_coeff 2 --privacy True --epsilon_sum_deascent_coeff 0.04 
