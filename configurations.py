import argparse
import torch
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()

    # store_true suprisingly means that if the argument is not given, it is False

    parser.add_argument("--full_exp", type=str2bool, default=True,
                        help=("if true, runs all the methods in the defined environment, else, running only the experiment \
                              defined in method_choosing_users") )
    parser.add_argument('--data', type=str, default='cifar10',
                        choices=['mnist', 'cifar10', "fashion mnist"],
                        help="dataset to use (mnist, cifar10, fashion mnist)")
    parser.add_argument("--i_i_d", type = str2bool, default=False,
                        help="if True, the data is distributed i.i.d, if False, the data is non-i.i.d")
    parser.add_argument("--wandb", type=str2bool, default=True,
                         help="use wandb for logging")
    parser.add_argument('--method_choosing_users', type=str, default='sa_pause',
                        choices=["sa_pause",'pause brute', 'random', 'all users', "fastest ones", "fraboni"],
                        help="method to choose users for each round")
    parser.add_argument('--model', type=str, default='cnn3',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn2, cnn3, mlp, linear)")
    parser.add_argument('--num_users', type=int, default=30,
                        help="number of users participating in the federated learning")
    parser.add_argument('--num_users_per_round', type=int, default=5,
                        help="number of users participating in each round")
    parser.add_argument('--global_epochs', type=int, default=400,
                        help="number of global epochs")
    parser.add_argument('--max_seconds', type=float, default=600,
                        help="max seconds to run the learning process")
    parser.add_argument('--epsilon_bar', type=float, default=100,
                        help="privacy budget (epsilon)")
    parser.add_argument('--epsilon_sum_deascent_coeff', type=float, default=0.04,
                        help="the coefficient for the deascent of the epsilon sum")
    parser.add_argument('--delta_f', type=float, default=0.012,
                        help="constant delta f, the sensitivity for the laplace noise")
    parser.add_argument('--accel_ucb_coeff', type=float, default=1,
                        help="the coefficient for the acceleration of the ucb")
    parser.add_argument('--alpha', type=float, default=20,
                        help="alpha parameter for the MAB")
    parser.add_argument('--beta', type=float, default=2,
                        help="beta parameter for the MAB")
    parser.add_argument('--gamma', type=float, default=40,
                        help="gamma parameter for the MAB")
    parser.add_argument("--alternative_privacy_reward", action='store_true',
                        help="if True, uses the variance reward instead of accumulated reward for the privacy reward")


    #verbose arguments
    parser.add_argument('--choosing_users_verbose', action='store_true',
                        help="weather to print the chosen users for each round with their g, delay, and ucb values")
    parser.add_argument('--sa_pause_verbose', action='store_true',
                        help="weather to print the sa_pause algorithm's progress")
    parser.add_argument('--snr_verbose', action='store_true',
                        help="weather to print the snr of the deltas theta for each user")

    #non-i.i.d arguments
    parser.add_argument("--dirichlet_coeff", type=float, default=3,
                        help = ("the coefficient for the dirichlet distribution that generates the data distribution, \
                                The larger the coeffiecient, the more uniform is the distribution")) 
    parser.add_argument("--label_dominance", type=float, default=0.25,
                        help = "Percentage of the data that is generated from the most dominant label")  

    #ploting arguments
    parser.add_argument("--bar_plot_interval", type=int, default=30,
                         help="the interval to plot the bar plot of the chosen users")
    
    #sa-pause arguments
    parser.add_argument('--max_iterations_sa_pause', type=int, default=500,
                        help="maximum number of iterations for the sa_pause algorithm")
    parser.add_argument('--sa_pause_simulation', action='store_true',
                        help="weather to perform sa_pause in simulation mode (outside the main code) or not")
    parser.add_argument('--max_time_sa_pause', type=float, default=600,
                        help="maximum seconds for the sa_pause algorithm")
    parser.add_argument('--pre_sa_pause_rounds', type=int, default=1,
                        help=("in the (num_of_users/num_of_users_per_round)*pre_sa_pause_rounds, sa_pause is not performed\
                              and the users are chosen uniformly. this value is deafult equal to 1 and should only be\
                              changed in simulations if the number of users is very large and the sa_pause algorithm is very slow"))
    parser.add_argument('--beta_max_reduction', type=float, default=70,
                        help="the acount we divide the beta_max we compute in sa_pause to accelerate the convergence")

    #things that I don't touch often:
    parser.add_argument('--data_truncation', default=None,
                        help="if None, the data is not truncated, if a number is given, the data is truncated to that number")
    parser.add_argument('--tau_min', type=float, default=0.05,
                        help = "minimum communication time for all users")
    parser.add_argument("--production", action='store_true',
                        help="if True, the code will run in production mode, if False, it will run in development mode")
    parser.add_argument('--privacy_noise', type=str, default='laplace')
    parser.add_argument('--privacy', action='store_false',
                        help="weather to perform privacy or not")
    parser.add_argument('--save_best_model', action='store_true',
                        help="weather to save the model eith the best accuracy on the validation set")
    parser.add_argument('--seed', type=float, default=1,
                        help="manual seed for reproducibility")    
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=20,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")
    parser.add_argument('--local_epochs', type=int, default=1,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=100,
                        help="maximum number of iterations for the local training process")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate is 0.01 for cnn and 0.1  for linear")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_true',
                        help="reduce the learning rate when val_acc has stopped improving (increasing)")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")

    args = parser.parse_args()
    return args




