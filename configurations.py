import argparse
import torch
import numpy as np



class arguments:

    def __init__(self, method_choosing_users = "fastest ones", data_truncation = 2000, model = "mlp",
                  num_users = 30, num_users_per_round = 5, data = "mnist", 
                  save_best_model = False, global_epochs = 300, max_seconds = 300, privacy = True,
                  privacy_choosing_users = True, epsilon_bar = 200, epsilon_sum_deascent_coeff = 0.04,
                  delta_f = 0.3*(10**-2), snr_verbose = True, choosing_users_verbose = True, 
                  max_iterations_alsa = 500, ALSA_simulation = False, ALSA_verbose = False,
                  alpha = 10**2, beta = 2, gamma = 5, accel_ucb_coeff = 1, pre_ALSA_rounds = 1,
                  beta_max_reduction = 30, max_time_alsa = 600,
                  norm_std = 0.5, norm_mean = 0.5, train_batch_size = 20, test_batch_size = 1000, local_epochs = 1,
                 local_iterations = 100, tau_min = 0.05, privacy_noise = "laplace",
                  optimizer = "Adam", lr = 0.01, momentum = 0.5, lr_scheduler = False,
                    seed = 0):
        self.eval = eval
        self.data = data
        self.norm_std = norm_std
        self.norm_mean = norm_mean
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.model = model
        self.num_users = num_users
        self.num_users_per_round = num_users_per_round
        self.local_epochs = local_epochs
        self.local_iterations = local_iterations
        self.global_epochs = global_epochs
        self.tau_min = tau_min
        self.privacy_noise = privacy_noise
        self.epsilon_bar = epsilon_bar
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.seed = seed
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_seconds = max_seconds
        self.method_choosing_users = method_choosing_users
        self.data_truncation = data_truncation
        self.choosing_users_verbose = choosing_users_verbose
        self.save_best_model = save_best_model
        self.privacy = privacy
        self.privacy_choosing_users = privacy_choosing_users
        self.epsilon_sum_deascent_coeff = epsilon_sum_deascent_coeff #the coefficient for the deascent of the epsilon sum
        self.delta_f = delta_f #constant delta f
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snr_verbose = snr_verbose  #weather to print the snr of the deltas theta for each user
        self.max_iterations_alsa = max_iterations_alsa
        self.ALSA_simulation = ALSA_simulation
        self.ALSA_verbose = ALSA_verbose
        self.beta_max_reduction = beta_max_reduction
        self.accel_ucb_coeff = accel_ucb_coeff
        self.max_time_alsa = max_time_alsa
        self.pre_ALSA_rounds = pre_ALSA_rounds



def args_parser():
    parser = argparse.ArgumentParser()

    # store_true suprisingly means that if the argument is not given, it is False

    parser.add_argument('--privacy', action='store_false',
                        help="weather to perform privacy or not")
    parser.add_argument('--choosing_users_verbose', action='store_false',
                        help="weather to print the chosen users for each round with their g, delay, and ucb values")
    parser.add_argument('--save_best_model', action='store_true',
                        help="weather to save the model eith the best accuracy on the validation set")
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10', "fashion mnist"],
                        help="dataset to use (mnist, cifar10, fashion mnist)")


    # federated arguments
    parser.add_argument('--method_choosing_users', type=str, default='ALSA',
                        choices=["ALSA",'BSFL brute', 'random', 'all users', "fastest ones"],
                        help="method to choose users for each round")
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn2, cnn3, mlp, linear)")
    parser.add_argument('--num_users', type=int, default=300,
                        help="number of users participating in the federated learning")
    parser.add_argument('--num_users_per_round', type=int, default=15,
                        help="number of users participating in each round")
    parser.add_argument('--global_epochs', type=int, default=300,
                        help="number of global epochs")
    parser.add_argument('--max_seconds', type=float, default=600,
                        help="max seconds to run the learning process")
    parser.add_argument('--privacy_choosing_users', action='store_false',
                        help="weather to perform privacy for the choosing users or not")
    parser.add_argument('--max_iterations_alsa', type=int, default=3000,
                        help="maximum number of iterations for the ALSA algorithm")
    parser.add_argument('--ALSA_simulation', action='store_true',
                        help="weather to perform ALSA in simulation mode (outside the main code) or not")
    parser.add_argument('--epsilon_bar', type=float, default=100,
                        help="privacy budget (epsilon)")
    parser.add_argument('--epsilon_sum_deascent_coeff', type=float, default=0.04,
                        help="the coefficient for the deascent of the epsilon sum")
    parser.add_argument('--delta_f', type=float, default=0.3*(10**-2),
                        help="constant delta f, the sensitivity for the laplace noise")
    parser.add_argument('--accel_ucb_coeff', type=float, default=3,
                        help="the coefficient for the acceleration of the ucb")
    parser.add_argument('--pre_ALSA_rounds', type=int, default=1,
                        help=("in the (num_of_users/num_of_users_per_round)*pre_ALSA_rounds, ALSA is not performed\
                              and the users are chosen uniformly. this value is deafult equal to 1 and should only be\
                              changed in simulations if the number of users is very large and the ALSA algorithm is very slow"))
    parser.add_argument('--max_time_alsa', type=float, default=600,
                        help="maximum seconds for the ALSA algorithm")
    parser.add_argument('--snr_verbose', action='store_false',
                        help="weather to print the snr of the deltas theta for each user")
    parser.add_argument('--ALSA_verbose', action='store_true',
                        help="weather to print the ALSA algorithm's progress")
    parser.add_argument('--beta_max_reduction', type=float, default=70,
                        help="the aonut we divide the beta_max we compute in ALSA to accelerate the convergence")
    parser.add_argument('--alpha', type=float, default=100,
                        help="alpha parameter for the MAB")
    parser.add_argument('--beta', type=float, default=2,
                        help="beta parameter for the MAB")
    parser.add_argument('--gamma', type=float, default=2,
                        help="gamma parameter for the MAB")
    parser.add_argument('--data_truncation', default=None,
                        help="if None, the data is not truncated, if a number is given, the data is truncated to that number")
    parser.add_argument('--tau_min', type=float, default=0.05,
                        help = "minimum communication time for all users")
    

    #things that I don't touch often:
    parser.add_argument('--seed', type=float, default=0,
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
    parser.add_argument('--lr_scheduler', action='store_True',
                        help="reduce the learning rate when val_acc has stopped improving (increasing)")
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")

    args = parser.parse_args()
    return args



