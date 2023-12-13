import argparse
import numpy as np



class arguments:

    def __init__(self, exp_name = "random selection", eval = False, data = "mnist", norm_std = 0.5,
                  norm_mean = 0.5, train_batch_size = 20, test_batch_size = 1000, 
                  model = "cnn2", num_users = 30, num_users_per_round = 5, local_epochs = 1,
                 local_iterations = 100, global_epochs = 200, tau_min = 0.05, privacy_noise = "laplace",
                   epsilon = 4, optimizer = "sgd", lr = 0.01, momentum = 0.5, lr_scheduler = True,
                 device = "cpu", seed = 0, zeta_coeff = 3/2, alpha = 1, beta = 2, gamma = 1, max_seconds = 200,
                 method_choosing_users = "random", data_truncation = 700,
                  choosing_users_verbose = False):
        self.exp_name = exp_name
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
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.seed = seed
        self.zeta_coeff = zeta_coeff
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_seconds = max_seconds
        self.method_choosing_users = method_choosing_users
        self.data_truncation = data_truncation
        self.choosing_users_verbose = choosing_users_verbose
        



def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--eval', action='store_true',
                        help="weather to perform inference of training")
    parser.add_argument('--choosing_users_verbose', action='store_true',
                        help="weather to print the chosen users for each round with their g, delay, and ucb values")

    # data arguments
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=20,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='linear',
                        choices=['cnn2', 'cnn3', 'mlp', 'linear'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--num_users', type=int, default=20,
                        help="number of users participating in the federated learning")
    parser.add_argument('--num_users_per_round', type=int, default=5,
                        help="number of users participating in each round")
    parser.add_argument('--local_epochs', type=int, default=5,
                        help="number of local epochs")
    parser.add_argument('--local_iterations', type=int, default=100,
                        help="number of local iterations instead of local epoch")
    parser.add_argument('--global_epochs', type=int, default=95,
                        help="number of global epochs")
    parser.add_argument('--tau_min', type=float, default=0.5)
    parser.add_argument('--method_choosing_users', type=str, default='brute force',
                        choices=['brute force', 'random', 'all users', "first ones"],
                        help="method to choose users for each round")

    # privacy arguments

    parser.add_argument('--privacy_noise', type=str, default='laplace',
                        choices=['laplace', 't', 'jopeq_scalar', 'jopeq_vector'],
                        help="types of PPNs to choose from")
    parser.add_argument('--epsilon', type=float, default=10,
                        help="privacy budget (epsilon)")
    parser.add_argument('--zeta_coeff', type=float, default=3/2, help="zeta coefficient for the privacy sum")
    

    # learning arguments
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate is 0.01 for cnn and 0.1  for linear")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', action='store_false',
                        help="reduce the learning rate when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda:0', 'cuda:1', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=0,
                        help="manual seed for reproducibility")
    parser.add_argument('--max_seconds', type=float, default=6000,
                        help="max seconds to run the learning process")

    # MAB arguments
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="alpha parameter for the MAB")
    parser.add_argument('--beta', type=float, default=2,
                        help="beta parameter for the MAB")
    parser.add_argument('--gamma', type=float, default=2,
                        help="gamma parameter for the MAB")
    parser.add_argument('--data_truncation', default=False,
                        help="if False, the data is not truncated, if a number is given, the data is truncated to that number")

    args = parser.parse_args()
    return args



