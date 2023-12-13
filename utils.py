import numpy as np
import torch
import torch.optim as optim
import copy
import math
import os
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from pathlib import Path
from scipy import special
import itertools
import time
import datetime
import matplotlib.pyplot as plt



class user:
    def __init__(self, args, user_idx, data_loader, model, opt,
                  scheduler=None, data_quality = 1):
        self.user_idx = user_idx
        self.data_loader = data_loader
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.args = args
        self.ucb = float('inf')
        self.g = 0
        self.num_of_obs = 0
        self.emp_avg = 0
        self.data_quality = data_quality
        self.data_equility_partition = 0
        self.privacy_violation = 0
        self.privacy_series = iter(self.partial_sum_series())
        self.privacy_reward = 1
        self.last_access_time = 0
        self.next_privacy_term = 0
    
    def update_acceess_time(self):
            """
            Updates the last access time for the user.

            if the user index is smaller than half the number of users, the last access time is sampled from a gaussian
            with mean between 0.05 and 0.2 and std of 1/(2*num_of_users).
            else, the last access time is sampled from a gaussian with mean between 0.7 and 0.9 and std of 1/(2*num_of_users).
            meaning the first half is the fast achievable users and the second half is the slow achievable users.
            """
            
            num_of_half_the_users = math.floor(self.args.num_users//2)
            if self.user_idx < self.args.num_users//2:
                self.last_access_time = max(self.args.tau_min,
                                        (self.args.tau_min + ((0.2-self.args.tau_min)/num_of_half_the_users)*(self.user_idx+1)) 
                                         + (1/(2*self.args.num_users))*np.random.randn())
            else:
                self.last_access_time = max(self.args.tau_min,
                                            (0.7 + (0.2/(self.args.num_users - num_of_half_the_users))*(self.user_idx+1 - num_of_half_the_users)
                                             + (1/(2*self.args.num_users))*np.random.randn()))

        
            return self.args.tau_min/self.last_access_time
        
    
    def update_ucb(self, global_epoch):
        if self.num_of_obs == 0:
            return float('inf')
        else:
            self.ucb = self.emp_avg + math.sqrt((self.args.num_users*math.log(global_epoch))/self.num_of_obs)
            return self.ucb
    
    def update_emp_avg(self):
        """needed to be activeated before we increase the number of observations"""
        self.emp_avg = (self.num_of_obs*self.emp_avg + self.update_acceess_time())/(self.num_of_obs + 1)


    def increase_num_of_obs(self):
        self.num_of_obs += 1

    def update_g(self, global_epoch):
        inner_diff = self.data_equility_partition - self.num_of_obs/global_epoch
        self.g = (abs(inner_diff)**self.args.beta)*math.copysign(1,inner_diff)

    def partial_sum_series(self):
        i = 0
        partial_sum = 0
        while True:
            i += 1
            new_term = self.args.epsilon/((i**self.args.zeta_coeff)*special.zeta(self.args.zeta_coeff))
            partial_sum += new_term
            yield new_term ,partial_sum

    def update_privacy_violation_and_reward(self):
        self.next_privacy_term, self.privacy_violation = next(self.privacy_series)
        self.privacy_reward = 1 - self.privacy_violation/self.args.epsilon
        

        
    


def update_data_equility_partititon(local_models, args):
    sum = 0 
    for i in range(len(local_models)):
        local_models[i].data_equility_partition = (local_models[i].data_quality
                                                   *len(local_models[i].data_loader.dataset))
        sum += local_models[i].data_equility_partition
    for i in range(len(local_models)):
        local_models[i].data_equility_partition = (local_models[i].data_equility_partition
                                                   *args.num_users_per_round/sum)


def choose_users(local_models, args, method = "brute force", privacy = False):
    """
    Chooses the users that will be used for the current round.

    Args:
        initiated_delay (int): time in seconds to pause the other methods than brute force to make the running time
        of all methods about the same
        local_models (dict): A dictionary containing the local models for each user.
        args: Additional arguments for configuring the federated setup.

    Returns:
        list: A list containing the user indices of the chosen users.

    """

    if method == "brute force":
        users_idxs_comb = list(itertools.combinations([x for x in range(args.num_users)], args.num_users_per_round))
        #permute the users_idxs_comb to make the order of the users random
        np.random.shuffle(users_idxs_comb)
        winning_comb = None
        best_score = 0
        for comb in users_idxs_comb:
            min_ucb = min([local_models[i].ucb for i in comb])
            sum_g = args.alpha*sum([local_models[i].g for i in comb])/args.num_users_per_round
            sum_privacy_reward = (args.gamma*sum([local_models[i].privacy_reward for i in comb])
                                  /args.num_users_per_round)
            
            score = min_ucb + sum_g + (sum_privacy_reward if privacy else 0)
            if score > best_score:
                best_score = score
                winning_comb = comb
        
        return winning_comb
    
    if method == "random":
        return tuple(np.random.choice(args.num_users, args.num_users_per_round, replace = False))
    
    if method == "all users":
        return tuple(range(args.num_users))
    
    if method == "first ones":
        return tuple(range(args.num_users_per_round))






    

#TODO: check if the non-i.i.d partition is doing fine
def federated_setup(global_model, train_data: torch.utils.data.Dataset , args, i_i_d = True):
    """
    Sets up the federated learning environment by creating local models for each user.

    Args:
        global_model (torch.nn.Module): The global model to be used as a starting point for each local model.
        train_data (torch.utils.data.Dataset): The training dataset.
        args: Additional arguments for configuring the federated setup.

    Returns:
        dict: A dictionary containing the local models for each user.

    """
    indexes = torch.randperm(len(train_data))
    local_models = {}


    #creating non-iid data partition
    if not i_i_d:
        idxs_of_indices = np.array([0])

        #checking that there is at least 20 samples for each user
        while 0 in (idxs_of_indices>=20):
            probs = np.random.uniform(0,10, args.num_users)
            probs = probs/sum(probs)
            if probs.sum()!=1:
                probs[-1] = 1-sum(probs[:-1])
            idxs_of_indices = np.random.multinomial(len(train_data), probs)
            
        idxs_of_indices = np.cumsum(idxs_of_indices)
        idxs_of_indices = np.insert(idxs_of_indices, 0, 0)

    user_data_len = math.floor(len(train_data) / args.num_users)
    #TODO: why using deepcopy for the model?
    for user_idx in range(args.num_users):
        user_dict = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    (indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len] if i_i_d
                                      else indexes[idxs_of_indices[user_idx]:idxs_of_indices[user_idx+1]])),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user_dict['opt'] = optim.SGD(user_dict['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user_dict['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user_dict['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user_dict['opt'], patience=10,
                                                                           factor=0.1, verbose=True)
        local_models[user_idx] = user(args, user_idx, user_dict['data'], user_dict['model'], user_dict['opt'],
                                    user_dict['scheduler'] if args.lr_scheduler else None)
        

    return local_models


def initializations(args):
    """
    Sets the experiment deterministicly as possible for reproducibility, 
    create the relevant folders and perform necessary initializations for the experiment

    Args:
        args: An object containing the experiment arguments.

    Returns:
        boardio: SummaryWriter object for writing TensorBoard logs.
        textio: IOStream object for writing experiment logs.
        best_val_acc: value for the best validation accuracy, initially set to -inf.
        path_best_model: Path to save the best model.

    """
    #  reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    #  documentation
    #make a string for the current datetime (for the folder name) using datetime module
    now = datetime.datetime.now()
    now = str(now.strftime("%d-%m-%Y_%H-%M-%S"))
    (Path.cwd() / 'checkpoints' / args.exp_name / args.model / now).mkdir(exist_ok=True, parents=True)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name+ '/' + args.model + "/" + now)
    textio = IOStream('checkpoints/' + args.exp_name +"/" + args.model + "/" + now + '/run.log')

    best_val_acc = np.NINF
    path_best_model = Path.cwd() / 'checkpoints' / args.exp_name / args.model / now  /'best_model.pth.tar'
    last_model_path = Path.cwd() / 'checkpoints' / args.exp_name / args.model / now  /'last_model.pth.tar'


    return boardio, textio, best_val_acc, path_best_model, last_model_path


class IOStream:
    """A class for input/output operations.
    self.f is the internal file object.
    cprint: prints to console and writes to the end of the file following '/n'."""

    def __init__(self, path):
        self.f = open(path, 'a', encoding='utf-8', errors='ignore')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def data(args):
    """Returns train dataset and test loader (which is set to be with no shuffling, need to check why)
    the data is pictures with pixels normalized to be between 0 and 1, the normalizing mean and std are 0.5 both so 
    pixel with value 0 is -1 and pixel with value 1 is 1. this applies only a linear invertible transformation to the data"""
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
        

    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
        


    return train_data, test_loader


def data_split(data, amount, args):
    """
    Splits the given data into train and validation sets.

    Args:
        data (torch.utils.data.Dataset): The dataset to be split.
        amount (int): The number of samples to be included in the validation set.
        args: additional information variable.

    Returns:
        tuple: A tuple containing the following elements:
            - input (int): The size of the picture in linears models or the number of channels in CNN models.
            - output (int): The number of classes.
            - train_data (torch.utils.data.Dataset): The training dataset.
            - val_loader (torch.utils.data.DataLoader): The validation data loader.
    """
    
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    #the train data is truncated to the first args.data_truncation samples
    if args.data_truncation:
        train_data = torch.utils.data.Subset(train_data, range(args.data_truncation))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader

def plot_graphs(paths_dict: dict):
    """
    Plots graphs for the given paths dictionary.

    Args:
        paths_dict (dict): A dictionary containing mathods names as keys and paths as values.

    Returns:
        None
    """
    
    for key, value in paths_dict:
        paths_dict[key] = torch.load(value)
    
    fig, ax = plt.subplots(2,2, figsize=(15,15))

    for key, value in paths_dict.items():
        ax[0,0].plot(value["global_epochs_time_list"], value['val_losses_list'], label = f"{key} validation loss")
        ax[0,1].plot(value["global_epochs_time_list"], value['val_acc_list'], label = f"{key} validation accuracy")
        ax[1,0].plot(value["global_epochs_time_list"], value['train_loss_list'], label = f"{key} avg train loss")
    
    ax[0,0].set_title("val loss over time")
    ax[0,0].set_xlabel("time(sec)")
    ax[0,0].set_ylabel("val loss")
    ax[0,0].legend()

    ax[0,1].set_title("val acc over time")
    ax[0,1].set_xlabel("time(sec)")
    ax[0,1].set_ylabel("val acc")
    ax[0,1].legend()

    ax[1,0].set_title("avg train loss over time")
    ax[1,0].set_xlabel("time(sec)")
    ax[1,0].set_ylabel("train loss")
    ax[1,0].legend()

    plt.show()





