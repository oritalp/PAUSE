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
        self.inf_sum = np.exp(-self.args.epsilon_sum_deascent_coeff)/(1-np.exp(-self.args.epsilon_sum_deascent_coeff))
    
    def update_acceess_time(self):
            """
            Updates the last access time for the user.

            if the user index is smaller than half the number of users, the last access time is sampled from a gaussian
            with mean between 0.05 and 0.2 and std of 1/(2*num_users).
            else, the last access time is sampled from a gaussian with mean between 0.7 and 0.9 and std of 1/(2*num_users).
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
        partial_sum = 0
        i = 0
        while True:
            i += 1
            new_term = (self.args.epsilon_bar/self.inf_sum) * np.exp(-self.args.epsilon_sum_deascent_coeff*i)
            partial_sum += new_term
            yield new_term ,partial_sum

    def update_privacy_violation_and_reward(self):
        self.next_privacy_term, self.privacy_violation = next(self.privacy_series)
        self.privacy_reward = 1 - self.privacy_violation/self.args.epsilon_bar
        

        
    


def update_data_equility_partititon(local_models, args):
    """updates the data equility partition for each user according to the data quality and the number of samples after 
    the models for each user are created"""
    sum = 0 
    for i in range(len(local_models)):
        local_models[i].data_equility_partition = (local_models[i].data_quality
                                                   *len(local_models[i].data_loader.dataset))
        sum += local_models[i].data_equility_partition
    for i in range(len(local_models)):
        local_models[i].data_equility_partition = (local_models[i].data_equility_partition
                                                   *args.num_users_per_round/sum)



def compute_energy(users_idxes, local_models, args):
    """An auxilary function for the ALSA method, computes the energy of a given group of users according to the ALSA"""
    min_ucb = min([local_models[i].ucb for i in users_idxes])
    sum_g = args.alpha * (sum([local_models[i].g for i in users_idxes]) / args.num_users_per_round)
    sum_privacy_reward = (args.gamma * sum([local_models[i].privacy_reward for i in users_idxes])
                            / args.num_users_per_round)

    return min_ucb + sum_g + (sum_privacy_reward if args.privacy_choosing_users else 0)

def compute_relative_energy_of_neighbor(new_user, replaced_user, min_ucb_without_replaced_user,
                                        current_state, local_models, args, current_energy, neigbors_dict):

    """An auxilary function for the ALSA method, computes the relative energy of a neighboring set of the current state and
      adds it to the neighboring set dictionary. In addition, it returns the new enrgy and the new state."""
  
    copied_current_state = current_state.copy()
    copied_current_state.remove(replaced_user)
    copied_current_state.append(new_user)
    new_state = ",".join(str(user) for user in sorted(copied_current_state))
    new_energy = current_energy + (args.alpha * (local_models[new_user].g - local_models[replaced_user].g) 
                                    / args.num_users_per_round)
    if args.privacy_choosing_users:
        new_energy += (args.gamma * (local_models[new_user].privacy_reward - local_models[replaced_user].privacy_reward)
                        / args.num_users_per_round)

    if local_models[new_user].ucb < min_ucb_without_replaced_user:
        new_energy += (local_models[new_user].ucb - min_ucb_without_replaced_user)
    

    return new_state, new_energy

def create_neighbor_state(new_user, replaced_user, current_state, local_models, axis_to_sort_by = "ucb"):
    """An auxilary function for the ALSA method, gets a list of users as a current state and returns a new state
    as a string to be a key in the neigbors_dict dictionary."""
    copied_current_state = current_state.copy()
    copied_current_state.remove(replaced_user)
    copied_current_state.append(new_user)
    if axis_to_sort_by == "ucb":
        return ",".join(str(user) for user in sorted(copied_current_state, key = lambda x: local_models[x].ucb))
    
    elif axis_to_sort_by == "g":
        return ",".join(str(user) for user in sorted(copied_current_state, key = lambda x: local_models[x].g))
    elif axis_to_sort_by == "p":
        return ",".join(str(user) for user in sorted(copied_current_state, key = lambda x: local_models[x].privacy_reward))
    elif axis_to_sort_by == "user_idx":
        return ",".join(str(user) for user in sorted(copied_current_state))
    
    else:
        raise ValueError(f"the axis to sort by {axis_to_sort_by} is not valid, choose from ucb, g, p")
    

def create_passive_neighbor_states(sorted_current_state, sorted_all_users, current_state, local_models, axis_to_sort_by="ucb"):
    """
    Creates a dictionary of passive neighbor states based on the given parameters.

    Parameters:
    - sorted_current_state (list): A sorted list of the current state.
    - sorted_all_users (list): A sorted list of all users.
    - current_state (dict): The current state.
    - local_models (dict): A dictionary of local models.
    - axis_to_sort_by (str): The axis to sort the states by. Defaults to "ucb".

    Returns:
    - passive_neigbors_dict (dict): A dictionary of passive neighbor states.
    """
    min_current_state = sorted_current_state[0]
    second_min_current_state = sorted_current_state[1]

    passive_neigbors_dict = {}

    for nominated_new_user in sorted_all_users:
        if nominated_new_user == min_current_state:
            continue
        elif nominated_new_user == second_min_current_state:
            break
        else:
            replaced_user = min_current_state
            new_state = create_neighbor_state(nominated_new_user, replaced_user, current_state, local_models, axis_to_sort_by)
            passive_neigbors_dict[new_state] = None

    for replaced_user in sorted_current_state[1:]:
        for nominated_new_user in sorted_all_users:
            if nominated_new_user == min_current_state:
                break
            else:
                new_state = create_neighbor_state(nominated_new_user, replaced_user, current_state, local_models, axis_to_sort_by)
                passive_neigbors_dict[new_state] = None
    
    return passive_neigbors_dict

def sort_dict_keys_by_idx(original_dict):
    res_dict = {}
    for key in original_dict.keys():
        new_key = sorted(key.split(","))
        res_dict[",".join(new_key)] = original_dict[key]
    return res_dict



#TODO: after privacy issue is sealed, before publishing the code,
# need to change args.privacy_choosing_users and unite it with args.privacy
def choose_users(local_models,  args, global_epoch ,num_users = None, num_users_per_round = None, method="ALSA"):
    """
    Selects a group of users based on the specified method.

    Args:
        local_models (list): List of local models.
        args: Arguments for user selection.
        method (str, optional): The method for user selection. Defaults to "BSFL brute".
        privacy (bool, optional): Flag indicating whether privacy is considered. Defaults to False.

    Returns:
        tuple: A tuple containing the indices of the selected users.
    Raises:
        ValueError: If an invalid method is specified.
    """



    if method == "ALSA":
        


        if not args.ALSA_simulation:
            num_users_per_round = args.num_users_per_round
            num_users = args.num_users
        # before each user is chosen at least once, we choose the users randomly, because they all set
        # to have ucb which is eauals to inf

        #if we are in simulation mode, skip the initial random choosing
        3
        condition = global_epoch < math.floor(num_users/num_users_per_round) + 1 if  not args.ALSA_simulation else False
        #TODO: the condition for random initial starting is good if the number of users is a multiple of the number of 
        # users per round, otherwise, the condition should be changed (maybe with a change in the initialization of the
        # values of p and g in the user class)

        if condition:
            list_of_unchosen_users = [i for i in range(num_users) if local_models[i].num_of_obs == 0]
            print(list_of_unchosen_users)
            return tuple(np.random.choice(list_of_unchosen_users, num_users_per_round, replace=False))

        else:
            #create a uniform rnadom noise for the g and p axes, the noise should be uniformly bewtween -10^-7 and 10^-7
            g_noise = np.random.uniform(-10**-8, 10**-8, num_users)
            p_noise = np.random.uniform(-10**-8, 10**-8, num_users)

            for i in range(num_users):
                local_models[i].g += g_noise[i]
                local_models[i].privacy_reward += p_noise[i]


            current_state  = list(np.random.choice(num_users, num_users_per_round, replace=False))
            sorted_ucb = [model.user_idx for model in sorted([local_models[i] for i in range(num_users)], key = lambda x: x.ucb)]
            sorted_g = [model.user_idx for model in sorted([local_models[i] for i in range(num_users)], key = lambda x: x.g)]  
            sorted_privacy_reward = [model.user_idx for model in sorted([local_models[i] for i in range(num_users)], key = lambda x: x.privacy_reward)]  
            energy_list = []

            # computing the upper bound on the maximum of energy gap as beta_max

            upper_bound_energy = 0
            upper_bound_energy += local_models[sorted_ucb[-num_users_per_round]].ucb
            upper_bound_energy += (args.alpha/num_users_per_round) * sum([local_models[i].g for i in sorted_g[-num_users_per_round:]]) 
            upper_bound_energy += (args.gamma/num_users_per_round) * sum([local_models[i].privacy_reward for i in sorted_privacy_reward[-num_users_per_round:]])

            lower_bound_energy = 0
            lower_bound_energy += local_models[sorted_ucb[0]].ucb
            lower_bound_energy += (args.alpha/num_users_per_round) * sum([local_models[i].g for i in sorted_g[:num_users_per_round]])
            lower_bound_energy += (args.gamma/num_users_per_round) * sum([local_models[i].privacy_reward for i in sorted_privacy_reward[:num_users_per_round]])

            beta_max = upper_bound_energy - lower_bound_energy

            beta_max /= args.beta_max_reduction

            winning_comb = None
            best_score = 0

            if args.ALSA_verbose:
                print(f"iter: 0, current_state: {current_state}")

            for iter in range(args.max_iterations_alsa):
                
            
                ### part 1: cheking for active neighbors
                """Active neigbors are neighbors when the replaced user from the current state is the one with either 
                the minimal ucb, the minimal g, or the minimal privacy reward"""

                sorted_ucb_users_current_state = [model.user_idx for model in sorted([local_models[i] for i in current_state], key = lambda x: x.ucb)]
                sorted_g_users_current_state = [model.user_idx for model in sorted([local_models[i] for i in current_state], key = lambda x: x.g)]
                sorted_p_users_current_state = [model.user_idx for model in sorted([local_models[i] for i in current_state], key = lambda x: x.privacy_reward)]    


                min_ucb_value_current_state = local_models[sorted_ucb_users_current_state[0]].ucb
                min_g_value_current_state = local_models[sorted_g_users_current_state[0]].g
                min_privacy_reward_value_current_state = local_models[sorted_p_users_current_state[0]].privacy_reward

                active_neighbors_dict = {}
                for replaced_user in current_state:                                      
                    if ((local_models[replaced_user].ucb == min_ucb_value_current_state)
                         or (local_models[replaced_user].g == min_g_value_current_state)
                         or (local_models[replaced_user].privacy_reward == min_privacy_reward_value_current_state)):                        
                        range_of_users = list(range(num_users))
                        for new_user in range_of_users:
                            if new_user not in current_state:
                                # beacuse its actibe neighbor's part' we don't need to check for the energy of 
                                # new states yet
                                new_state = create_neighbor_state(new_user, replaced_user, current_state, local_models,
                                                                  axis_to_sort_by="user_idx")
                                # A None in neigbors_dict means that the new state is an active neighborand its energy wasnt
                                # computed yet
                                active_neighbors_dict[new_state] = None


                    else:
                        continue

                    #for debugging purposes

                if args.ALSA_verbose:
                    print("*"*10 + "active neighbors" + "*"*10)
                    print(f"iter: {iter}, current_state: {sorted(current_state)}")
                    print(f"min_ucb_idx_current_state: {sorted_ucb_users_current_state[0]}")
                    print(f"min_g_idx_current_state: {sorted_g_users_current_state[0]}")
                    print(f"min_privacy_reward_idx_current_state: {sorted_p_users_current_state[0]}")
                    for key in active_neighbors_dict.keys():
                        print(key)

                ### part 2: cheking for passive neighbors
                """Passive neigbors are neighbors we take out one user and add another one, the new one has the minimal value in 
                one of the axes of the g, ucb, or privacy reward in the new neighbor state we created."""


                #find the indexes of the users with the minimal ucb in the sorted ucb
                # in the average case, we are supposed to find about 3(K-m) passive neigbors, and after sorting the axes,
                # every such search is O(1)


                passive_neighbors_dict_ucb = create_passive_neighbor_states(sorted_ucb_users_current_state,
                                                                            sorted_ucb, current_state, local_models, 
                                                                            axis_to_sort_by="ucb")
                
                passive_neighbors_dict_g = create_passive_neighbor_states(sorted_g_users_current_state,
                                                                            sorted_g, current_state, local_models, 
                                                                            axis_to_sort_by="g")
                
                passive_neighbors_dict_p = create_passive_neighbor_states(sorted_p_users_current_state,  
                                                                            sorted_privacy_reward, current_state, local_models, 
                                                                            axis_to_sort_by="p")
                
                # sort the keys of the passive neighbors dict by the user_idx to be presented in an uniform order

                if args.ALSA_verbose:
                    print("*"*10 + "passive neighbors ucb" + "*"*10)
                    print(f"iter: {iter}, sorted_ucb_current_state: {sorted_ucb_users_current_state}")
                    print(f"sorted_ucb: {sorted_ucb}")
                    for key in passive_neighbors_dict_ucb.keys():
                        print(key)

                    print("*"*10 + "passive neighbors g" + "*"*10)
                    print(f"iter: {iter}, sorted_g_current_state: {sorted_g_users_current_state}")
                    print(f"sorted_g: {sorted_g}")
                    for key in passive_neighbors_dict_g.keys():
                        print(key)

                    print("*"*10 + "passive neighbors privacy" + "*"*10)
                    print(f"iter: {iter}, sorted_p_current_state: {sorted_p_users_current_state}")
                    print(f"sorted_privacy_reward: {sorted_privacy_reward}")
                    for key in passive_neighbors_dict_p.keys():
                        print(key)


                all_neighbors_dict = {**sort_dict_keys_by_idx(active_neighbors_dict),
                                      **sort_dict_keys_by_idx(passive_neighbors_dict_ucb),
                                       **sort_dict_keys_by_idx(passive_neighbors_dict_g),
                                         **sort_dict_keys_by_idx(passive_neighbors_dict_p)}
                


                current_energy = compute_energy(current_state, local_models, args)

                energy_list.append(current_energy)
                # print(f"iter: {iter}, current_state: {current_state}, current_energy: {current_energy}")

                # choose at random the next state from the neighbors dict
                next_state = np.random.choice(list(all_neighbors_dict.keys()))

                next_state = [int(x) for x in next_state.split(",")]
                next_energy = compute_energy(next_state, local_models, args)
                # print(f"iter: {iter}, next__propodes_state: {next_state}, next_energy: {next_energy}")

                if next_energy >= current_energy:
                    current_state = next_state
                    if next_energy > best_score:
                        best_score = next_energy
                        winning_comb = next_state


                else:
                    temp = beta_max / np.log(iter+1) if iter > 0 else float('inf')
                    prob = np.exp((next_energy - current_energy) / temp)
                    # print(f"temp: {temp}")
                    # print(f"iter: {iter}, prob: {prob}")

                    #flip a coin with the probability of prob
                    if np.random.rand() < prob:
                        current_state = next_state 


                    else:
                        continue
                
            for i in range(num_users):
                local_models[i].g -= g_noise[i]
                local_models[i].privacy_reward -= p_noise[i]

            if not args.ALSA_simulation:
                if len(set(winning_comb)) != args.num_users_per_round:
                    print(f"something went wrong, the winning comb is of the size {len(set(winning_comb))} and should be of the size {args.num_users_per_round}")
                return tuple(winning_comb)
            else:
                return energy_list, tuple(winning_comb), best_score

                


    elif method == "BSFL brute":
        users_idxs_comb = list(itertools.combinations([x for x in range(args.num_users)], args.num_users_per_round))
        # permute the users_idxs_comb to make the order of the users random
        np.random.shuffle(users_idxs_comb)
        winning_comb = None
        best_score = 0
        for comb in users_idxs_comb:
            score = compute_energy(comb, local_models, args)
            if score > best_score:
                best_score = score
                winning_comb = comb

        return winning_comb

    elif method == "random":
        return tuple(np.random.choice(args.num_users, args.num_users_per_round, replace=False))

    elif method == "all users":
        ret_val = list(range(args.num_users))
        np.random.shuffle(ret_val)
        return tuple(ret_val)

    elif method == "fastest ones":
        return tuple(range(args.num_users_per_round))

    else:
        raise ValueError(f"There is no such method as {method}, choose a method from:\nALSA, BSFL brute, random, all users, fastest ones ")
        






    


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
    (Path.cwd() / 'checkpoints' / args.method_choosing_users / args.model / now).mkdir(exist_ok=True, parents=True)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.method_choosing_users+ '/' + args.model + "/" + now)
    textio = IOStream('checkpoints/' + args.method_choosing_users +"/" + args.model + "/" + now + '/run.log')

    best_val_acc = np.NINF
    path_best_model = Path.cwd() / 'checkpoints' / args.method_choosing_users / args.model / now  /'best_model.pth.tar'
    last_model_path = Path.cwd() / 'checkpoints' / args.method_choosing_users / args.model / now  /'last_model.pth.tar'


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
        
    elif args.data == 'fashion mnist':
        train_data = datasets.FashionMNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
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
    Splits the given data into train and validation sets with possible truncation set by the args.data_truncation argument.

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
    if args.data_truncation is not None:
        train_data = torch.utils.data.Subset(train_data, range(args.data_truncation))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader

def plot_graphs(paths_dict: dict, x_axis_time = True):
    """
    Plots graphs for validation loss, validation accuracy, and average train loss over time or epochs.
    
    Args:
        paths_dict (dict): A dictionary containing the paths to the data for each graph.
        x_axis_time (bool, optional): Determines whether the x-axis represents time or epochs. 
                                      Defaults to True (time).
    """
    
    for key, value in paths_dict.items():
        paths_dict[key] = torch.load(value)
    
    fig, ax = plt.subplots(2,2, figsize=(15,15))

    for key, value in paths_dict.items():
        x_var = value["global_epochs_time_list"] if x_axis_time else range(1, value["global_epoch"]+1)
        ax[0,0].plot(x_var, value['val_losses_list'], label = f"{key} validation loss")
        ax[0,1].plot(x_var, value['val_acc_list'], label = f"{key} validation accuracy")
        ax[1,0].plot(x_var, value['train_loss_list'], label = f"{key} avg train loss")
    
    ax[0,0].set_title("val loss over time") if x_axis_time else ax[0,0].set_title("val loss over epochs")
    ax[0,0].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[0,0].set_xlabel("epochs", fontsize=10)
    ax[0,0].set_ylabel("val loss")
    ax[0,0].legend()

    ax[0,1].set_title("val acc over time") if x_axis_time else ax[0,1].set_title("val acc over epochs")
    ax[0,1].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[0,1].set_xlabel("epochs", fontsize=10)
    ax[0,1].set_ylabel("val acc")
    ax[0,1].legend()

    ax[1,0].set_title("avg train loss over time") if x_axis_time else ax[1,0].set_title("avg train loss over epochs")
    ax[1,0].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[1,0].set_xlabel("epochs", fontsize=10)
    ax[1,0].set_ylabel("train loss")
    ax[1,0].legend()

    plt.show()





