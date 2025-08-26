import numpy as np
from sympy import comp
import torch
import torch.optim as optim

import copy
import math
import os
from statistics import mean
from torchvision import datasets, transforms
from pathlib import Path
import itertools
import time
import datetime
import matplotlib.pyplot as plt
import random
import heapq
import pickle






class user:
    def __init__(self, args, user_idx, data_loader, model, opt,
                  scheduler=None, data_quality = 1):
        self.user_idx = user_idx
        self.data_loader = data_loader
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.args = args
        self.ucb_generalization = 0 #the term in ucb that corresponds to the generalization (namely h)
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

        # pay attention that the summation of \epsilon_i is starting from i=1
    
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
                                         + (1/20)*np.random.randn())
            else:
                self.last_access_time = max(self.args.tau_min,
                                            (0.7 + (0.2/(self.args.num_users - num_of_half_the_users))*(self.user_idx+1 - num_of_half_the_users)
                                             + (1/20)*np.random.randn()))

        
            return self.args.tau_min/self.last_access_time
    


        
    
    def update_ucb(self, global_epoch):
        if self.num_of_obs == 0:
            return float('inf')
        else:
            self.ucb_generalization = math.sqrt(((self.args.num_users_per_round+1)*math.log(global_epoch))/self.num_of_obs)
            self.ucb = self.args.accel_ucb_coeff*self.emp_avg + self.ucb_generalization
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

    def update_privacy_terms_and_violations(self):
        self.next_privacy_term, self.privacy_violation = next(self.privacy_series)
        if not self.args.alternative_privacy_reward:
            self.privacy_reward = 1 - self.privacy_violation/self.args.epsilon_bar

    
    def compute_privacy_term(self, num_of_obs):
        return self.args.epsilon_bar * np.exp(-self.args.epsilon_sum_deascent_coeff*num_of_obs)/self.inf_sum

    def compute_var_term(self, num_of_obs):
        return self.args.delta_f/self.compute_privacy_term(num_of_obs)
        
    
class HuggingFaceImageDataset(torch.utils.data.Dataset):
    """Dataset wrapper for Hugging Face datasets"""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_dataset)
        
    def __getitem__(self, idx):
        # FIX: Convert numpy int64 to Python int
        if hasattr(idx, 'item'):
            idx = idx.item()
        elif isinstance(idx, np.integer):
            idx = int(idx)
        
        sample = self.hf_dataset[idx]
        image = sample['image']
        label = sample['label']
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

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


def get_user_resource_cluster(user_idx, args):
    """
    Get the resource cluster ID for a given user.
    
    Args:
        user_idx: Index of the user
        args: Arguments containing num_users and clustering parameters
    
    Returns:
        int: Resource cluster ID (0 to ceil(N/S)-1), or None if feature disabled
    """
    if not args.shared_res_constraint:
        return None
    
    # Use clusters_partition_level if specified, otherwise use default formula
    if args.clusters_partition_level is not None:
        S = args.clusters_partition_level
    else:
        S = int(np.floor(np.sqrt(args.num_users) / 2))
    
    total_clusters = int(np.ceil(args.num_users / S))
    cluster_id = user_idx % total_clusters
    return cluster_id


def compute_resource_constraint_penalty(users_idxes, args):
    """
    Compute the resource constraint penalty for a given set of users.
    
    Args:
        users_idxes: List or array of user indices
        args: Arguments containing shared resource constraint parameters
    
    Returns:
        tuple: (latency_penalty, reward_penalty) both as float
    """
    if not args.shared_res_constraint:
        return 0.0, 0.0
    
    # Only count users in constrained clusters (0 to J-1)
    J = args.resources_cluster_num
    cluster_counts = np.zeros(J, dtype=int)
    
    for user_idx in users_idxes:
        cluster_id = get_user_resource_cluster(user_idx, args)
        # Only count if cluster is constrained (cluster_id < J)
        if cluster_id < J:
            cluster_counts[cluster_id] += 1
    
    # Calculate penalties: sum over constrained clusters of max(0, count-1)
    excess_users_per_cluster = np.maximum(0, cluster_counts - 1)
    total_excess_users = np.sum(excess_users_per_cluster)
    
    latency_penalty = 0.05 * total_excess_users
    reward_penalty = args.rho * total_excess_users
    
    return latency_penalty, reward_penalty


def get_resource_constraint_verbose_info(users_idxes, args):
    """
    Get verbose information about resource constraint violations.
    
    Args:
        users_idxes: List or array of user indices
        args: Arguments containing shared resource constraint parameters
    
    Returns:
        dict: Information about cluster collisions and penalties
    """
    if not args.shared_res_constraint:
        return {}
    
    # Only track constrained clusters (0 to J-1)
    J = args.resources_cluster_num
    cluster_counts = {}
    cluster_users = {}
    
    for user_idx in users_idxes:
        cluster_id = get_user_resource_cluster(user_idx, args)
        # Only track if cluster is constrained (cluster_id < J)
        if cluster_id < J:
            if cluster_id not in cluster_counts:
                cluster_counts[cluster_id] = 0
                cluster_users[cluster_id] = []
            cluster_counts[cluster_id] += 1
            cluster_users[cluster_id].append(user_idx)
    
    # Find clusters with collisions (more than 1 user)
    collisions = {}
    for cluster_id, count in cluster_counts.items():
        if count > 1:
            collisions[cluster_id] = {
                'users': cluster_users[cluster_id],
                'count': count,
                'excess': count - 1
            }
    
    latency_penalty, reward_penalty = compute_resource_constraint_penalty(users_idxes, args)
    
    return {
        'collisions': collisions,
        'latency_penalty': latency_penalty,
        'reward_penalty': reward_penalty,
        'total_excess_users': sum(info['excess'] for info in collisions.values())
    }


def compute_energy(users_idxes, local_models, args, num_users=None, num_users_per_round=None):
    """An auxilary function for the sa_pause method, 
    computes the energy of a given group of users according to the sa_pause
    Updated to use float64 precision for numerical stability."""
    if num_users is None and num_users_per_round is None:
        num_users = args.num_users
        num_users_per_round = args.num_users_per_round
    
    # Convert to float64 for numerical stability
    ucb_vals = np.array([local_models[i].ucb for i in users_idxes], dtype=np.float64)
    g_vals = np.array([local_models[i].g for i in users_idxes], dtype=np.float64)
    p_vals = np.array([local_models[i].privacy_reward for i in users_idxes], dtype=np.float64)
    
    # Perform calculations in float64
    min_ucb = np.min(ucb_vals)
    alpha_f64 = np.float64(args.alpha)
    gamma_f64 = np.float64(args.gamma)
    num_users_f64 = np.float64(num_users_per_round)
    
    sum_g = alpha_f64 * (np.sum(g_vals) / num_users_f64)
    sum_privacy_reward = (gamma_f64 * np.sum(p_vals) / num_users_f64) if args.privacy else np.float64(0.0)
    
    # Add shared resource constraint penalty
    resource_constraint_penalty = np.float64(0.0)
    if args.shared_res_constraint:
        _, reward_penalty = compute_resource_constraint_penalty(users_idxes, args)
        resource_constraint_penalty = np.float64(reward_penalty)

    return float(min_ucb + sum_g + sum_privacy_reward - resource_constraint_penalty)

def compute_relative_energy_of_neighbor(new_user, replaced_user, min_ucb_without_replaced_user,
                                        current_state, local_models, args, current_energy, neigbors_dict):

    """An auxilary function for the sa_pause method, computes the relative energy of a neighboring set of the current state and
      adds it to the neighboring set dictionary. In addition, it returns the new enrgy and the new state."""
  
    copied_current_state = current_state.copy()
    copied_current_state.remove(replaced_user)
    copied_current_state.append(new_user)
    new_state = ",".join(str(user) for user in sorted(copied_current_state))
    new_energy = current_energy + (args.alpha * (local_models[new_user].g - local_models[replaced_user].g) 
                                    / args.num_users_per_round)
    if args.privacy:
        new_energy += (args.gamma * (local_models[new_user].privacy_reward - local_models[replaced_user].privacy_reward)
                        / args.num_users_per_round)

    # Add shared resource constraint penalty difference
    if args.shared_res_constraint:
        # Calculate penalty for current state
        _, current_penalty = compute_resource_constraint_penalty(current_state, args)
        # Calculate penalty for new state
        _, new_penalty = compute_resource_constraint_penalty(copied_current_state, args)
        # Add the difference (penalty reduces energy, so we subtract)
        new_energy = new_energy + current_penalty - new_penalty

    if local_models[new_user].ucb < min_ucb_without_replaced_user:
        new_energy += (local_models[new_user].ucb - min_ucb_without_replaced_user)
    

    return new_state, new_energy

def create_neighbor_state(new_user, replaced_user, current_state, local_models, axis_to_sort_by = "ucb"):
    """An auxilary function for the sa_pause method, gets a list of users as a current state and returns a new state
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


def create_fraboni_probs(local_models, args, verbose = True):
    # made a slight modification to the algorithm shown in the paper because otherwise it's not correct
    m = args.num_users_per_round
    distributions = np.zeros((m + 1, args.num_users))
    
    # Get total number of samples across all clients
    M = 0
    for i in range(args.num_users):
        M += len(local_models[i].data_loader.dataset)
    
    # Order clients by descending sample size
    ordered_clients = sorted(range(args.num_users), 
                           key=lambda x: len(local_models[x].data_loader.dataset),
                           reverse=True)
    
    k = 0  # distribution index 
    q = 0  # running sum of samples
    prev_b = 0  # previous b value

    for i in ordered_clients:
        # Calculate scaled number of samples for client i
        ni_scaled = m * len(local_models[i].data_loader.dataset)
        q += ni_scaled
        
        # Integer division to get quotient ai and remainder bi
        a = q // M  # Number of full distributions
        b = q % M   # Remainder samples
        
        # If client fills multiple distributions
        if a > k:
            # Fill all distributions from k up to but not including ai with probability 1
            distributions[k,i] = (M - prev_b) / M
            if a-2 >= k:
                distributions[k+1:a, i] = 1
            
            distributions[a, i] = b / M
            
        else:
            distributions[k, i] = (b - prev_b) / M
        
        k = a
        prev_b = b
    
    # The m+1-th row is always all zeros and it's not needed
    # check if the sum of the last row is zero and if not yield an error
    if sum(distributions[-1]) != 0:
        raise ValueError("the last row of the fraboni distribution should be all zeros")
    else:
        distributions = distributions[:m]

    # check if the sum of every row is equal to 1 and if not yield an error
    if not np.allclose(distributions.sum(axis=1), np.ones(m, dtype=np.float32), rtol=1e-5, atol=1e-5):
        print("absolute difference", abs(sum(distributions.sum(axis=1) - np.ones(m, dtype=np.float32))))
        print("dist sums: ",distributions.sum(axis=1))
        print("wanted: ", np.ones(m, dtype=np.float32))
        print("comparison: ", distributions.sum(axis=1) != np.ones(m, dtype=np.float32))
        raise ValueError(f"the sum of every row in the fraboni distribution should be equal to 1, instead we got the following sums: {distributions.sum(axis=1)}")
    

    return distributions
    

def compute_energy_numpy(state_tuple, ucb_values, g_values, p_values, args, num_users_per_round, privacy=True):
    """
    Compute energy for a given state using numpy arrays for efficiency.
    Updated to use float64 precision for numerical stability.
    
    Args:
        state_tuple: Tuple of user indices in the state
        ucb_values: Numpy array of UCB values for all users (should be float64)
        g_values: Numpy array of G values for all users (should be float64)
        p_values: Numpy array of privacy values for all users (should be float64)
        args: Arguments object containing alpha and gamma parameters
        num_users_per_round: Number of users per round
        privacy: Whether to include privacy term (default True)
    
    Returns:
        float: Computed energy value
    """
    # Ensure state indices are proper integers
    state_arr = np.array(state_tuple, dtype=np.int64)
    
    # Extract values for selected users (already float64 from input arrays)
    ucb_vals = ucb_values[state_arr]
    g_vals = g_values[state_arr]
    p_vals = p_values[state_arr]
    
    # Perform calculations in float64
    min_ucb = np.min(ucb_vals)
    alpha_f64 = np.float64(args.alpha)
    gamma_f64 = np.float64(args.gamma)
    num_users_f64 = np.float64(num_users_per_round)
    
    sum_g = (alpha_f64 / num_users_f64) * np.sum(g_vals)
    sum_p = (gamma_f64 / num_users_f64) * np.sum(p_vals) if privacy and args.privacy else np.float64(0.0)
    
    # Add shared resource constraint penalty
    resource_constraint_penalty = np.float64(0.0)
    if args.shared_res_constraint:
        _, reward_penalty = compute_resource_constraint_penalty(state_tuple, args)
        resource_constraint_penalty = np.float64(reward_penalty)
    
    # Return as Python float (which is float64)
    return float(min_ucb + sum_g + sum_p - resource_constraint_penalty)

def save_sapause_data(global_epoch, local_models, args, num_users, timestamp):
    """
    Save SA-PAUSE optimization data for later analysis.
    Each epoch is saved in its own pickle file.
    
    Args:
        global_epoch: Current global epoch
        local_models: Dictionary of local models
        args: Arguments object
        num_users: Number of users
        timestamp: Timestamp when SA-PAUSE started
    """
    # Parse the range from args.save_data_global_epochs
    if args.save_data_global_epochs is None:
        return None
        
    try:
        # Parse range like "80,85" to (80, 85)
        range_parts = args.save_data_global_epochs.split(',')
        if len(range_parts) != 2:
            return None
        start_epoch, end_epoch = int(range_parts[0]), int(range_parts[1])
        
        # Check if current epoch is in range
        if not (start_epoch <= global_epoch <= end_epoch):
            return None
            
    except (ValueError, AttributeError):
        return None
    
    # Create data directory
    data_dir = Path('data_gathering_exp') / f"{args.save_data_global_epochs}_{timestamp}"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data: user_idx, ucb, g, p
    data_matrix = []
    for user_idx in range(num_users):
        user_data = [
            user_idx,
            float(local_models[user_idx].ucb),
            float(local_models[user_idx].g),
            float(local_models[user_idx].privacy_reward)
        ]
        data_matrix.append(user_data)
    
    # Convert to numpy array and sort by UCB (column 1)
    data_matrix = np.array(data_matrix)
    sorted_indices = np.argsort(data_matrix[:, 1])  # Sort by UCB column
    sorted_data = data_matrix[sorted_indices]
    
    # Save this epoch's data in its own file
    epoch_file = data_dir / f'epoch_{global_epoch}.pkl'
    with open(epoch_file, 'wb') as f:
        pickle.dump(sorted_data, f)
    
    return data_dir

def choose_users(local_models, args, global_epoch, textio, num_users=1,
                  num_users_per_round=1, method="sa_pause", target_energy=None):
    """
    Selects a group of users based on the specified method.
    Updated with float64 precision for SA-PAUSE and data gathering capability.
    """

    if method == "sa_pause":
        if not args.sa_pause_simulation:
            num_users_per_round = args.num_users_per_round
            num_users = args.num_users
        
        # Skip initial random choosing
        condition = (global_epoch <= (args.pre_sa_pause_rounds * args.num_users/args.num_users_per_round)
                      if not args.sa_pause_simulation else False)
        
        if condition:
            round_no = (global_epoch-1) // (args.num_users/args.num_users_per_round)
            list_of_unchosen_users = [i for i in range(num_users) if local_models[i].num_of_obs == round_no]
            result = tuple(np.random.choice(list_of_unchosen_users, num_users_per_round, replace=False))
            return result
        else:
            start_time = time.time()
            
            # Data gathering - save data before SA-PAUSE optimization starts
            if not args.sa_pause_simulation and hasattr(args, 'save_data_global_epochs') and args.save_data_global_epochs is not None:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                save_sapause_data(global_epoch, local_models, args, num_users, timestamp)
            
            accelerated_sa = args.sa_pause_accelerated
            
            # UCB-only mode: no noise addition
            if args.add_tiny_noise:
                # Original: Add tiny uniform random noise for g and p axes
                ucb_noise = np.random.uniform(-10**-8, 10**-8, num_users)
                g_noise = np.random.uniform(-10**-8, 10**-8, num_users)
                p_noise = np.random.uniform(-10**-8, 10**-8, num_users)

                for i in range(num_users):
                    local_models[i].ucb += ucb_noise[i]
                    local_models[i].g += g_noise[i]
                    local_models[i].privacy_reward += p_noise[i]

            current_state = list(np.random.choice(num_users, num_users_per_round, replace=False))
            
            # Pre-compute arrays with float64 precision
            ucb_values = np.array([local_models[i].ucb for i in range(num_users)], dtype=np.float64)
            g_values = np.array([local_models[i].g for i in range(num_users)], dtype=np.float64)
            p_values = np.array([local_models[i].privacy_reward for i in range(num_users)], dtype=np.float64)

            
            sorted_ucb_indices = np.argsort(ucb_values)
            
            if not args.ucb_neighbors_only:
                # Original: sort all three
                sorted_g_indices = np.argsort(g_values)
                sorted_p_indices = np.argsort(p_values)
            
            energy_list = []

            # Computing beta_max with float64 precision
            if args.ucb_neighbors_only:
                # UCB-only: use mth place UCB - min UCB + alpha/gamma terms
                mth_place_ucb = ucb_values[sorted_ucb_indices[num_users_per_round-1]]
                min_ucb = ucb_values[sorted_ucb_indices[0]]
                alpha_f64 = np.float64(args.alpha)
                gamma_f64 = np.float64(args.gamma)
                beta_max = (mth_place_ucb - min_ucb) + np.float64(2.0) * alpha_f64 + gamma_f64
            else:
                # Original beta_max computation with float64
                num_users_f64 = np.float64(num_users_per_round)
                alpha_f64 = np.float64(args.alpha)
                gamma_f64 = np.float64(args.gamma)
                
                upper_bound_energy = np.float64(0.0)
                upper_bound_energy += ucb_values[sorted_ucb_indices[-num_users_per_round:]][0]
                upper_bound_energy += (alpha_f64 / num_users_f64) * np.sum(g_values[sorted_g_indices[-num_users_per_round:]])
                if args.privacy:
                    upper_bound_energy += (gamma_f64 / num_users_f64) * np.sum(p_values[sorted_p_indices[-num_users_per_round:]])

                lower_bound_energy = np.float64(0.0)
                lower_bound_energy += ucb_values[sorted_ucb_indices[0]]
                lower_bound_energy += (alpha_f64 / num_users_f64) * np.sum(g_values[sorted_g_indices[:num_users_per_round]])
                if args.privacy:
                    lower_bound_energy += (gamma_f64 / num_users_f64) * np.sum(p_values[sorted_p_indices[:num_users_per_round]])

                beta_max = upper_bound_energy - lower_bound_energy

            # Update energy scale and threshold calculations
            energy_scale = float(beta_max)
            min_improvement_threshold = max(1e-8, energy_scale * 1e-8)  # 0.01% of energy scale or 1e-6, whichever is larger

            if args.sa_pause_verbose:
                print(f"Energy scale (beta_max): {energy_scale:.6f}")
                print(f"Minimum improvement threshold: {min_improvement_threshold:.6f}")

            beta_max /= args.beta_max_reduction

            winning_comb = []
            best_score = 0.0
            
            # Initialize cache variables properly
            previous_state_key = None
            cached_neighbors = None
            
            # Initialize counters for debugging
            cache_hits = 0
            cache_misses = 0
            neighbor_computation_time = 0
            neighbor_stats = []  # Store neighbor counts for statistics

            for iter in range(args.max_iterations_sa_pause):
                
                # Create consistent state key for caching
                current_state_key = tuple(sorted(current_state))
                
                # Check if we need to recompute neighbors (state changed)
                if previous_state_key != current_state_key:
                    # State changed, recompute neighbors
                    cache_misses += 1
                    neighbor_start_time = time.time()
                    
                    current_state_arr = np.array(current_state)
                    current_ucb = ucb_values[current_state_arr]
                    current_g = g_values[current_state_arr]
                    current_p = p_values[current_state_arr]
                    
                    min_ucb_value = np.min(current_ucb)
                    
                    if not args.ucb_neighbors_only:
                        min_g_value = np.min(current_g)
                        min_p_value = np.min(current_p)
                    
                    # Find users with minimum values in current state
                    min_ucb_users = current_state_arr[current_ucb == min_ucb_value]
                    
                    if not args.ucb_neighbors_only:
                        min_g_users = current_state_arr[current_g == min_g_value]
                        min_p_users = current_state_arr[current_p == min_p_value]
                    
                    all_neighbors = set()
                    neighbor_count = 0  # Track number of neighbors generated
                    
                    # [Continue with the same neighbor generation logic but using float64 arrays]
                    # The neighbor generation logic remains the same as in the original code
                    # but now operates on the float64 arrays
                    
                    if accelerated_sa:
                        if args.ucb_neighbors_only:
                            # UCB-only accelerated: only UCB-based strategies
                            strategies_attempted = []
                            max_attempts = 2  # Only active and passive UCB strategies
                            
                            while len(all_neighbors) == 0 and len(strategies_attempted) < max_attempts:
                                available_strategies = [('active', 'ucb'), ('passive', 'ucb')]
                                remaining_strategies = [s for s in available_strategies if s not in strategies_attempted]
                                
                                if not remaining_strategies:
                                    break
                                
                                strategy_idx = np.random.choice(len(remaining_strategies))
                                strategy_type, criterion = remaining_strategies[strategy_idx]
                                strategies_attempted.append((strategy_type, criterion))
                                
                                if strategy_type == 'active':
                                    # Active neighbors: replace minimal UCB users
                                    users_not_in_state = np.setdiff1d(np.arange(num_users), current_state_arr)
                                    for replaced_user in min_ucb_users:
                                        for new_user in users_not_in_state:
                                            new_state = current_state_arr.copy()
                                            new_state[new_state == replaced_user] = new_user
                                            all_neighbors.add(tuple(sorted(new_state)))
                                            neighbor_count += 1
                                            
                                            # Check neighbor limit
                                            if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                break
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break
                                
                                else:  # passive UCB
                                    current_ucb_sorted = current_state_arr[np.argsort(current_ucb)]
                                    min_ucb_current = current_ucb_sorted[0]
                                    second_min_ucb_current = current_ucb_sorted[1] if len(current_ucb_sorted) > 1 else min_ucb_current
                                    
                                    min_pos = np.where(sorted_ucb_indices == min_ucb_current)[0][0]
                                    second_min_pos = np.where(sorted_ucb_indices == second_min_ucb_current)[0][0]
                                    
                                    # Strategy 1: Replace min with better users
                                    for new_user in sorted_ucb_indices[:second_min_pos]:
                                        if new_user not in current_state_arr:
                                            new_state = current_state_arr.copy()
                                            new_state[new_state == min_ucb_current] = new_user
                                            all_neighbors.add(tuple(sorted(new_state)))
                                            neighbor_count += 1
                                            
                                            if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                break
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                                    
                                    # Strategy 2: Replace others with users better than min
                                    for replaced_user in current_ucb_sorted[1:]:
                                        for new_user in sorted_ucb_indices[:min_pos]:
                                            if new_user not in current_state_arr:
                                                new_state = current_state_arr.copy()
                                                new_state[new_state == replaced_user] = new_user
                                                all_neighbors.add(tuple(sorted(new_state)))
                                                neighbor_count += 1
                                                
                                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                    break
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break

                            if len(all_neighbors) == 0:
                                raise RuntimeError(f"UCB-only accelerated SA-PAUSE failed to find any neighbors at iteration {iter}.")
                        
                        else:
                            # Original accelerated logic (unchanged)
                            strategies_attempted = []
                            max_attempts = 6
                            
                            while len(all_neighbors) == 0 and len(strategies_attempted) < max_attempts:
                                available_strategies = [
                                    ('active', 'ucb'), ('active', 'g'), ('active', 'p'),
                                    ('passive', 'ucb'), ('passive', 'g'), ('passive', 'p')
                                ]
                                
                                remaining_strategies = [s for s in available_strategies if s not in strategies_attempted]
                                
                                if not remaining_strategies:
                                    break
                                
                                strategy_idx = np.random.choice(len(remaining_strategies))
                                strategy_type, criterion = remaining_strategies[strategy_idx]
                                strategies_attempted.append((strategy_type, criterion))
                                
                                if strategy_type == 'active':
                                    if criterion == 'ucb':
                                        replaceable_users = min_ucb_users
                                    elif criterion == 'g':
                                        replaceable_users = min_g_users
                                    else:  # 'p'
                                        replaceable_users = min_p_users
                                    
                                    users_not_in_state = np.setdiff1d(np.arange(num_users), current_state_arr)
                                    for replaced_user in replaceable_users:
                                        for new_user in users_not_in_state:
                                            new_state = current_state_arr.copy()
                                            new_state[new_state == replaced_user] = new_user
                                            all_neighbors.add(tuple(sorted(new_state)))
                                            neighbor_count += 1
                                            
                                            if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                break
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break
                                
                                else:  # passive
                                    if criterion == 'ucb':
                                        sorted_indices = sorted_ucb_indices
                                        current_sorted = current_state_arr[np.argsort(current_ucb)]
                                    elif criterion == 'g':
                                        sorted_indices = sorted_g_indices
                                        current_sorted = current_state_arr[np.argsort(current_g)]
                                    else:  # 'p'
                                        sorted_indices = sorted_p_indices
                                        current_sorted = current_state_arr[np.argsort(current_p)]
                                    
                                    min_user = current_sorted[0]
                                    second_min_user = current_sorted[1] if len(current_sorted) > 1 else min_user
                                    
                                    passive_strategies = ['replace_min', 'replace_others']
                                    np.random.shuffle(passive_strategies)
                                    
                                    for passive_strategy in passive_strategies:
                                        temp_neighbors = set()
                                        
                                        if passive_strategy == 'replace_min':
                                            min_pos = np.where(sorted_indices == min_user)[0][0]
                                            second_min_pos = np.where(sorted_indices == second_min_user)[0][0]
                                            
                                            better_users = sorted_indices[:second_min_pos]
                                            better_users = np.setdiff1d(better_users, current_state_arr)
                                            
                                            for new_user in better_users:
                                                new_state = current_state_arr.copy()
                                                new_state[new_state == min_user] = new_user
                                                temp_neighbors.add(tuple(sorted(new_state)))
                                                neighbor_count += 1
                                                
                                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                    break
                                        
                                        else:  # 'replace_others'
                                            min_pos = np.where(sorted_indices == min_user)[0][0]
                                            better_users = sorted_indices[:min_pos]
                                            
                                            for replaced_user in current_sorted[1:]:
                                                for new_user in better_users:
                                                    if new_user not in current_state_arr:
                                                        new_state = current_state_arr.copy()
                                                        new_state[new_state == replaced_user] = new_user
                                                        temp_neighbors.add(tuple(sorted(new_state)))
                                                        neighbor_count += 1
                                                        
                                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                            break
                                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                    break
                                        
                                        if temp_neighbors:
                                            all_neighbors.update(temp_neighbors)
                                            break
                                        
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break

                            if len(all_neighbors) == 0:
                                raise RuntimeError(f"Accelerated SA-PAUSE failed to find any neighbors at iteration {iter}.")
                    
                    else:
                        # Non-accelerated: comprehensive approach
                        if args.ucb_neighbors_only:
                            # UCB-only comprehensive
                            # Active neighbors
                            users_not_in_state = np.setdiff1d(np.arange(num_users), current_state_arr)
                            for replaced_user in min_ucb_users:
                                for new_user in users_not_in_state:
                                    new_state = current_state_arr.copy()
                                    new_state[new_state == replaced_user] = new_user
                                    all_neighbors.add(tuple(sorted(new_state)))
                                    neighbor_count += 1
                                    
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                    break

                            # Passive neighbors (UCB only) - only if we haven't hit limit
                            if args.max_neighbors is None or neighbor_count < args.max_neighbors:
                                current_ucb_sorted = current_state_arr[np.argsort(current_ucb)]
                                min_ucb_current = current_ucb_sorted[0]
                                second_min_ucb_current = current_ucb_sorted[1] if len(current_ucb_sorted) > 1 else min_ucb_current
                                
                                min_pos = np.where(sorted_ucb_indices == min_ucb_current)[0][0]
                                second_min_pos = np.where(sorted_ucb_indices == second_min_ucb_current)[0][0]
                                
                                # Strategy 1: Replace min with better users
                                for new_user in sorted_ucb_indices[:second_min_pos]:
                                    if new_user not in current_state_arr:
                                        new_state = current_state_arr.copy()
                                        new_state[new_state == min_ucb_current] = new_user
                                        all_neighbors.add(tuple(sorted(new_state)))
                                        neighbor_count += 1
                                        
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break
                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                    break
                                
                                # Strategy 2: Replace others with users better than min
                                for replaced_user in current_ucb_sorted[1:]:
                                    for new_user in sorted_ucb_indices[:min_pos]:
                                        if new_user not in current_state_arr:
                                            new_state = current_state_arr.copy()
                                            new_state[new_state == replaced_user] = new_user
                                            all_neighbors.add(tuple(sorted(new_state)))
                                            neighbor_count += 1
                                            
                                            if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                break
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                        else:
                            # Original comprehensive approach with neighbor limit
                            # Active neighbors
                            min_users_union = np.union1d(np.union1d(min_ucb_users, min_g_users), min_p_users)
                            
                            for replaced_user in min_users_union:
                                users_not_in_state = np.setdiff1d(np.arange(num_users), current_state_arr)
                                for new_user in users_not_in_state:
                                    new_state = current_state_arr.copy()
                                    new_state[new_state == replaced_user] = new_user
                                    all_neighbors.add(tuple(sorted(new_state)))
                                    neighbor_count += 1
                                    
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                    break

                            # Passive neighbors for all criteria - only if we haven't hit limit
                            if args.max_neighbors is None or neighbor_count < args.max_neighbors:
                                for criterion, sorted_indices_crit, current_sorted_crit in [
                                    ('ucb', sorted_ucb_indices, current_state_arr[np.argsort(current_ucb)]),
                                    ('g', sorted_g_indices, current_state_arr[np.argsort(current_g)]),
                                    ('p', sorted_p_indices, current_state_arr[np.argsort(current_p)])
                                ]:
                                    min_current = current_sorted_crit[0]
                                    second_min_current = current_sorted_crit[1] if len(current_sorted_crit) > 1 else min_current
                                    
                                    min_pos = np.where(sorted_indices_crit == min_current)[0][0]
                                    second_min_pos = np.where(sorted_indices_crit == second_min_current)[0][0]
                                    
                                    # Strategy 1
                                    for new_user in sorted_indices_crit[:second_min_pos]:
                                        if new_user not in current_state_arr:
                                            new_state = current_state_arr.copy()
                                            new_state[new_state == min_current] = new_user
                                            all_neighbors.add(tuple(sorted(new_state)))
                                            neighbor_count += 1
                                            
                                            if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                break
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                                    
                                    # Strategy 2
                                    for replaced_user in current_sorted_crit[1:]:
                                        for new_user in sorted_indices_crit[:min_pos]:
                                            if new_user not in current_state_arr:
                                                new_state = current_state_arr.copy()
                                                new_state[new_state == replaced_user] = new_user
                                                all_neighbors.add(tuple(sorted(new_state)))
                                                neighbor_count += 1
                                                
                                                if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                                    break
                                        if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                            break
                                    if args.max_neighbors is not None and neighbor_count >= args.max_neighbors:
                                        break
                    
                    # Cache the computed neighbors
                    cached_neighbors = all_neighbors
                    previous_state_key = current_state_key
                    
                    neighbor_computation_time += time.time() - neighbor_start_time
                    neighbor_stats.append(len(all_neighbors))
                    
                else:
                    # State didn't change, use cached neighbors
                    cache_hits += 1
                    all_neighbors = cached_neighbors

                # Rest of the SA algorithm with float64 precision
                current_energy = compute_energy_numpy(tuple(sorted(current_state)), ucb_values, g_values, p_values, args, num_users_per_round)
                energy_list.append(current_energy)
                
                if target_energy is not None and iter % 100 == 0 and current_energy >= target_energy:
                    if args.sa_pause_verbose:
                        print(f"Target energy {target_energy:.6f} reached at iteration {iter}. Stopping early.")
                    winning_comb = current_state
                    best_score = current_energy
                    break
                    
                if len(all_neighbors) == 0:
                    raise RuntimeError(f"No neighbors found at iteration {iter} for method={method}, accelerated={accelerated_sa}")

                if len(all_neighbors) == 0:
                    raise RuntimeError(f"No neighbors found at iteration {iter} for method={method}, accelerated={accelerated_sa}")

                # Choose random neighbor  <-- FIND THIS COMMENT
                neighbors_list = list(all_neighbors)
                next_state_tuple = neighbors_list[np.random.randint(len(neighbors_list))]
                next_state = list(next_state_tuple)
                next_energy = compute_energy_numpy(next_state_tuple, ucb_values, g_values, p_values, args, num_users_per_round)

                if len(next_state) != num_users_per_round:
                    if args.sa_pause_verbose:
                        print(f"Warning: next state size {len(next_state)} != num_users_per_round {num_users_per_round}")
                        print(f"next_state: {next_state}, current_state: {current_state}")
                        print(f"staying at the current state.")
                    continue

                # Choose neighbor (with informed sampling if enabled)
                if args.sa_informed_sampling:
                    # Informed neighbor sampling logic - now works for both UCB-only and full mode
                    neighbors_list = list(all_neighbors)
                    
                    # Pre-compute current state arrays for efficiency
                    current_state_arr = np.array(current_state)
                    current_ucb_arr = ucb_values[current_state_arr]
                    current_g_arr = g_values[current_state_arr]
                    current_p_arr = p_values[current_state_arr]
                    
                    # Calculate energy differences for all neighbors
                    energy_diffs = []
                    for neighbor_tuple in neighbors_list:
                        energy_diff = compute_neighbor_energy_difference(
                            neighbor_tuple, current_state, current_ucb_arr, current_g_arr, current_p_arr,
                            ucb_values, g_values, p_values, args, num_users_per_round
                        )
                        energy_diffs.append(energy_diff)
                    
                    # Apply softmax sampling
                    selected_idx, probabilities, selected_prob = softmax_sampling(energy_diffs, args.sampling_temp)
                    
                    # Select the neighbor based on softmax probabilities
                    next_state_tuple = neighbors_list[selected_idx]
                    next_state = list(next_state_tuple)
                    
                    # For debugging: selected_prob contains the probability of the chosen neighbor
                    
                else:
                    # Original random neighbor selection
                    neighbors_list = list(all_neighbors)
                    next_state_tuple = neighbors_list[np.random.randint(len(neighbors_list))]
                    next_state = list(next_state_tuple)
                    selected_prob = 1.0 / len(neighbors_list)  # Uniform probability for debugging
                
                next_energy = compute_energy_numpy(next_state_tuple, ucb_values, g_values, p_values, args, num_users_per_round)

                if len(next_state) != num_users_per_round:
                    if args.sa_pause_verbose:
                        print(f"Warning: next state size {len(next_state)} != num_users_per_round {num_users_per_round}")
                        print(f"next_state: {next_state}, current_state: {current_state}")
                        print(f"staying at the current state.")
                    continue

                if next_energy >= current_energy + min_improvement_threshold:
                    current_state = next_state
                    if next_energy > best_score:
                        best_score = next_energy
                        winning_comb = next_state
                # Uncomment the following lines if you want to allow minimal improvements
                elif (next_energy - current_energy) > 0 and next_energy < current_energy + min_improvement_threshold:
                    # If it's a minimal improvement we flip a coin to decide
                    if np.random.rand() < 0.5:
                        current_state = next_state
                else:
                    beta_max_f64 = np.float64(beta_max)
                    iter_f64 = np.float64(iter + 1)
                    temp = beta_max_f64 / np.log(iter_f64) if iter > 0 else np.float64('inf')

                    next_energy_f64 = np.float64(next_energy)
                    current_energy_f64 = np.float64(current_energy)
                    energy_diff = next_energy_f64 - current_energy_f64
                    prob = float(np.exp(energy_diff / temp))

                    if np.random.rand() < prob:
                        current_state = next_state

                if (time.time() - start_time > args.max_time_sa_pause) and not args.sa_pause_simulation:
                    textio.cprint(f"sa_pause took more than {args.max_time_sa_pause} seconds to run, breaking at iter: {iter}")
                    break

            # Print final statistics at the end of SA-PAUSE run
            if args.sa_pause_verbose and neighbor_stats:
                neighbor_stats = np.array(neighbor_stats)
                print(f"\nNeighbor Generation Statistics (over {len(neighbor_stats)} state changes):")
                print(f"Min neighbors: {np.min(neighbor_stats)}")
                print(f"Max neighbors: {np.max(neighbor_stats)}")
                print(f"Mean neighbors: {np.mean(neighbor_stats):.2f}")
                print(f"Std neighbors: {np.std(neighbor_stats):.2f}")
                if args.max_neighbors is not None:
                    limit_reached_count = np.sum(neighbor_stats >= args.max_neighbors)
                    print(f"Times limit reached: {limit_reached_count}/{len(neighbor_stats)} ({100*limit_reached_count/len(neighbor_stats):.1f}%)")
                print(f"Total neighbor computation time: {neighbor_computation_time:.2f}s")
                print(f"Cache hit rate: {cache_hits / (cache_hits + cache_misses) * 100:.1f}%")

            # Remove noise only if it was added
            if args.add_tiny_noise:
                for i in range(num_users):
                    local_models[i].ucb -= ucb_noise[i]
                    local_models[i].g -= g_noise[i]
                    local_models[i].privacy_reward -= p_noise[i]

            if not args.sa_pause_simulation:
                if len(set(winning_comb)) != args.num_users_per_round:
                    raise ValueError(f"Winning combination {winning_comb} does not have the correct number of users {args.num_users_per_round}.")
                print(f"sa_pause took {(time.time()-start_time)//60} minutes and {(time.time()-start_time)%60} seconds to run")
                return tuple(winning_comb)
            else:
                print(f"sa_pause took {(time.time()-start_time)//60} minutes and {(time.time()-start_time)%60} seconds to run")
                return energy_list, tuple(winning_comb), best_score

    # [Rest of the methods remain the same...]
    elif method == "pause brute":
        users_idxs_comb = list(itertools.combinations([x for x in range(args.num_users)], args.num_users_per_round))
        np.random.shuffle(users_idxs_comb)
        winning_comb = None
        best_score = 0
        for comb in users_idxs_comb:
            score = compute_energy(comb, local_models, args)
            if score > best_score:
                best_score = score
                winning_comb = comb
        return winning_comb
    
    elif method == "pivot_fill":
        if args.sa_pause_simulation:
            # For simulation mode, pass the num_users and num_users_per_round parameters
            energy_list, winning_comb, best_score = choose_users_pivot_fill(local_models, args, global_epoch, textio, num_users, num_users_per_round)
            return energy_list, winning_comb, best_score
        else:
            # For normal mode, use args values
            winning_comb, best_score = choose_users_pivot_fill(local_models, args, global_epoch, textio)
            if len(set(winning_comb)) != args.num_users_per_round:
                raise ValueError(f"Winning combination {winning_comb} does not have the correct number of users {args.num_users_per_round}.")
            return winning_comb

    elif method == "fraboni":
        users_idxs = []
        for i in range(args.num_users_per_round):
            users_idxs.append(np.random.choice(args.num_users, p=args.fraboni_distributions[i]))
        print(f"idxs of chosen users for epoch No. {global_epoch}: {users_idxs}")
        return tuple(users_idxs)

    elif method == "random":
        return tuple(np.random.choice(args.num_users, args.num_users_per_round, replace=False))

    elif method == "all users":
        ret_val = list(range(args.num_users))
        np.random.shuffle(ret_val)
        return tuple(ret_val)

    elif method == "fastest ones":
        return tuple(range(args.num_users_per_round))

    else:
        raise ValueError(f"There is no such method as {method}, choose a method from:\nsa_pause, pause brute, fraboni, random, all users, fastest ones")
    


def compute_neighbor_energy_difference(neighbor_state, current_state, current_ucb_arr, current_g_arr, current_p_arr,
                                      ucb_values, g_values, p_values, args, num_users_per_round):
    """
    Compute the energy difference between a neighbor state and current state.
    
    Args:
        neighbor_state: Tuple of user indices for the neighbor
        current_state: List of user indices for the current state
        current_ucb_arr: UCB values for current state users
        current_g_arr: G values for current state users  
        current_p_arr: P values for current state users
        ucb_values: All UCB values (numpy array)
        g_values: All G values (numpy array)
        p_values: All P values (numpy array)
        args: Arguments object
        num_users_per_round: Number of users per round
    
    Returns:
        float: Energy difference (neighbor_energy - current_energy)
    """
    neighbor_arr = np.array(neighbor_state)
    neighbor_ucb = ucb_values[neighbor_arr]
    neighbor_g = g_values[neighbor_arr]
    neighbor_p = p_values[neighbor_arr]
    
    # Find the replaced user (in current but not in neighbor)
    current_set = set(current_state)
    neighbor_set = set(neighbor_state)
    
    replaced_user = list(current_set - neighbor_set)[0]
    new_user = list(neighbor_set - current_set)[0]
    
    # Calculate energy difference components
    energy_diff = 0.0
    
    # UCB term: check if new user has lower UCB than current minimum
    current_min_ucb = np.min(current_ucb_arr)
    neighbor_min_ucb = np.min(neighbor_ucb)
    
    if neighbor_min_ucb < current_min_ucb:
        energy_diff += (neighbor_min_ucb - current_min_ucb)
    
    # G and P terms: α/m*(g_new - g_replaced) + γ/m*(p_new - p_replaced)
    alpha_f64 = np.float64(args.alpha)
    gamma_f64 = np.float64(args.gamma) if args.privacy else np.float64(0.0)
    num_users_f64 = np.float64(num_users_per_round)
    
    g_diff = (alpha_f64 / num_users_f64) * (g_values[new_user] - g_values[replaced_user])
    p_diff = (gamma_f64 / num_users_f64) * (p_values[new_user] - p_values[replaced_user])
    
    energy_diff += g_diff + p_diff
    
    return float(energy_diff)


def softmax_sampling(energy_diffs, temperature=1.0):
    """
    Apply softmax sampling to energy differences.
    
    Args:
        energy_diffs: List of energy differences
        temperature: Temperature parameter for softmax
    
    Returns:
        tuple: (selected_index, probabilities_array, selected_probability)
    """
    # Convert to numpy array and apply temperature
    energy_arr = np.array(energy_diffs, dtype=np.float64)
    scaled_energy = energy_arr / temperature
    
    # Apply softmax (subtract max for numerical stability)
    max_energy = np.max(scaled_energy)
    exp_energy = np.exp(scaled_energy - max_energy)
    probabilities = exp_energy / np.sum(exp_energy)
    
    # Sample from the probability distribution
    selected_idx = np.random.choice(len(probabilities), p=probabilities)
    selected_prob = probabilities[selected_idx]
    
    return selected_idx, probabilities, selected_prob



def choose_users_pivot_fill(local_models, args, global_epoch, textio, num_users=None, num_users_per_round=None):
    """
    Optimized Pivot-and-Fill algorithm for user selection.
    
    Args:
        local_models: List of user models
        args: Configuration arguments
        global_epoch: Current global epoch
        textio: IO object for logging
        num_users: Number of users (for simulation mode)
        num_users_per_round: Number of users per round (for simulation mode)
        
    Returns:
        If args.sa_pause_simulation is True: tuple of (energy_value, selected_users, max_energy)
        Otherwise: tuple of (selected_users, max_energy)
    """
    start_time = time.time()
    
    # Use passed parameters if in simulation mode, otherwise use args values
    if args.sa_pause_simulation and num_users is not None and num_users_per_round is not None:
        K = num_users  # Total number of users (candidate pool size)
        m = num_users_per_round  # Number of users to select
    else:
        K = args.num_users  # Total number of users (candidate pool size)
        m = args.num_users_per_round  # Number of users to select
    
    # Handle edge case
    if K < m:
        raise ValueError(f"Cannot select {m} users from {K} available users")
    
    condition = (global_epoch <= (args.pre_sa_pause_rounds * K/m)
                      if not args.sa_pause_simulation else False)
        
    if condition: # Needed because otherwise it will choose not reallt randomly due to implementation + initialization
        round_no = (global_epoch-1) // (K/m)
        list_of_unchosen_users = [i for i in range(K) if local_models[i].num_of_obs == round_no]
        result = tuple(np.random.choice(list_of_unchosen_users, m, replace=False))
        return result, np.inf

    else:
        # Extract UCB, g, and p values for all users
        ucb_values = [local_models[i].ucb for i in range(K)]
        g_values = [local_models[i].g for i in range(K)]
        p_values = [local_models[i].privacy_reward for i in range(K)]
        
        # Sort candidate pool in descending order of UCB
        sorted_indices = sorted(range(K), key=lambda i: ucb_values[i], reverse=True)
        
        # Initialize min-heap H keyed by s(ℓ) = αg_ℓ + γp_ℓ
        # Python's heapq is a min-heap, so we store (s_value, user_index)
        alpha = args.alpha
        gamma = args.gamma if args.privacy else 0
        
        # Step 3: Initialize variables
        S_star = set()
        R_star = float('-inf')
        
        # Compute initial sums for the first m users
        initial_users = sorted_indices[:m]
        G_H = sum(g_values[k] for k in initial_users)
        P_H = sum(p_values[k] for k in initial_users)
        
        # Initialize heap with first m users
        H = []
        for k in initial_users:
            s_k = alpha * g_values[k] + gamma * p_values[k]
            heapq.heappush(H, (s_k, k))
        
        # For simulation mode, track energy at each step (similar to SA-PAUSE)
        energy_list = []
        
        # Main algorithm loop (steps 4-13)
        for i in range(m, K):
            # Step 5: Get current pivot (user with minimum s value)
            current_pivot_s, current_pivot = H[0]  # Peek at min without popping
            
            # Step 6: Current subset S_k
            S_k = {user for _, user in H}
            
            # Step 7: Compute fast reward R_k
            min_ucb_in_S_k = min(ucb_values[k] for k in S_k)
            R_k = min_ucb_in_S_k + (alpha/m) * G_H + (gamma/m) * P_H
            
            # Track energy for simulation mode
            if args.sa_pause_simulation:
                energy_list.append(R_k)
            
            # Step 8-9: Update best solution if better
            if R_k > R_star:
                R_star = R_k
                S_star = S_k.copy()
            
            # Step 10-13: Update partner heap for next pivot if not at end
            if i < K:  # We still have users to process
                next_user = sorted_indices[i]
                next_s = alpha * g_values[next_user] + gamma * p_values[next_user]
                
                # Step 10: Check if we should add the next user
                if next_s > current_pivot_s:
                    # Step 11: Remove user with minimum s value and add next user
                    removed_s, removed_user = heapq.heappop(H)
                    
                    # Step 12: Add next user to heap
                    s_next = alpha * g_values[next_user] + gamma * p_values[next_user]
                    heapq.heappush(H, (s_next, next_user))
                    
                    # Step 13: Update sums
                    G_H = G_H - g_values[removed_user] + g_values[next_user]
                    P_H = P_H - p_values[removed_user] + p_values[next_user]
        
        # Final check for the last configuration
        final_S_k = {user for _, user in H}
        final_min_ucb = min(ucb_values[k] for k in final_S_k)
        final_R_k = final_min_ucb + (alpha/m) * G_H + (gamma/m) * P_H
        
        # Track final energy for simulation mode
        if args.sa_pause_simulation:
            energy_list.append(final_R_k)
        
        if final_R_k > R_star:
            R_star = final_R_k
            S_star = final_S_k.copy()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if args.choosing_users_verbose and textio is not None:
            textio.cprint(f"Pivot-Fill algorithm completed in {elapsed_time:.6f} seconds")
            textio.cprint(f"Selected users: {sorted(S_star)}")
            textio.cprint(f"Maximum energy found: {R_star:.6f}")
        
        # Return format consistent with SA-PAUSE simulation mode
        if args.sa_pause_simulation:
            return energy_list, tuple(sorted(S_star)), R_star
        else:
            return tuple(sorted(S_star)), R_star


def initializations(args):
    """
    Create the relevant folders and perform necessary initializations for the experiment.
    Note: Seed setting is now handled in main.py to ensure consistent seeding across methods.

    Args:
        args: An object containing the experiment arguments.

    Returns:
        boardio: SummaryWriter object for writing TensorBoard logs.
        textio: IOStream object for writing experiment logs.
        best_val_acc: value for the best validation accuracy, initially set to -inf.
        path_best_model: Path to save the best model.

    """
    
    #  documentation
    now = datetime.datetime.now()
    now = str(now.strftime("%d-%m-%Y_%H-%M-%S"))
    base_path = Path.cwd() / 'checkpoints' / args.method_choosing_users / args.model / now
    base_path.mkdir(exist_ok=True, parents=True)
    textio = IOStream(str(base_path) + '/run.log')
    best_val_acc = -np.inf
    path_best_model = base_path  /'best_model.pth.tar'
    last_model_path = base_path  /'last_model.pth.tar'

    return base_path, textio, best_val_acc, path_best_model, last_model_path

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
    """Returns train dataset and test loader with ImageNet-100 and ImageWoof support"""
    
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

    elif args.data == 'imagenet100':
        from datasets import load_dataset
    
        # Download ImageNet-100 from Hugging Face (with cache)
        print("Loading ImageNet-100 from Hugging Face...")
        start_time = time.time()
        
        try:
            dataset = load_dataset("clane9/imagenet-100", cache_dir="./data/hf_cache")
            load_time = time.time() - start_time
            print(f"✓ Dataset loaded in {load_time:.1f}s! Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
        
        # Standard ImageNet transforms
        transform_start = time.time()
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to PyTorch datasets
        print("Creating PyTorch datasets...")
        dataset_start = time.time()
        train_data = HuggingFaceImageDataset(dataset['train'], transform=train_transform)
        test_data = HuggingFaceImageDataset(dataset['validation'], transform=test_transform)
        dataset_time = time.time() - dataset_start
        print(f"✓ PyTorch datasets created in {dataset_time:.1f}s!")
        
        loader_start = time.time()
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=args.test_batch_size, 
            shuffle=False,
            num_workers=2  # Add this for faster loading
        )
        loader_time = time.time() - loader_start
        print(f"✓ Data loader created in {loader_time:.1f}s!")
        
        total_time = time.time() - start_time
        print(f"🕒 Total data setup time: {total_time:.1f}s")

    elif args.data == 'imagewoof':
        import tarfile
        from torchvision.datasets.utils import download_url
        from torchvision.datasets import ImageFolder
        
        # Download ImageWoof dataset
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
        data_dir = './data'
        tgz_path = './imagewoof2-160.tgz'
        extract_path = './data/imagewoof2-160'
        
        # Check if dataset already exists
        if not os.path.exists(extract_path):
            print("Downloading ImageWoof dataset...")
            try:
                # Download if .tgz doesn't exist
                if not os.path.exists(tgz_path):
                    download_url(dataset_url, '.')
                    print("✓ ImageWoof dataset downloaded!")
                
                # Extract the dataset
                print("Extracting ImageWoof dataset...")
                with tarfile.open(tgz_path, 'r:gz') as tar:
                    tar.extractall(path=data_dir)
                print("✓ ImageWoof dataset extracted!")
                
            except Exception as e:
                print(f"Error downloading/extracting ImageWoof: {e}")
                raise
        else:
            print("✓ ImageWoof dataset already exists!")
        
        # ImageNet normalization (ImageWoof is subset of ImageNet)
        # Use Resize with fixed size to ensure all images are exactly 64x64
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Force exact size (64, 64)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Force exact size (64, 64)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_data = ImageFolder(extract_path + '/train', train_transform)
        test_data = ImageFolder(extract_path + '/val', test_transform)
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"✓ ImageWoof loaded! Train: {len(train_data)}, Test: {len(test_data)}")

    elif args.data == 'tiny_imagenet':
        from torchvision.datasets.utils import download_url
        from torchvision.datasets import ImageFolder
        import tarfile
        import zipfile
        import shutil   
        
        # Download Tiny ImageNet dataset  
        dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        data_dir = './data'
        zip_path = './tiny-imagenet-200.zip'
        extract_path = './data/tiny-imagenet-200'
        organized_path = './data/tiny-imagenet-organized'
        
        # Check if organized dataset already exists
        if not os.path.exists(organized_path):
            print("Setting up Tiny ImageNet dataset...")
            
            # Download if zip doesn't exist
            if not os.path.exists(extract_path):
                try:
                    if not os.path.exists(zip_path):
                        print("Downloading Tiny ImageNet dataset...")
                        download_url(dataset_url, '.')
                        print("✓ Tiny ImageNet dataset downloaded!")
                    
                    # Extract the dataset
                    print("Extracting Tiny ImageNet dataset...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print("✓ Tiny ImageNet dataset extracted!")
                    
                except Exception as e:
                    print(f"Error downloading/extracting Tiny ImageNet: {e}")
                    raise
            
            # Reorganize dataset for ImageFolder compatibility
            print("Reorganizing Tiny ImageNet for ImageFolder...")
            os.makedirs(organized_path, exist_ok=True)
            os.makedirs(f"{organized_path}/train", exist_ok=True)
            os.makedirs(f"{organized_path}/val", exist_ok=True)
            
            # Get all available classes from train directory
            train_source = f"{extract_path}/train"
            all_classes = [d for d in os.listdir(train_source) if os.path.isdir(f"{train_source}/{d}")]
            all_classes.sort()  # Ensure consistent ordering
            
            print(f"Found {len(all_classes)} classes in Tiny ImageNet")
            
            # Copy training data (already organized by class)
            for class_dir in all_classes:
                src_train_images = f"{train_source}/{class_dir}/images"
                dst_train_class = f"{organized_path}/train/{class_dir}"
                if os.path.exists(src_train_images):
                    shutil.copytree(src_train_images, dst_train_class)
            
            # Reorganize validation data using val_annotations.txt
            val_source = f"{extract_path}/val"
            val_annotations = f"{val_source}/val_annotations.txt"
            
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            
                            # Create class directory in val if not exists
                            val_class_dir = f"{organized_path}/val/{class_name}"
                            os.makedirs(val_class_dir, exist_ok=True)
                            
                            # Copy image to appropriate class folder
                            src_img = f"{val_source}/images/{img_name}"
                            dst_img = f"{val_class_dir}/{img_name}"
                            if os.path.exists(src_img):
                                shutil.copy2(src_img, dst_img)
            
            print("✓ Tiny ImageNet reorganized!")
        else:
            print("✓ Tiny ImageNet dataset already organized!")
        
        # Standard transforms for 64x64 images
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Load the full dataset first
        full_train_data = ImageFolder(organized_path + '/train', train_transform)
        full_test_data = ImageFolder(organized_path + '/val', test_transform)
        
        # Apply class subsetting if requested
        if args.num_classes_subset is not None:
            print(f"Creating {args.num_classes_subset}-class subset...")
            
            # Get all available classes
            all_classes = full_train_data.classes
            
            if args.num_classes_subset > len(all_classes):
                print(f"Warning: Requested {args.num_classes_subset} classes but only {len(all_classes)} available. Using all classes.")
                selected_classes = all_classes
            else:
                # Randomly select subset of classes (with fixed seed for reproducibility)
                random.seed(args.seed)
                selected_classes = random.sample(all_classes, args.num_classes_subset)
                selected_classes.sort()
            
            print(f"Selected classes: {selected_classes}")
            
            # Create mapping from old class indices to new class indices
            old_to_new_class_map = {full_train_data.class_to_idx[cls]: i 
                                for i, cls in enumerate(selected_classes)}
            
            # Filter training data
            train_indices = []
            for idx, (_, class_idx) in enumerate(full_train_data.samples):
                if class_idx in old_to_new_class_map:
                    train_indices.append(idx)
            

            print("Debug: Checking test data structure...")
            print(f"Test data classes: {len(full_test_data.classes)}")
            print(f"First 10 test samples: {full_test_data.samples[:10]}")
            print(f"Test class_to_idx: {list(full_test_data.class_to_idx.items())[:10]}")

            # Check which classes actually exist in validation set
            actual_test_classes = set()
            for _, class_idx in full_test_data.samples:
                actual_test_classes.add(class_idx)
            print(f"All classes present in validation set: {sorted(actual_test_classes)}")
            print(f"Number of classes in validation: {len(actual_test_classes)}")


            # Filter test data
            test_indices = []
            class_sample_count = {}
            for idx, (path, class_idx) in enumerate(full_test_data.samples):
                if class_idx in old_to_new_class_map:
                    test_indices.append(idx)
                    class_sample_count[class_idx] = class_sample_count.get(class_idx, 0) + 1

            print(f"Debug: Samples per class in validation: {class_sample_count}")
            print(f"Debug: Classes with 0 samples: {[cls for cls in old_to_new_class_map.keys() if cls not in class_sample_count]}")
                        
            print(f"Debug: Original train samples: {len(full_train_data.samples)}")
            print(f"Debug: Filtered train indices: {len(train_indices)}")
            print(f"Debug: Original test samples: {len(full_test_data.samples)}")  
            print(f"Debug: Filtered test indices: {len(test_indices)}")



            # Create subset datasets
            train_data = torch.utils.data.Subset(full_train_data, train_indices)
            test_data = torch.utils.data.Subset(full_test_data, test_indices)
            
            # Create a custom dataset wrapper to handle class remapping
            class ClassSubsetDataset(torch.utils.data.Dataset):
                def __init__(self, subset_dataset, old_to_new_map):
                    self.subset_dataset = subset_dataset
                    self.old_to_new_map = old_to_new_map
                    
                def __len__(self):
                    return len(self.subset_dataset)
                
                def __getitem__(self, idx):
                    image, old_class_idx = self.subset_dataset[idx]
                    new_class_idx = self.old_to_new_map[old_class_idx]
                    return image, new_class_idx
            
            # Apply class remapping
            train_data = ClassSubsetDataset(train_data, old_to_new_class_map)
            test_data = ClassSubsetDataset(test_data, old_to_new_class_map)
            


            # Add classes attribute for compatibility
            train_data.classes = selected_classes
            
            print(f"✓ {len(selected_classes)}-class subset created!")
        else:
            train_data = full_train_data
            test_data = full_test_data
            print("✓ Using all 200 classes!")
        
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=2
        )
        
        num_classes_actual = len(train_data.classes)
        print(f"✓ Tiny ImageNet loaded! Train: {len(train_data)}, Test: {len(test_data)}, Classes: {num_classes_actual}")

    else:  # cifar10
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

def data_arrangement(data, args):
    """Updated to handle ImageNet-100, ImageWoof, and Tiny ImageNet with truncation support"""
    train_data = data
    
    # Apply data truncation if specified
    if args.data_truncation is not None:
        truncation_size = int(args.data_truncation)
        print(f"Truncating dataset from {len(train_data)} to {truncation_size} samples")
        train_data = torch.utils.data.Subset(train_data, range(truncation_size))

    # input, output sizes
    if args.data == 'imagenet100':
        input_var = 3  # RGB channels for MobileNetV2
        output = 100   # ImageNet-100 classes
    elif args.data == 'imagewoof':
        input_var = 3  # RGB channels
        output = 10    # ImageWoof classes (10 dog breeds)
    elif args.data == 'tiny_imagenet':
        input_var = 3  # RGB channels
        # FIXED: Get the actual number of classes being used
        if hasattr(train_data, 'classes'):
            output = len(train_data.classes)
        else:
            # For subset datasets, get from the arguments
            output = args.num_classes_subset if args.num_classes_subset is not None else 200
        print(f"Using {output} classes for Tiny ImageNet")
    else:
        in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
        input_var = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
        output = len(data.classes)  # number of classes

    return input_var, output, train_data

def plot_graphs(paths_dict: dict, x_axis_time = True, path_to_save = None, print_graph = True):
    """
    Plots graphs for validation loss, validation accuracy, and average train loss over time or epochs.
    
    Args:
        paths_dict (dict): A dictionary containing the paths to the data for each graph.
        x_axis_time (bool, optional): Determines whether the x-axis represents time or epochs. 
                                      Defaults to True (time).
    """
    paths_dict_copy = paths_dict.copy()

    for key, value in paths_dict_copy.items():
        #check if the value is an absolute path or a relative path
        if not value.is_absolute():
            paths_dict_copy[key] = torch.load(Path.cwd() / value, map_location=(torch.device('cuda') if 
                                                                            torch.cuda.is_available() else
                                                                                torch.device('cpu')), weights_only = False)
        else:
            paths_dict_copy[key] = torch.load(value, map_location=(torch.device('cuda') if 
                                                                            torch.cuda.is_available() else
                                                                                torch.device('cpu')), weights_only = False)

    colors_list = ["C0", "orange", "green", "indigo", "olive", "brown", "pink", "gray", "red", "purple"]
    line_styles = ["-", "--", "-."]
    
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    max_acc =  0
    min_acc =100

    for idx, zipped_key_value in enumerate(paths_dict_copy.items()):
        key, value = zipped_key_value
        x_var = value["global_epochs_time_list"] if x_axis_time else list(range(1, value["global_epoch"]+1))
        ax[0,0].stairs(value["privacy_violations_list"],edges=[0] + x_var, baseline=None,
                       label = f"{key} privacy violation", ls = line_styles[idx%len(line_styles)], color = colors_list[idx])
        ax[0,1].plot(x_var, value['val_acc_list'], label = f"{key} validation accuracy",
                     ls = line_styles[idx%len(line_styles)], color = colors_list[idx])
        ax[1,0].plot(x_var, value['train_loss_list'], label = f"{key} avg train loss",
                     ls = line_styles[idx%len(line_styles)], color = colors_list[idx])
        ax[1,1].plot(x_var, value['val_losses_list'], label = f"{key} validation loss",
                     ls = line_styles[idx%len(line_styles)], color = colors_list[idx])
        
        if max(value['val_acc_list']) > max_acc:
            max_acc = max(value['val_acc_list'])
        if min(value['val_acc_list']) < min_acc:
            min_acc = min(value['val_acc_list'])
        


    ax[0,0].set_title(r"privacy violation over time (max $\epsilon$ budget exceeded)") if x_axis_time else ax[0,0].set_title("privacy violation over epochs")
    ax[0,0].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[0,0].set_xlabel("epochs", fontsize=10)
    ax[0,0].set_ylabel("privacy violation")
    ax[0,0].legend(fontsize=7)

    ax[0,1].set_title("val acc over time") if x_axis_time else ax[0,1].set_title("val acc over epochs")
    ax[0,1].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[0,1].set_xlabel("epochs", fontsize=10)
    ax[0,1].set_ylabel("val acc")
    ax[0,1].set_yticks(list(np.arange(0,int(max_acc) + 5,5)))
    ax[0,1].set_ylim(min_acc-1, max_acc + 1)
    ax[0,1].legend(fontsize=7)

    ax[1,0].set_title("avg train loss over time") if x_axis_time else ax[1,0].set_title("avg train loss over epochs")
    ax[1,0].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[1,0].set_xlabel("epochs", fontsize=10)
    ax[1,0].set_ylabel("train loss")
    ax[1,0].legend(fontsize=7)
    ax[1,0].set_ylim(0,3)

    ax[1,1].set_title("val loss over time") if x_axis_time else ax[1,1].set_title("val loss over epochs")
    ax[1,1].set_xlabel("time(sec)", fontsize=10) if x_axis_time else ax[1,1].set_xlabel("epochs", fontsize=10)
    ax[1,1].set_ylabel("val loss")
    ax[1,1].legend(fontsize=7)
    ax[1,1].set_ylim(0,3)

    fig.suptitle("Metrics over time" if x_axis_time else "Metrics over epochs", fontsize=20)
    fig.tight_layout()
    if path_to_save is not None:
        # save in the path_to_save folder under the name f"metrics over {'time' if x_axis_time else 'epochs'}.png"
        fig.savefig(path_to_save / f"metrics over {'time' if x_axis_time else 'epochs'}.png")
    if print_graph:
        plt.show()


def plot_graphs_for_paper(father_path, graph = "accuracy", x_axis_time = True, moving_average = None,
 x_axis_trunc=None):

    father_path = Path(father_path)
    paths_dict = {}
    # for every subdir add the path to the dict
    for subdir in father_path.iterdir():
        if subdir.is_dir():
            if subdir.name == "fraboni":
                name = "Clustered Sampling"
                line_dict = {"color": '#FF1493', "ls": ":", "lw": 2}
            elif subdir.name == "sa_pause":
                name = "SA-PAUSE"
                color = "C0"
                ls = "-"
                line_dict = {"color": "C0", "ls": "-", "lw": 2}
            elif subdir.name == "pause brute":
                name = "PAUSE"
                color = "green"
                ls = "--"
                line_dict = {"color": "green", "ls": "--", "lw": 2}
            elif subdir.name == "random":
                name = "Random"
                color = "indigo"
                ls = "-."
                line_dict = {"color": "indigo", "ls": "-.", "lw": 2}
            elif subdir.name == "fastest ones":
                name = "Fastest in expectation"
                color = '#FF4500'
                ls = "-"
                line_dict = {"color": '#FF4500', "ls": "-.", "lw": 1}
            elif subdir.name == "all users":
                name = "FedAvg with privacy"
                color = "gray"
                ls = "-."
                line_dict = {"color": "gray", "ls": "-.", "lw": 1}
            elif subdir.name == "all users - no privacy":
                name = "FedAvg w.o. privacy"
                color = "green"
                ls = "--"
                line_dict = {"color": "green", "ls": "-", "lw": 1}
            elif subdir.name == "pivot_fill" or subdir.name == "pivot fill":
                name = "PAUSE"
                color = "green"
                ls = "-"
                line_dict = {"color": "green", "ls": "--", "lw": 2}

            paths_dict[name] = [subdir / "last_model.pth.tar", line_dict]

    paths_dict_copy = paths_dict.copy()

    for key, value in paths_dict_copy.items():
        line_dict = value[1]
        value = value[0]
        #check if the value is an absolute path or a relative path
        if not value.is_absolute():
            paths_dict_copy[key] = [torch.load(Path.cwd() / value, map_location=(torch.device('cuda') if 
                                                                            torch.cuda.is_available() else
                                                                                torch.device('cpu')), weights_only=False)]
            paths_dict_copy[key].append(line_dict)
        else:
            paths_dict_copy[key] = [torch.load(value, map_location=(torch.device('cuda') if 
                                                                            torch.cuda.is_available() else
                                                                                torch.device('cpu')),weights_only=False)]
            paths_dict_copy[key].append(line_dict)


    
    fig, ax = plt.subplots(1,1)
    max_acc =  0
    min_acc = 100


    for idx, zipped_key_value in enumerate(paths_dict_copy.items()):
        key, value = zipped_key_value
        line_dict = value[1]
        value = value[0]
        x_var = value["global_epochs_time_list"] if x_axis_time else list(range(1, value["global_epoch"]+1))
        if graph == "accuracy":
            # if moving average is not None,, plot the moving average of the validation accuracy in casual manner.
            # make sure to have at least 10 epochs before starting the moving average
            if moving_average is not None:
                value["val_acc_list"] = custom_moving_average(value["val_acc_list"], window_size=moving_average)
            ax.plot(x_var, value['val_acc_list'], label = f"{key}",
                ls = line_dict["ls"], color = line_dict["color"], lw = line_dict["lw"])
            
        elif graph == "privacy":
            if "no privacy" in key.lower() or "w.o. privacy" in key.lower():
                continue
            ax.stairs(value["privacy_violations_list"],edges=[0] + x_var, baseline=None,
                       label = f"{key}", ls = line_dict["ls"], color = line_dict["color"], lw = line_dict["lw"])
        
        if max(value['val_acc_list']) > max_acc:
            max_acc = max(value['val_acc_list'])
        if min(value['val_acc_list']) < min_acc:
            min_acc = min(value['val_acc_list'])

    if graph == "accuracy":
        ax.set_xlabel("Time [sec]", fontsize=10) if x_axis_time else ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("Validation accuracy [%]")
        ax.set_yticks(list(np.arange(0,int(max_acc) + 5,5)))
        ax.set_ylim(min_acc-1, max_acc + 1)
        if x_axis_trunc is not None:
            ax.set_xlim(0,x_axis_trunc)

    elif graph == "privacy":
        ax.set_xlabel("Time [sec]", fontsize=10) if x_axis_time else ax.set_xlabel("Epochs", fontsize=10)
        ax.set_ylabel("System's maximum privacy violation")
        if x_axis_trunc is not None:
            ax.set_xlim(0,x_axis_trunc)
    
    ax.legend(fontsize=10)

    fig.tight_layout()
    plt.show()

def custom_moving_average(series, window_size=10):
    """
    Apply a moving average to a series with:
    - Progressive window sizes (1, 2, ..., window_size) at the beginning
    - Full window_size for all elements with complete overlap ('valid' convolution)
    - Returns array of same length as input
    
    Parameters:
    series (array-like): Input data series
    window_size (int): Size of the moving average window, defaults to 10
    
    Returns:
    numpy.ndarray: Series with the custom moving average applied, same length as input
    """
    if window_size is None or window_size <= 0:
        print("No smoothing applied, returning original series.")
        return np.asarray(series, dtype=float)

    series = np.asarray(series)
    result = np.zeros_like(series, dtype=float)
    
    # Handle the first window_size-1 elements with increasing window sizes
    for i in range(window_size - 1):
        current_window = i + 1
        window = np.ones(current_window) / current_window
        result[i] = np.sum(series[:current_window] * window)
    
    window = np.ones(window_size) / window_size
    valid_convolution = np.convolve(series, window, mode='valid')

    result[window_size-1:] = valid_convolution
    
    return result

def users_partition(global_model, train_data: torch.utils.data.Dataset, args, i_i_d=True, verbose=True):
    """
    Sets up the federated learning environment by creating local models for each user.
    
    Args:
        global_model (torch.nn.Module): The global model to be used as a starting point for each local model.
        train_data (torch.utils.data.Dataset or torch.utils.data.subset): The training dataset.
        args: Additional arguments for configuring the federated setup.
        i_i_d (bool): If True, distribute data uniformly. If False, create non-IID distribution.
        verbose (bool): Whether to print distribution statistics and show visualization. Defaults to True.
    
    Returns:
        dict: A dictionary containing the local models for each user.
    """
    local_models = {}
    
    if i_i_d:
        # Original IID distribution code
        print("Creating IID data distribution...")
        indexes = torch.randperm(len(train_data))
        user_data_len = math.floor(len(train_data) / args.num_users)
        
        print(f"Creating {args.num_users} users with ~{user_data_len} samples each...")
        for user_idx in range(args.num_users):
            if user_idx % 50 == 0:
                print(f"  Creating user {user_idx}/{args.num_users}")
            
            user_indices = indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]
            user_dict = create_user_dict(global_model, train_data, user_indices, user_idx, args)
            local_models[user_idx] = create_user(user_dict, args, user_idx)
    else:
        # Non-IID distribution with primary labels and varying dataset sizes
        print("Creating non-IID data distribution...")
        
        # Get to the base dataset if train_data is a Subset
        base_dataset = train_data
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        # Handle different dataset types
        if args.data == 'imagenet100':
            num_classes = 100
        elif hasattr(base_dataset, 'classes'):
            num_classes = len(base_dataset.classes)
        else:
            # Optimized fallback: fast method for HuggingFace datasets
            if hasattr(train_data, 'hf_dataset'):
                print("Determining number of classes from HuggingFace dataset...")
                sample_labels = [train_data.hf_dataset[i]['label'] for i in range(min(100, len(train_data.hf_dataset)))]
                num_classes = max(sample_labels) + 1
                print(f"✓ Found {num_classes} classes")
            else:
                print("Determining number of classes from dataset samples...")
                num_classes = len(set([train_data[i][1] for i in range(min(1000, len(train_data)))]))
                print(f"✓ Found {num_classes} classes")
        
        print(f"Using {num_classes} classes for non-IID distribution")
        
        # Get indices by label
        indexes_by_label = get_indices_by_label(train_data)
        
        # Print initial data distribution
        print("\nInitial data available per label:")
        for label, indices in indexes_by_label.items():
            print(f"Label {label}: {len(indices)} samples")
        
        # Generate data size distribution
        print(f"Generating data size distribution (Dirichlet coefficient: {args.dirichlet_coeff})...")
        size_proportions = np.random.dirichlet(alpha=[args.dirichlet_coeff] * args.num_users)
        total_samples = len(train_data)
        user_data_sizes = [max(total_samples // (args.num_users * 3), 
                             int(p * total_samples)) for p in size_proportions]
        user_data_sizes = [int(size * total_samples / sum(user_data_sizes)) 
                          for size in user_data_sizes]
        
        # Assign primary labels
        print("Assigning primary labels to users...")
        primary_labels = []
        users_per_label = math.ceil(args.num_users / num_classes)
        for label in range(num_classes):
            primary_labels.extend([label] * users_per_label)
        primary_labels = primary_labels[:args.num_users]
        random.shuffle(primary_labels)
        
        # Create dataset for each user
        print(f"Creating datasets for {args.num_users} users...")
        import time
        start_time = time.time()
        
        for user_idx in range(args.num_users):
            # Add progress indicator every 50 users
            if user_idx % 50 == 0 and user_idx > 0:
                elapsed = time.time() - start_time
                rate = user_idx / elapsed
                eta = (args.num_users - user_idx) / rate
                print(f"  Progress: {user_idx}/{args.num_users} users ({rate:.0f} users/sec, ETA: {eta:.0f}s)")
            
            primary_label = primary_labels[user_idx]
            target_size = user_data_sizes[user_idx]
            
            # Calculate exact sizes for primary and other labels
            primary_size = int(args.label_dominance * target_size)
            other_size = target_size - primary_size
            
            # Add primary label data
            primary_indices = indexes_by_label[primary_label]
            if len(primary_indices) < primary_size:
                primary_size = len(primary_indices)
                other_size = target_size - primary_size
            
            # Select primary label samples
            selected_primary = np.random.choice(primary_indices, primary_size, replace=False)
            user_indices = list(selected_primary)
            
            # Update available primary label indices
            indexes_by_label[primary_label] = np.setdiff1d(primary_indices, selected_primary)
            
            # Select other label samples
            other_labels = [l for l in range(num_classes) if l != primary_label]
            samples_per_other = other_size // max(1, len(other_labels))
            
            for label in other_labels:
                available = indexes_by_label[label]
                size = min(samples_per_other, len(available))
                if size > 0:
                    selected = np.random.choice(available, size, replace=False)
                    user_indices.extend(selected)
                    indexes_by_label[label] = np.setdiff1d(available, selected)
            
            # Create user dictionary and model
            user_dict = create_user_dict(global_model, train_data, user_indices, user_idx, args)
            local_models[user_idx] = create_user(user_dict, args, user_idx)

        elapsed_total = time.time() - start_time
        print(f"✓ All {args.num_users} users created in {elapsed_total:.1f}s")
    
    if verbose:
        print("Generating distribution statistics...")
        print_distribution_stats(local_models, args)
        print("Creating distribution visualization...")
        fig, ax = visualize_user_data_distribution(local_models, args)
        # save this figure in args.exp_path / "data_distribution.png"
        path_to_save = args.exp_path / "data_distribution.png"
        fig.savefig(path_to_save)
        plt.close(fig)
        print(f"✓ Distribution plot saved to {path_to_save}")
            
    return local_models

def get_indices_by_label(dataset):
    """Helper function to separate dataset indices by label - optimized for HuggingFace and ImageFolder datasets."""
    indices_by_label = {}
    
    # Handle HuggingFace datasets differently to avoid loading images
    if hasattr(dataset, 'hf_dataset'):
        print("Extracting labels from HuggingFace dataset (fast method)...")
        # Direct access to labels without loading images
        all_labels = [dataset.hf_dataset[i]['label'] for i in range(len(dataset.hf_dataset))]
        num_classes = max(all_labels) + 1
    elif hasattr(dataset, 'samples'):
        # ImageFolder datasets (ImageWoof, Tiny ImageNet, etc.) - very fast!
        print("Extracting labels from ImageFolder dataset (fast method)...")
        all_labels = [dataset.samples[i][1] for i in range(len(dataset.samples))]  # Direct access to labels
        num_classes = len(dataset.classes)
        print(f"✓ Found {num_classes} classes: {dataset.classes}")
    elif hasattr(dataset, 'classes') and hasattr(dataset, 'targets'):
        # CIFAR10, MNIST, FashionMNIST datasets - also fast!
        print("Extracting labels from torchvision dataset (fast method)...")
        all_labels = dataset.targets
        num_classes = len(dataset.classes)
        print(f"✓ Found {num_classes} classes: {dataset.classes}")
    else:
        # Fallback method for other datasets
        print("Using fallback method for label extraction...")
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        else:
            # Sample a subset to determine number of classes
            sample_size = min(1000, len(dataset))
            all_labels = [dataset[i][1] for i in range(sample_size)]
            num_classes = len(set(all_labels))
            # Get all labels
            all_labels = [dataset[i][1] for i in range(len(dataset))]
    
    # Initialize empty lists for each label
    for i in range(num_classes):
        indices_by_label[i] = []
    
    # Collect indices for each label
    for idx, label in enumerate(all_labels):
        indices_by_label[label].append(idx)
    
    # Convert lists to numpy arrays
    for label in indices_by_label:
        indices_by_label[label] = np.array(indices_by_label[label])
    
    print(f"✓ Extracted labels for {len(all_labels)} samples across {num_classes} classes")
    return indices_by_label

def create_user_dict(global_model, train_data, indices, user_idx, args):
    """Helper function to create user dictionary with model and optimizer."""
    # FIX: Convert numpy indices to Python ints
    if isinstance(indices, np.ndarray):
        indices = [int(i) for i in indices]
    elif isinstance(indices, list):
        indices = [int(i) if isinstance(i, np.integer) else i for i in indices]
    
    user_dict = {
        'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, indices),
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0  # FIX: Avoid multiprocessing issues
        ),
        'model': copy.deepcopy(global_model)
    }
    
    user_dict['opt'] = (
        optim.SGD(user_dict['model'].parameters(), lr=args.lr, momentum=args.momentum)
        if args.optimizer == 'sgd'
        else optim.Adam(user_dict['model'].parameters(), lr=args.lr)
    )
    
    if args.lr_scheduler:
        user_dict['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
            user_dict['opt'], patience=10, factor=0.1, verbose=False)
    
    return user_dict

def create_user(user_dict, args, user_idx):
    """Helper function to create user instance."""
    return user(
        args,
        user_idx,
        user_dict['data'],
        user_dict['model'],
        user_dict['opt'],
        user_dict['scheduler'] if args.lr_scheduler else None
    )
def visualize_user_data_distribution(local_models, args):
    """
    Creates a stacked bar chart showing the distribution of labels across users.
    Optimized for HuggingFace datasets and ImageFolder datasets to avoid loading images.
    """
    
    # Get number of classes
    if args.data == 'imagenet100':
        num_classes = 100
    elif args.data == 'imagewoof':
        num_classes = 10
    elif args.data == 'tiny_imagenet':
        num_classes = 200
    else:
        first_user_data = next(iter(local_models.values())).data_loader.dataset
        while hasattr(first_user_data, 'dataset'):
            first_user_data = first_user_data.dataset
        num_classes = len(first_user_data.classes)
    
    # Initialize array to store counts: [num_users × num_classes]
    label_counts = np.zeros((args.num_users, num_classes))
    
    # Count samples per label for each user
    for user_idx in range(args.num_users):
        dataset = local_models[user_idx].data_loader.dataset
        
        # Handle different dataset types to get labels efficiently
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'hf_dataset'):
            # This is a Subset of HuggingFaceImageDataset
            subset_indices = dataset.indices
            base_dataset = dataset.dataset.hf_dataset
            labels = [base_dataset[int(idx)]['label'] for idx in subset_indices]
        elif hasattr(dataset, 'hf_dataset'):
            # This is a HuggingFaceImageDataset
            labels = [dataset.hf_dataset[i]['label'] for i in range(len(dataset.hf_dataset))]
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
            # This is a Subset of ImageFolder (ImageWoof case)
            subset_indices = dataset.indices
            base_samples = dataset.dataset.samples
            labels = [base_samples[int(idx)][1] for idx in subset_indices]
        elif hasattr(dataset, 'samples'):
            # This is an ImageFolder dataset directly
            labels = [dataset.samples[i][1] for i in range(len(dataset.samples))]
        else:
            # This is a regular torchvision dataset (fallback)
            labels = [dataset[i][1] for i in range(len(dataset))]
        
        unique, counts = np.unique(labels, return_counts=True)
        label_counts[user_idx, unique] = counts
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Create bars
    bottom = np.zeros(args.num_users)
    bars = []
    for label in range(num_classes):
        bar = ax.bar(range(args.num_users), label_counts[:, label], 
                    bottom=bottom, label=f'Label {label}')
        bars.append(bar)
        bottom += label_counts[:, label]
    
    # Calculate and display dominant label percentage for each user
    for user_idx in range(args.num_users):
        total = sum(label_counts[user_idx, :])
        if total > 0:
            dominant_label_pct = (np.max(label_counts[user_idx, :]) / total) * 100
            if args.num_users <= 100:
                ax.text(user_idx, total, f'{int(total)}\n({dominant_label_pct:.1f}%)', 
                   ha='center', va='bottom')
    
    # sum label counts in both axes to get total samples
    total_samples = int(label_counts.sum())

    # Customize plot
    ax.set_title(f'Distribution of Labels Across Users\n(Total samples & Dominant label percentage)\nTotal No. of samples: {total_samples}', 
                fontsize=14, pad=20)
    ax.set_xlabel('User Index', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.legend(title='Labels', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ensure all bars are visible
    plt.tight_layout()
    
    return fig, ax


def print_distribution_stats(local_models, args):
    """
    Prints statistics about the data distribution.
    
    Args:
        local_models (dict): Dictionary of user models from federated_setup
        args: Arguments containing num_users and other parameters
    """
    total_samples = []
    for user_idx in range(args.num_users):
        dataset = local_models[user_idx].data_loader.dataset
        total_samples.append(len(dataset))
    
    print(f"Data Distribution Statistics:")
    print(f"Average samples per user: {np.mean(total_samples):.1f}")
    print(f"Std of samples per user: {np.std(total_samples):.1f}")
    print(f"Min samples: {np.min(total_samples)}")
    print(f"Max samples: {np.max(total_samples)}")
    print(f"Total samples: {np.sum(total_samples)}")
    

def plot_layered_user_selections(choices_table, save_path, args, interval=30, figsize=(18.5, 10.5)):
    """
    Creates a stacked bar plot showing the number of times each user was chosen,
    with different colors for each interval of epochs.
    
    Args:
        choices_table (np.ndarray): Binary matrix where rows are epochs and columns are users
        interval (int): Number of epochs per layer
        figsize (tuple): Figure size in inches
    """
    num_epochs, num_users = choices_table.shape
    cumulative_counts = np.cumsum(choices_table, axis=0)
    
    # Calculate the number of complete intervals
    num_intervals = num_epochs // interval
    if num_epochs % interval > 0:
        num_intervals += 1
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create x-axis labels
    users_idxs = [str(x) for x in range(1, args.num_users + 1)]
    
    # Define hatching patterns for different layers
    # These patterns work well for distinguishing layers in grayscale
    hatches = ['/', 'x', '+', '*', 'o', 'O', '.', '-', '|']
    
    # Plot each layer
    bottom = np.zeros(args.num_users)
    for i in range(num_intervals):
        start_idx = i * interval
        end_idx = min((i + 1) * interval, num_epochs)
        
        if i == 0:
            layer_data = cumulative_counts[end_idx - 1]
        else:
            layer_data = cumulative_counts[end_idx - 1] - cumulative_counts[start_idx - 1]
        
        # Use both color and hatching pattern for each layer
        plt.bar(users_idxs, layer_data, bottom=bottom, 
               label=f'Epochs {start_idx + 1}-{end_idx}',
               alpha=0.7,
               hatch=hatches[i % len(hatches)],  # Cycle through hatching patterns
               edgecolor='black')
        
        bottom += layer_data
    
    plt.title("Cumulative Number of Times Each User Was Chosen")
    plt.xlabel("User Index")
    plt.ylabel("Number of times")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if len(users_idxs) >= 40:
        plt.xticks([])
    else:
        plt.xticks(rotation=45, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
