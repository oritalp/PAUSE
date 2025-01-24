import numpy as np
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

import utils

def partition_users(global_model, train_data: torch.utils.data.Dataset, args, i_i_d=True, dirichlet_coeff=5,
                    label_dominance=0.4, verbose=True):
    """
    Sets up the federated learning environment by creating local models for each user.
    
    Args:
        global_model (torch.nn.Module): The global model to be used as a starting point for each local model.
        train_data (torch.utils.data.Dataset or torch.utils.data.subset): The training dataset.
        args: Additional arguments for configuring the federated setup.
        i_i_d (bool): If True, distribute data uniformly. If False, create non-IID distribution.
        dirichlet_coeff (float): The coefficient for the Dirichlet distribution, the lower the value the higher the probability to
        produce probability vector with lower entropy. Defaults to 5.
        label_dominance (float): The proportion of data from the primary label for each user in non-IID setting. Defaults to 0.4.
        verbose (bool): Whether to print distribution statistics and show visualization. Defaults to True.
    
    Returns:
        dict: A dictionary containing the local models for each user.
    """
    local_models = {}
    
    if i_i_d:
        # Original IID distribution code
        indexes = torch.randperm(len(train_data))
        user_data_len = math.floor(len(train_data) / args.num_users)
        for user_idx in range(args.num_users):
            user_indices = indexes[user_idx * user_data_len:(user_idx + 1) * user_data_len]
            user_dict = create_user_dict(global_model, train_data, user_indices, user_idx, args)
            local_models[user_idx] = create_user(user_dict, args, user_idx)
    else:
        # Non-IID distribution with primary labels and varying dataset sizes
        # Get to the base dataset if train_data is a Subset
        base_dataset = train_data
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset
        
        num_classes = len(base_dataset.classes)
        indexes_by_label = get_indices_by_label(train_data)
        
        # Print initial data distribution
        print("\nInitial data available per label:")
        for label, indices in indexes_by_label.items():
            print(f"Label {label}: {len(indices)} samples")
        
        # Generate data size distribution
        size_proportions = np.random.dirichlet(alpha=[dirichlet_coeff] * args.num_users)
        total_samples = len(train_data)
        user_data_sizes = [max(total_samples // (args.num_users * 3), 
                             int(p * total_samples)) for p in size_proportions]
        user_data_sizes = [int(size * total_samples / sum(user_data_sizes)) 
                          for size in user_data_sizes]
        
        # Assign primary labels
        primary_labels = []
        users_per_label = math.ceil(args.num_users / num_classes)
        for label in range(num_classes):
            primary_labels.extend([label] * users_per_label)
        primary_labels = primary_labels[:args.num_users]
        random.shuffle(primary_labels)
        
        # Create dataset for each user
        for user_idx in range(args.num_users):
            primary_label = primary_labels[user_idx]
            target_size = user_data_sizes[user_idx]
            
            # Calculate exact sizes for primary and other labels
            primary_size = int(label_dominance * target_size)
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
            samples_per_other = other_size // (num_classes - 1)
            
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
    
    if verbose:
        print_distribution_stats(local_models, args)
        fig, ax = visualize_user_data_distribution(local_models, args)
        plt.show()
            
    return local_models



def get_indices_by_label(dataset):
    """Helper function to separate dataset indices by label."""
    indices_by_label = {}
    
    # Initialize empty lists for each label
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    else:
        # If classes not available, find from data
        all_labels = [dataset[i][1] for i in range(len(dataset))]
        num_classes = len(set(all_labels))
    
    for i in range(num_classes):
        indices_by_label[i] = []
    
    # Go through dataset and collect indices for each label
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices_by_label[label].append(idx)
    
    # Convert lists to numpy arrays
    for label in indices_by_label:
        indices_by_label[label] = np.array(indices_by_label[label])
    
    return indices_by_label

def create_user_dict(global_model, train_data, indices, user_idx, args):
    """Helper function to create user dictionary with model and optimizer."""
    user_dict = {
        'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data, indices),
            batch_size=args.train_batch_size,
            shuffle=True
        ),
        'model': copy.deepcopy(global_model)
    }
    
    # Set up optimizer
    user_dict['opt'] = (
        optim.SGD(user_dict['model'].parameters(), lr=args.lr, momentum=args.momentum)
        if args.optimizer == 'sgd'
        else optim.Adam(user_dict['model'].parameters(), lr=args.lr)
    )
    
    # Set up learning rate scheduler if requested
    if args.lr_scheduler:
        user_dict['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(
            user_dict['opt'],
            patience=10,
            factor=0.1,
            verbose=True
        )
    
    return user_dict


def create_user(user_dict, args, user_idx):
    """Helper function to create user instance."""
    return utils.user(
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
    Each bar represents a user, with stacks showing the number of samples per label.
    Also shows the percentage of the dominant label for each user.
    
    Args:
        local_models (dict): Dictionary of user models from federated_setup
        args: Arguments containing num_users and other parameters
    """
    
    # Get number of classes from first user's dataset
    first_user_data = next(iter(local_models.values())).data_loader.dataset
    while hasattr(first_user_data, 'dataset'):
        first_user_data = first_user_data.dataset
    num_classes = len(first_user_data.classes)
    
    # Initialize array to store counts: [num_users × num_classes]
    label_counts = np.zeros((args.num_users, num_classes))
    
    # Count samples per label for each user
    for user_idx in range(args.num_users):
        dataset = local_models[user_idx].data_loader.dataset
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
            ax.text(user_idx, total, f'{int(total)}\n({dominant_label_pct:.1f}%)', 
                   ha='center', va='bottom')
    
    # Customize plot
    ax.set_title('Distribution of Labels Across Users\n(Total samples & Dominant label percentage)', 
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
    