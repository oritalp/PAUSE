import gc
import sys
from statistics import mean
import time
import torch
from tqdm import tqdm
import numpy as np
import itertools
from pathlib import Path
import scipy
import matplotlib.pyplot as plt
import datetime
import torch.linalg as LA
from torch.distributions.laplace import Laplace
import wandb
import shutil
import copy

import utils
from configurations import args_parser
from run_exp import run_exp


import gc
import sys
from statistics import mean
import time
import torch
from tqdm import tqdm
import numpy as np
import itertools
from pathlib import Path
import scipy
import matplotlib.pyplot as plt
import datetime
import torch.linalg as LA
from torch.distributions.laplace import Laplace
import wandb
import shutil
import copy

import utils
from configurations import args_parser
from run_exp import run_exp


def calculate_cv_for_alpha(dirichlet_coeff, args):
    """
    Calculate coefficient of variation for a specific Dirichlet coefficient value.
    Recalculates the data distribution using the same logic as the experiments.
    """
    # Set random seed to match the experiment
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    
    if args.i_i_d:
        # For IID, all users get the same amount of data
        train_data, _ = utils.data(args)
        input_var, output, train_data = utils.data_arrangement(train_data, args)
        user_data_len = len(train_data) // args.num_users
        data_sizes = [user_data_len] * args.num_users
    else:
        # For non-IID, generate sizes using the specific Dirichlet coefficient
        train_data, _ = utils.data(args)
        input_var, output, train_data = utils.data_arrangement(train_data, args)
        size_proportions = np.random.dirichlet(alpha=[dirichlet_coeff] * args.num_users)
        total_samples = len(train_data)
        user_data_sizes = [max(total_samples // (args.num_users * 3), 
                             int(p * total_samples)) for p in size_proportions]
        data_sizes = [int(size * total_samples / sum(user_data_sizes)) 
                     for size in user_data_sizes]
    
    mean_size = np.mean(data_sizes)
    std_size = np.std(data_sizes)
    cv = std_size / mean_size if mean_size > 0 else 0
    
    return cv


def plot_dirichlet_results_detached(results_dir, alpha_values_to_plot, graph="val_accuracy_time", 
                                   moving_average=None, save_path=None):
    """
    Plot results for specific alpha values after experiments are completed.
    Follows the style of plot_graphs_for_paper from utils.py.
    
    Args:
        results_dir (str or Path): Path to the dirichlet test results directory
        alpha_values_to_plot (list): List of alpha values to include in the plot
        graph (str): Type of graph to plot. Options:
            - "val_accuracy_time": Validation accuracy vs time
            - "val_accuracy_epochs": Validation accuracy vs epochs  
            - "train_loss_time": Training loss vs time
            - "privacy_violations": Privacy violations vs time
            - "final_accuracy": Final accuracy vs alpha
            - "convergence_time": Convergence time vs alpha
        moving_average (int, optional): Window size for moving average
        save_path (str or Path, optional): Where to save the plot. If None, uses results_dir
    """
    results_dir = Path(results_dir)
    if save_path is None:
        save_path = results_dir
    else:
        save_path = Path(save_path)
    
    # Load results for specified alpha values
    loaded_results = {}
    cv_values = {}
    
    # Parse args to calculate CV values
    args = args_parser()
    
    for alpha_val in alpha_values_to_plot:
        alpha_path = results_dir / f"dirichlet_{alpha_val}" / "last_model.pth.tar"
        if alpha_path.exists():
            loaded_results[alpha_val] = torch.load(alpha_path, map_location=torch.device('cpu'), weights_only=False)
            cv_values[alpha_val] = calculate_cv_for_alpha(alpha_val, args)
        else:
            print(f"Warning: Results for α={alpha_val} not found at {alpha_path}")
    
    if not loaded_results:
        print("No valid results found for the specified alpha values.")
        return
    
    # Create line style dictionary (following plot_graphs_for_paper style)
    colors = plt.cm.jet(np.linspace(0, 1, len(loaded_results)))
    line_styles = ["-", "--", "-.", ":"]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot based on graph type
    if graph == "val_accuracy_time":
        for idx, (alpha_val, data) in enumerate(loaded_results.items()):
            cv = cv_values[alpha_val]
            label = f"α={alpha_val} (CV={cv:.3f})"
            y_data = data['val_acc_list']
            
            if moving_average is not None:
                y_data = utils.custom_moving_average(y_data, window_size=moving_average)
            
            ax.plot(data["global_epochs_time_list"], y_data, 
                   label=label, color=colors[idx], 
                   linestyle=line_styles[idx % len(line_styles)], linewidth=2.5)
        
        ax.set_xlabel("Time [sec]", fontsize=18)
        ax.set_ylabel("Validation Accuracy [%]", fontsize=18)
        # ax.set_title("Validation Accuracy vs Time", fontsize=14)
        filename = "validation_accuracy_vs_time"
        
    elif graph == "val_accuracy_epochs":
        for idx, (alpha_val, data) in enumerate(loaded_results.items()):
            cv = cv_values[alpha_val]
            label = f"α={alpha_val} (CV={cv:.3f})"
            epochs = list(range(1, len(data["val_acc_list"]) + 1))
            y_data = data['val_acc_list']
            
            if moving_average is not None:
                y_data = utils.custom_moving_average(y_data, window_size=moving_average)
            
            ax.plot(epochs, y_data, 
                   label=label, color=colors[idx], 
                   linestyle=line_styles[idx % len(line_styles)], linewidth=2)
        
        ax.set_xlabel("Epochs", fontsize=16)
        ax.set_ylabel("Validation Accuracy [%]", fontsize=16)
        # ax.set_title("Validation Accuracy vs Epochs", fontsize=14)
        filename = "validation_accuracy_vs_epochs"
        
    elif graph == "train_loss_time":
        for idx, (alpha_val, data) in enumerate(loaded_results.items()):
            cv = cv_values[alpha_val]
            label = f"α={alpha_val} (CV={cv:.3f})"
            y_data = data['train_loss_list']
            
            if moving_average is not None:
                y_data = utils.custom_moving_average(y_data, window_size=moving_average)
            
            ax.plot(data["global_epochs_time_list"], y_data, 
                   label=label, color=colors[idx], 
                   linestyle=line_styles[idx % len(line_styles)], linewidth=2)
        
        ax.set_xlabel("Time [sec]", fontsize=16)
        ax.set_ylabel("Training Loss", fontsize=16)
        # ax.set_title("Training Loss vs Time", fontsize=14)
        filename = "training_loss_vs_time"
        
    elif graph == "privacy_violations":
        for idx, (alpha_val, data) in enumerate(loaded_results.items()):
            cv = cv_values[alpha_val]
            label = f"α={alpha_val} (CV={cv:.3f})"
            
            # # Clip the last 10 data points to avoid edge artifacts
            # privacy_data = data["privacy_violations_list"][:-10] if len(data["privacy_violations_list"]) > 10 else data["privacy_violations_list"]
            # time_data = data["global_epochs_time_list"][:-10] if len(data["global_epochs_time_list"]) > 10 else data["global_epochs_time_list"]

            privacy_data = data["privacy_violations_list"]
            time_data = data["global_epochs_time_list"]
            
            # For stairs(), edges must have exactly len(privacy_data) + 1 elements
            # Ensure proper alignment: [0, time1, time2, ..., timeN] for N privacy values
            if len(time_data) == len(privacy_data):
                x_time = [0] + time_data
            else:
                # If there's still a mismatch, truncate to ensure proper alignment
                min_len = min(len(time_data), len(privacy_data))
                privacy_data = privacy_data[:min_len]
                x_time = [0] + time_data[:min_len]
            
            ax.stairs(privacy_data, edges=x_time, 
                     label=label, color=colors[idx], 
                     linestyle=line_styles[idx % len(line_styles)], linewidth=2, baseline=None)
        
        ax.set_xlabel("Time [sec]", fontsize=18)
        ax.set_ylabel("Maximum Privacy Violation", fontsize=18)
        # ax.set_title("Privacy Violations vs Time", fontsize=14)
        filename = "privacy_violations_vs_time"
        
    elif graph == "final_accuracy":
        final_accuracies = []
        alphas = []
        labels = []
        
        for alpha_val, data in loaded_results.items():
            cv = cv_values[alpha_val]
            alphas.append(alpha_val)
            final_accuracies.append(data["val_acc_list"][-1])
            labels.append(f"α={alpha_val} (CV={cv:.3f})")
        
        # Sort by alpha value for proper line plot
        sorted_data = sorted(zip(alphas, final_accuracies, labels))
        alphas_sorted, accuracies_sorted, labels_sorted = zip(*sorted_data)
        
        ax.plot(alphas_sorted, accuracies_sorted, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel(r"Dirichlet Coefficient ($\boldsymbol{\alpha}$)", fontsize=16)
        ax.set_ylabel("Final Validation Accuracy [%]", fontsize=16)
        # ax.set_title("Final Accuracy vs Dirichlet Coefficient", fontsize=14)
        ax.set_xticks(alphas_sorted)
        
        # Add CV values as text annotations
        for alpha, acc, label in zip(alphas_sorted, accuracies_sorted, labels_sorted):
            cv = cv_values[alpha]
            ax.annotate(f'CV={cv:.3f}', (alpha, acc), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        filename = "final_accuracy_vs_alpha"
        
    elif graph == "convergence_time":
        convergence_times = []
        alphas = []
        labels = []
        
        for alpha_val, data in loaded_results.items():
            cv = cv_values[alpha_val]
            final_acc = data["val_acc_list"][-1]
            target_acc = 0.9 * final_acc
            
            # Find first time we reach 90% of final accuracy
            convergence_time = None
            for i, acc in enumerate(data["val_acc_list"]):
                if acc >= target_acc:
                    convergence_time = data["global_epochs_time_list"][i]
                    break
            
            if convergence_time is None:
                convergence_time = data["global_epochs_time_list"][-1]
            
            alphas.append(alpha_val)
            convergence_times.append(convergence_time)
            labels.append(f"α={alpha_val} (CV={cv:.3f})")
        
        # Sort by alpha value
        sorted_data = sorted(zip(alphas, convergence_times, labels))
        alphas_sorted, times_sorted, labels_sorted = zip(*sorted_data)
        
        ax.plot(alphas_sorted, times_sorted, 's-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel(r"Dirichlet Coefficient ($\boldsymbol{\alpha}$)", fontsize=16)
        ax.set_ylabel("Time to 90% Final Accuracy [sec]", fontsize=16)
        # ax.set_title("Convergence Time vs Dirichlet Coefficient", fontsize=14)
        ax.set_xticks(alphas_sorted)
        
        # Add CV values as text annotations
        for alpha, time, label in zip(alphas_sorted, times_sorted, labels_sorted):
            cv = cv_values[alpha]
            ax.annotate(f'CV={cv:.3f}', (alpha, time), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
        
        filename = "convergence_time_vs_alpha"
        
    else:
        raise ValueError(f"Unknown graph type: {graph}")
    
    # Add legend and grid (except for final_accuracy and convergence_time which have annotations)
    if graph not in ["final_accuracy", "convergence_time"]:
        ax.legend(fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {save_path / filename}")


def plot_heterogeneity_index_detached(results_dir, alpha_values_to_plot, save_path=None):
    """
    Plot only the data heterogeneity index (CV) for specified alpha values.
    
    Args:
        results_dir (str or Path): Path to the dirichlet test results directory
        alpha_values_to_plot (list): List of alpha values to include in the plot
        save_path (str or Path, optional): Where to save the plot. If None, uses results_dir
    """
    results_dir = Path(results_dir)
    if save_path is None:
        save_path = results_dir
    else:
        save_path = Path(save_path)
    
    # Parse args to calculate CV values
    args = args_parser()
    
    # Calculate CV values for specified alphas
    cv_values = []
    valid_alphas = []
    
    for alpha_val in alpha_values_to_plot:
        alpha_path = results_dir / f"dirichlet_{alpha_val}" / "last_model.pth.tar"
        if alpha_path.exists():
            cv = calculate_cv_for_alpha(alpha_val, args)
            cv_values.append(cv)
            valid_alphas.append(alpha_val)
        else:
            print(f"Warning: Results for α={alpha_val} not found at {alpha_path}")
    
    if not cv_values:
        print("No valid results found for the specified alpha values.")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(valid_alphas, cv_values, 'o-', linewidth=3, markersize=8, color='red')
    ax.set_ylabel('Coefficient of Variation (σ/μ)', fontsize=16)
    ax.set_xlabel(r'Dirichlet Coefficient ($\boldsymbol{\alpha}$)', fontsize=16)
    ax.set_title('Data Heterogeneity Index\n(Higher CV = More Heterogeneous)', fontsize=14)
    ax.set_xticks(valid_alphas)
    ax.grid(True, alpha=0.3)
    
    # Add CV values as text
    for alpha, cv in zip(valid_alphas, cv_values):
        ax.text(alpha, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / "data_heterogeneity_index.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "data_heterogeneity_index.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Heterogeneity index plot saved: {save_path / 'data_heterogeneity_index'}")


if __name__ == "__main__":
    # Example usage
    results_dir = Path("dirichlet_test_results/2025-06-23_17-29-34")
    alpha_values = [0.25, 0.5, 1, 2]
    
    # Plot validation accuracy vs time
    plot_dirichlet_results_detached(results_dir, alpha_values, graph="val_accuracy_time", 
                                    moving_average=20)
    
    # Plot data heterogeneity index
    plot_heterogeneity_index_detached(results_dir, alpha_values)