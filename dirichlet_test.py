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


def run_dirichlet_test(dirichlet_values):
    """
    Runs SA-PAUSE experiments with different Dirichlet coefficient values to analyze the effect of data heterogeneity.
    
    Args:
        dirichlet_values (list): List of Dirichlet coefficient values to test
    """
    # Parse base arguments
    args = args_parser()
    
    # Force method to be sa_pause for this test
    args.method_choosing_users = "sa_pause"
    
    # Create timestamp for this test run
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create base results directory
    base_results_path = Path.cwd() / 'dirichlet_test_results' / start_time
    base_results_path.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create experiment identifier string (without dirichlet_coeff since that varies)
    dirichlet_values_str = '_'.join([str(dv) for dv in dirichlet_values])
    indicating_str = f"_alpha={args.alpha}_gamma={args.gamma}_eps_bar={args.epsilon_bar}_eta={args.epsilon_sum_deascent_coeff}_dirichlet_values={dirichlet_values_str}"
    
    print(f"Starting Dirichlet coefficient test with the following configuration:")
    print(f"Dirichlet coefficient values to test: {dirichlet_values}")
    print(f"Base configuration: {indicating_str}")
    print(f"Results will be saved in: {base_results_path}")
    
    # Dictionary to store results paths for comparison
    results_dict = {}
    data_sizes_dict = {}
    
    # Initialize wandb project for dirichlet testing if enabled
    wandb_project_name = f"DIRICHLET_TEST_PAUSE-{indicating_str}-{start_time}"
    
    # Run experiment for each Dirichlet coefficient value
    for dirichlet_coeff in dirichlet_values:
        print(f"\n{'='*50}")
        print(f"Running experiment with Dirichlet coefficient = {dirichlet_coeff}")
        print(f"{'='*50}")
        
        # Create a copy of args and set the specific Dirichlet coefficient
        dirichlet_args = copy.deepcopy(args)
        dirichlet_args.dirichlet_coeff = dirichlet_coeff
        
        # Initialize wandb for this specific Dirichlet coefficient run
        if dirichlet_args.wandb:
            # Ensure previous run is finished
            if wandb.run is not None:
                wandb.finish()
            
            wandb.init(
                project=wandb_project_name,
                name=f"dirichlet_{dirichlet_coeff}",
                group=f"dirichlet_test_{start_time}",
                config={
                    "exp_ID": indicating_str,
                    "model": dirichlet_args.model,
                    "num_users": dirichlet_args.num_users,
                    "num_users_per_round": dirichlet_args.num_users_per_round,
                    "global_epochs": dirichlet_args.global_epochs,
                    "max_seconds": dirichlet_args.max_seconds,
                    "epsilon_bar": dirichlet_args.epsilon_bar,
                    "epsilon_sum_deascent_coeff": dirichlet_args.epsilon_sum_deascent_coeff,
                    "delta_f": dirichlet_args.delta_f,
                    "data": dirichlet_args.data,
                    "alpha": dirichlet_args.alpha,
                    "gamma": dirichlet_args.gamma,
                    "dirichlet_coeff": dirichlet_coeff,  # This is the parameter we're testing
                    "accel_ucb_coeff": dirichlet_args.accel_ucb_coeff,
                    "method": dirichlet_args.method_choosing_users
                }
            )
        
        # Run the experiment
        experiment_path = Path(run_exp(dirichlet_args))
        
        # Copy results to Dirichlet test directory
        dirichlet_result_path = base_results_path / f"dirichlet_{dirichlet_coeff}"
        experiment_path_relative = experiment_path.relative_to(Path.cwd())
        shutil.copytree(experiment_path_relative, dirichlet_result_path)
        
        # Store path for later comparison
        results_dict[f"α={dirichlet_coeff}"] = dirichlet_result_path / "last_model.pth.tar"
        
        # Extract data sizes for distribution comparison
        # We'll need to reload the experiment and extract user data sizes
        # For now, we'll store the path and extract this later
        
        print(f"Completed experiment with Dirichlet coefficient = {dirichlet_coeff}")
        print(f"Results saved in: {dirichlet_result_path}")
        
        # Clean up memory
        gc.collect()
    
    # Finish wandb if it was used
    if args.wandb and wandb.run is not None:
        wandb.finish()
    
    print(f"\n{'='*50}")
    print("All Dirichlet coefficient experiments completed!")
    print("Generating comparison plots...")
    print(f"{'='*50}")
    
    # Generate comparison plots
    generate_dirichlet_comparison_plots(results_dict, base_results_path, dirichlet_values)
    generate_data_distribution_comparison(base_results_path, dirichlet_values, args)
    
    print(f"\nDirichlet coefficient test completed successfully!")
    print(f"All results saved in: {base_results_path}")
    
    return base_results_path


def generate_dirichlet_comparison_plots(results_dict, save_path, dirichlet_values):
    """
    Generate comparison plots showing how different metrics vary with Dirichlet coefficient values.
    """
    # Load all results
    loaded_results = {}
    for key, path in results_dict.items():
        loaded_results[key] = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define colors for different Dirichlet coefficient values
    colors = plt.cm.viridis(np.linspace(0, 1, len(dirichlet_values)))
    
    # Plot 1: Validation Accuracy over Time
    ax = axes[0]
    for i, (key, data) in enumerate(loaded_results.items()):
        ax.plot(data["global_epochs_time_list"], data["val_acc_list"], 
               label=key, color=colors[i], linewidth=2)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Validation Accuracy [%]")
    ax.set_title("Validation Accuracy vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy over Epochs
    ax = axes[1]
    for i, (key, data) in enumerate(loaded_results.items()):
        epochs = list(range(1, len(data["val_acc_list"]) + 1))
        ax.plot(epochs, data["val_acc_list"], 
               label=key, color=colors[i], linewidth=2)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Accuracy [%]")
    ax.set_title("Validation Accuracy vs Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss over Time
    ax = axes[2]
    for i, (key, data) in enumerate(loaded_results.items()):
        ax.plot(data["global_epochs_time_list"], data["train_loss_list"], 
               label=key, color=colors[i], linewidth=2)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Privacy Violations over Time
    ax = axes[3]
    for i, (key, data) in enumerate(loaded_results.items()):
        x_time = [0] + data["global_epochs_time_list"]
        ax.stairs(data["privacy_violations_list"], edges=x_time, 
                 label=key, color=colors[i], linewidth=2)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Maximum Privacy Violation")
    ax.set_title("Privacy Violations vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Final Accuracy vs Dirichlet Coefficient
    ax = axes[4]
    final_accuracies = []
    dirichlet_coeffs = []
    for key, data in loaded_results.items():
        dirichlet_val = float(key.split('=')[1])
        dirichlet_coeffs.append(dirichlet_val)
        final_accuracies.append(data["val_acc_list"][-1])
    
    # Sort by Dirichlet coefficient value for proper line plot
    sorted_pairs = sorted(zip(dirichlet_coeffs, final_accuracies))
    dirichlet_sorted, accuracies_sorted = zip(*sorted_pairs)
    
    ax.plot(dirichlet_sorted, accuracies_sorted, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Dirichlet Coefficient (α)")
    ax.set_ylabel("Final Validation Accuracy [%]")
    ax.set_title("Final Accuracy vs Dirichlet Coefficient")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dirichlet_values)
    
    # Plot 6: Convergence Time vs Dirichlet Coefficient (time to reach 90% of final accuracy)
    ax = axes[5]
    convergence_times = []
    for key, data in loaded_results.items():
        dirichlet_val = float(key.split('=')[1])
        final_acc = data["val_acc_list"][-1]
        target_acc = 0.9 * final_acc
        
        # Find first time we reach 90% of final accuracy
        convergence_time = None
        for i, acc in enumerate(data["val_acc_list"]):
            if acc >= target_acc:
                convergence_time = data["global_epochs_time_list"][i]
                break
        
        if convergence_time is None:
            convergence_time = data["global_epochs_time_list"][-1]  # If never reached, use final time
        
        convergence_times.append((dirichlet_val, convergence_time))
    
    # Sort by Dirichlet coefficient value
    convergence_times.sort()
    dirichlet_conv, times_conv = zip(*convergence_times)
    
    ax.plot(dirichlet_conv, times_conv, 's-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel("Dirichlet Coefficient (α)")
    ax.set_ylabel("Time to 90% Final Accuracy [sec]")
    ax.set_title("Convergence Time vs Dirichlet Coefficient")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dirichlet_values)
    
    plt.tight_layout()
    plt.savefig(save_path / "dirichlet_comparison_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "dirichlet_comparison_plots.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved in: {save_path}")


def generate_data_distribution_comparison(save_path, dirichlet_values, args):
    """
    Generate mean ± variance plot comparison of data distribution sizes across different Dirichlet coefficient values.
    This focuses on outliers and variance to show data heterogeneity effects.
    """
    # Calculate mean and variance for each Dirichlet coefficient
    means = []
    stds = []
    labels = []
    all_data_sizes = []
    
    for dirichlet_coeff in dirichlet_values:
        # Set random seed to ensure reproducibility for this specific coefficient
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        
        if args.i_i_d:
            # For IID, all users get the same amount of data regardless of Dirichlet coefficient
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
        
        means.append(np.mean(data_sizes))
        stds.append(np.std(data_sizes))
        labels.append(f'α={dirichlet_coeff}')
        all_data_sizes.append(data_sizes)
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean ± Standard Deviation
    colors = plt.cm.viridis(np.linspace(0, 1, len(dirichlet_values)))
    x_pos = np.arange(len(dirichlet_values))
    
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Dataset Size per User')
    ax1.set_xlabel('Dirichlet Coefficient (α)')
    ax1.set_title('Mean Dataset Size ± Standard Deviation\n(Error bars show data heterogeneity)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'α={d}' for d in dirichlet_values])
    ax1.grid(True, alpha=0.3)
    
    # Add variance values as text on bars
    for i, (bar, std) in enumerate(zip(bars, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 10,
                f'σ={std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Coefficient of Variation (CV = std/mean) to show relative heterogeneity
    cv_values = [std/mean for mean, std in zip(means, stds)]
    
    ax2.plot(dirichlet_values, cv_values, 'o-', linewidth=3, markersize=8, color='red')
    ax2.set_ylabel('Coefficient of Variation (σ/μ)')
    ax2.set_xlabel('Dirichlet Coefficient (α)')
    ax2.set_title('Data Heterogeneity Index\n(Higher CV = More Heterogeneous)')
    ax2.set_xticks(dirichlet_values)
    ax2.grid(True, alpha=0.3)
    
    # Add CV values as text
    for i, (d, cv) in enumerate(zip(dirichlet_values, cv_values)):
        ax2.text(d, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add interpretation text
    interpretation_text = ("Lower α → Higher variance → More heterogeneous\n"
                          "Higher α → Lower variance → More homogeneous\n"
                          "CV quantifies relative heterogeneity")
    fig.text(0.02, 0.02, interpretation_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for interpretation text
    plt.savefig(save_path / "data_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(save_path / "data_distribution_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Data distribution comparison saved in: {save_path}")
    
    # Print summary statistics
    print("\nData Distribution Summary:")
    for i, d in enumerate(dirichlet_values):
        print(f"α={d}: Mean={means[i]:.1f}, Std={stds[i]:.1f}, CV={cv_values[i]:.3f}")
    


def main(dirichlet_values):
    """
    Main function to run the Dirichlet coefficient test.
    
    Args:
        dirichlet_values (list, optional): List of Dirichlet coefficient values to test.
                                         Defaults to [1, 2, 3, 4, 5, 6, 7] if None.
    """
    
    print("Starting Dirichlet Coefficient Test for SA-PAUSE Algorithm")
    print("=" * 60)
    print(f"Testing Dirichlet coefficient values: {dirichlet_values}")
    
    try:
        results_path = run_dirichlet_test(dirichlet_values)
        print(f"\nDirichlet coefficient test completed successfully!")
        print(f"Results directory: {results_path}")
        
    except Exception as e:
        print(f"Error during Dirichlet coefficient test: {str(e)}")
        raise


if __name__ == '__main__':
    main(dirichlet_values=[0.1,0.25,0.5,1,2,3,4,5])