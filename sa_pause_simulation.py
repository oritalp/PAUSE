import gc
import sys
from statistics import mean
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime
import pickle
import os
from pathlib import Path

import utils
from configurations import args_parser

def single_run_comparison(num_users=300, num_users_per_round=15, 
                          brute_force_comparison=True, pivot_fill_comparison=True,
                          use_dummy_data=True, real_data_path=None):
    """
    Updated comparison code with support for real data loading.
    
    Args:
        num_users: Number of users (ignored if using real data)
        num_users_per_round: Number of users per round (ignored if using real data)
        brute_force_comparison: Whether to run brute force comparison
        pivot_fill_comparison: Whether to run pivot fill comparison
        use_dummy_data: If True, generate dummy data. If False, load real data.
        real_data_path: Path to real data pickle file (required if use_dummy_data=False)
    
    Returns:
        Comparison results
    """
    args = args_parser()
    args.beta_max_reduction = 70 
    args.max_iterations_sa_pause = 4000  
    args.sa_pause_accelerated = False
    args.ucb_neighbors_only = False
    args.sa_informed_sampling = False
    args.sampling_temp = 10**-2
    args.add_tiny_noise = False  # Disable tiny noise addition for simulation
    args.sa_pause_simulation = True  # Set to True to run in simulation mode
    args.sa_pause_verbose = True  # Enable verbose output for simulation
    
    # Clean up the memory
    gc.collect()

    local_models = {}

    class user():
        def __init__(self, ucb, g, p, args):
            self.ucb = ucb
            self.g = g
            self.privacy_reward = p
            self.args = args
            self.user_idx = None

    if use_dummy_data:
        # Original dummy data generation
        print("Using dummy data generation...")
        base_arange = np.arange(num_users)/10
        base_array = np.concatenate([np.random.permutation(base_arange).reshape(-1,1) for i in range(3)], axis=1)

        for row in range(base_array.shape[0]):
            #create a user object with the ucb, g, and p values
            u = user(base_array[row,0], base_array[row,1], base_array[row,2], args)
            u.user_idx = row
            local_models[row] = u
            
        # Use the provided parameters
        actual_num_users = num_users
        actual_num_users_per_round = num_users_per_round
        
    else:
        # Load real data
        if real_data_path is None:
            raise ValueError("real_data_path must be provided when use_dummy_data=False")
        
        print(f"Loading real data from {real_data_path}...")
        
        # Load the saved data directly (each file contains one epoch's data)
        if not Path(real_data_path).exists():
            raise FileNotFoundError(f"Data file not found: {real_data_path}")
            
        with open(real_data_path, 'rb') as f:
            epoch_data = pickle.load(f)
        
        actual_num_users = len(epoch_data)
        assert actual_num_users == num_users, \
            f"Expected {num_users} users but found {actual_num_users} in the real data file."
        
        # Determine num_users_per_round based on typical federated learning settings
        # You might want to adjust this based on your specific use case
        actual_num_users_per_round = num_users_per_round
        
        print(f"Real data loaded: {actual_num_users} users, using {actual_num_users_per_round} users per round")
        
        # Create user objects from real data
        for _, user_data in enumerate(epoch_data):
            user_idx, ucb, g, p = user_data
            u = user(float(ucb), float(g), float(p), args)
            u.user_idx = int(user_idx)
            local_models[user_idx] = u
        
        # Update args for the real data scenario
        num_users = actual_num_users
        num_users_per_round = actual_num_users_per_round

    print(f"Running comparison with {actual_num_users} users, {actual_num_users_per_round} users per round")

    # Brute Force comparison (if enabled and feasible)
    if brute_force_comparison and actual_num_users <= 50:  # Only run if feasible
        print("Running Brute Force comparison...")
        start_time_bf = time.time()
        
        users_idxs_comb = list(itertools.combinations([x for x in range(actual_num_users)], actual_num_users_per_round))
        # permute the users_idxs_comb to make the order of the users random
        np.random.shuffle(users_idxs_comb)
        winning_comb_bf = None
        best_score_BF = 0
        for comb in users_idxs_comb:
            score = utils.compute_energy(comb, local_models, args, num_users=actual_num_users, num_users_per_round=actual_num_users_per_round)
            if score > best_score_BF:
                best_score_BF = score
                winning_comb_bf = comb
        
        end_time_bf = time.time()
        bf_runtime = end_time_bf - start_time_bf

        print(f"Brute Force winning_comb: {winning_comb_bf}")
        print(f"Brute Force best score: {best_score_BF}")
        print(f"Brute Force runtime: {bf_runtime:.6f} seconds")
    elif brute_force_comparison:
        print(f"Skipping Brute Force comparison (too many users: {actual_num_users})")
        brute_force_comparison = False

    # Pivot-Fill comparison (if enabled)
    if pivot_fill_comparison:
        print("Running Pivot-Fill comparison...")
        start_time_pf = time.time()
        
        # For simulation, we need to handle the return value differently
        # The pivot_fill method returns (selected_users, best_energy) when not in simulation mode
        # But we want to compare energies, so we'll call it directly
        _, winning_comb_pf, best_score_PF = utils.choose_users_pivot_fill(local_models, args, global_epoch=1,
                                                                        textio=None, num_users=actual_num_users,
                                                                        num_users_per_round=actual_num_users_per_round)
        
        end_time_pf = time.time()
        pf_runtime = end_time_pf - start_time_pf

        print(f"Pivot-Fill winning_comb: {winning_comb_pf}")
        print(f"Pivot-Fill best score: {best_score_PF}")
        print(f"Pivot-Fill runtime: {pf_runtime:.6f} seconds")

    # SA-PAUSE comparison
    print("Running SA-PAUSE comparison...")
    start_time_sa = time.time()

    energy_list, winning_comb_sa, best_score_SA = utils.choose_users(local_models, args, global_epoch=1, textio=None,
                                                                    method="sa_pause", num_users=actual_num_users, 
                                                                    num_users_per_round=actual_num_users_per_round)

    end_time_sa = time.time()
    sa_runtime = end_time_sa - start_time_sa

    print(f"SA-PAUSE winning_comb: {winning_comb_sa}")
    print(f"SA-PAUSE best score: {best_score_SA}")
    print(f"SA-PAUSE runtime: {sa_runtime:.6f} seconds")

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SA-PAUSE energy evolution
    ax.plot(energy_list, label='SA-PAUSE Energy Evolution', alpha=0.7)

    # Add horizontal lines for comparison
    if brute_force_comparison:
        ax.axhline(y=best_score_BF, color='orange', linestyle='-.', 
                   label=f'Brute Force Best Score ({best_score_BF:.6f})')

    if pivot_fill_comparison:
        ax.axhline(y=best_score_PF, color='green', linestyle='--',
                   label=f'Pivot-Fill Best Score ({best_score_PF:.6f})')

    ax.axhline(y=best_score_SA, color='magenta', linestyle=':',
               label=f'SA-PAUSE Best Score ({best_score_SA:.6f})')

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Energy')
    
    data_type = "Real Data" if not use_dummy_data else "Dummy Data"
    ax.set_title(f'Algorithm Comparison: Energy vs Iterations ({data_type})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"algorithm_comparison_{'real' if not use_dummy_data else 'dummy'}.png", dpi=300)
    plt.show()

    # Print summary comparison
    print("\n" + "="*60)
    print(f"ALGORITHM COMPARISON SUMMARY ({data_type})")
    print("="*60)

    if brute_force_comparison:
        print(f"Brute Force     - Score: {best_score_BF:.6f}, Runtime: {bf_runtime:.6f}s")

    if pivot_fill_comparison:
        print(f"Pivot-Fill      - Score: {best_score_PF:.6f}, Runtime: {pf_runtime:.6f}s")

    print(f"SA-PAUSE        - Score: {best_score_SA:.6f}, Runtime: {sa_runtime:.6f}s")

    # Calculate performance ratios if we have multiple methods
    print("\n" + "-"*60)
    print("PERFORMANCE ANALYSIS")
    print("-"*60)

    if pivot_fill_comparison:
        pf_sa_score_ratio = best_score_PF / best_score_SA if best_score_SA != 0 else float('inf')
        pf_sa_time_ratio = pf_runtime / sa_runtime if sa_runtime != 0 else float('inf')
        
        print(f"Pivot-Fill vs SA-PAUSE:")
        print(f"  Score ratio: {pf_sa_score_ratio:.4f} (PF/SA)")
        print(f"  Time ratio:  {pf_sa_time_ratio:.4f} (PF/SA)")
        
        if pf_sa_score_ratio >= 0.99:  # Within 1% of optimal
            print(f"  → Pivot-Fill achieves {pf_sa_score_ratio*100:.2f}% of SA-PAUSE score")
        else:
            print(f"  → Pivot-Fill achieves {pf_sa_score_ratio*100:.2f}% of SA-PAUSE score")
        
        if pf_sa_time_ratio < 1:
            print(f"  → Pivot-Fill is {1/pf_sa_time_ratio:.2f}x faster than SA-PAUSE")
        else:
            print(f"  → Pivot-Fill is {pf_sa_time_ratio:.2f}x slower than SA-PAUSE")

    if brute_force_comparison:
        bf_sa_score_ratio = best_score_BF / best_score_SA if best_score_SA != 0 else float('inf')
        bf_sa_time_ratio = bf_runtime / sa_runtime if sa_runtime != 0 else float('inf')
        
        print(f"\nBrute Force vs SA-PAUSE:")
        print(f"  Score ratio: {bf_sa_score_ratio:.4f} (BF/SA)")
        print(f"  Time ratio:  {bf_sa_time_ratio:.4f} (BF/SA)")
        
        if pivot_fill_comparison:
            bf_pf_score_ratio = best_score_BF / best_score_PF if best_score_PF != 0 else float('inf')
            bf_pf_time_ratio = bf_runtime / pf_runtime if pf_runtime != 0 else float('inf')
            
            print(f"\nBrute Force vs Pivot-Fill:")
            print(f"  Score ratio: {bf_pf_score_ratio:.4f} (BF/PF)")
            print(f"  Time ratio:  {bf_pf_time_ratio:.4f} (BF/PF)")

    print("\n" + "="*60)

    # Return results for further analysis
    results = {
        'data_type': data_type,
        'num_users': actual_num_users,
        'num_users_per_round': actual_num_users_per_round,
        'sa_pause': {
            'score': best_score_SA,
            'runtime': sa_runtime,
            'solution': winning_comb_sa,
            'energy_evolution': energy_list
        }
    }
    
    if pivot_fill_comparison:
        results['pivot_fill'] = {
            'score': best_score_PF,
            'runtime': pf_runtime,
            'solution': winning_comb_pf
        }
    
    if brute_force_comparison:
        results['brute_force'] = {
            'score': best_score_BF,
            'runtime': bf_runtime,
            'solution': winning_comb_bf
        }
    
    return results



def timing_experiment(max_time_seconds=300):
    """
    Conducts timing experiment comparing Brute Force and SA-PAUSE algorithms.
    
    Args:
        max_time_seconds (int): Maximum time in seconds for brute force before breaking
    
    Returns:
        dict: Results dictionary containing timing and performance data
    """
    args = args_parser()
    args.sa_pause_simulation = True
    args.max_iterations_sa_pause = 30000  # Set max iterations to 30000
    
    # Initialize results storage
    results = {
        'K_values': [],
        'brute_force': {
            'times': [],
            'energies': [],
            'solutions': []
        },
        'sa_pause': {
            'times': [],
            'energies': [],
            'solutions': [],
            'iterations_to_99pct': []
        }
    }
    
    print("Starting timing experiment...")
    print(f"Max time limit for brute force: {max_time_seconds} seconds")
    print("="*60)
    
    for K in np.arange(10, 70, 5):
        m = max(1, int(np.sqrt(K)))
        print(f"\nTesting K={K}, m={m}")
        
        # Clean up memory
        gc.collect()
        
        # Create users for this test case
        local_models = {}
        
        class user():
            def __init__(self, ucb, g, p, args):
                self.ucb = ucb
                self.g = g
                self.privacy_reward = p
                self.args = args
                self.user_idx = None
        
        # Create test data
        base_arange = np.arange(K)/10
        base_array = np.concatenate([np.random.permutation(base_arange).reshape(-1,1) for i in range(3)], axis=1)
        
        for row in range(base_array.shape[0]):
            u = user(base_array[row,0], base_array[row,1], base_array[row,2], args)
            u.user_idx = row
            local_models[row] = u
        
        # Run Brute Force
        print(f"  Running Brute Force...")
        start_time_bf = time.time()
        
        users_idxs_comb = list(itertools.combinations([x for x in range(K)], m))
        np.random.shuffle(users_idxs_comb)
        winning_comb_bf = None
        best_score_BF = 0
        
        for comb in users_idxs_comb:
            score = utils.compute_energy(comb, local_models, args, num_users=K, num_users_per_round=m)
            if score > best_score_BF:
                best_score_BF = score
                winning_comb_bf = comb
        
        end_time_bf = time.time()
        bf_runtime = end_time_bf - start_time_bf
        
        print(f"    Brute Force: {bf_runtime:.3f}s, Energy: {best_score_BF:.6f}")
        
        # Store brute force results
        results['K_values'].append(K)
        results['brute_force']['times'].append(bf_runtime)
        results['brute_force']['energies'].append(best_score_BF)
        results['brute_force']['solutions'].append(winning_comb_bf)
        
        # Check if we exceeded time limit
        if bf_runtime > max_time_seconds:
            print(f"    ⚠️  Brute Force exceeded {max_time_seconds}s limit. Breaking loop.")
            # Remove this entry since it exceeded the limit
            results['K_values'].pop()
            results['brute_force']['times'].pop()
            results['brute_force']['energies'].pop()
            results['brute_force']['solutions'].pop()
            break
        
        # Run SA-PAUSE with target energy (99% of brute force optimal)
        target_energy = 0.99 * best_score_BF
        print(f"  Running SA-PAUSE (target: 99% = {target_energy:.6f})...")
        start_time_sa = time.time()
        
        energy_list, winning_comb_sa, best_score_SA = utils.choose_users(
            local_models, args, global_epoch=1, textio=None,
            method="sa_pause", num_users=K, num_users_per_round=m,
            target_energy=target_energy
        )
        
        end_time_sa = time.time()
        sa_runtime = end_time_sa - start_time_sa
        
        # Find iteration where 99% was reached
        iterations_to_99pct = len(energy_list)  # Default to all iterations if never reached
        for i, energy in enumerate(energy_list):
            if energy >= target_energy:
                iterations_to_99pct = i + 1
                break
        
        print(f"    SA-PAUSE: {sa_runtime:.3f}s, Energy: {best_score_SA:.6f}")
        print(f"    Reached 99% at iteration: {iterations_to_99pct}")
        
        # Store SA-PAUSE results
        results['sa_pause']['times'].append(sa_runtime)
        results['sa_pause']['energies'].append(best_score_SA)
        results['sa_pause']['solutions'].append(winning_comb_sa)
        results['sa_pause']['iterations_to_99pct'].append(iterations_to_99pct)
    
    return results


def save_timing_results(results, base_dir="running_time_comparison"):
    """
    Saves timing experiment results to a timestamped directory.
    
    Args:
        results (dict): Results from timing_experiment()
        base_dir (str): Base directory name
    
    Returns:
        Path: Path to the saved directory
    """
    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(base_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = save_dir / "timing_results.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {save_dir}")
    return save_dir


def plot_timing_comparison(results_path=None, results_dict=None, save_dir=None):
    """
    Plots timing comparison between Brute Force and SA-PAUSE algorithms.
    Can work with either saved results or fresh results dictionary.
    
    Args:
        results_path (str/Path): Path to saved timing_results.pkl file
        results_dict (dict): Fresh results dictionary from timing_experiment()
        save_dir (str/Path): Directory to save plots (if None, just display)
    
    Returns:
        tuple: (fig, ax) matplotlib objects
    """
    # Load results
    if results_dict is not None:
        results = results_dict
    elif results_path is not None:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    else:
        raise ValueError("Either results_path or results_dict must be provided")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    K_values = results['K_values']
    bf_times = results['brute_force']['times']
    sa_times = results['sa_pause']['times']
    
    # Plot both algorithms
    ax.plot(K_values, bf_times, 'o-', label='Brute Force', linewidth=2, markersize=8, color='red')
    ax.plot(K_values, sa_times, 's-', label='SA-PAUSE (to 99% optimal)', linewidth=2, markersize=8, color='blue')
    
    # Formatting
    ax.set_xlabel('Number of Users (K)', fontsize=18)
    ax.set_ylabel('Runtime (seconds)', fontsize=18)
    # ax.set_title('Runtime Comparison: Brute Force vs SA-PAUSE\n(SA-PAUSE stopped at 99% of optimal energy)', fontsize=14)
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Set log scale for y-axis if there's a large difference
    max_time = max(max(bf_times), max(sa_times))
    min_time = min(min(bf_times), min(sa_times))
    if max_time / min_time > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Runtime (seconds, log scale)', fontsize=18)
    
    # Add annotations for key points
    for i, (k, bf_time, sa_time) in enumerate(zip(K_values, bf_times, sa_times)):
        speedup = bf_time / sa_time
        # if i % 2 == 0:  # Annotate every other point to avoid clutter
        ax.annotate(f'{speedup:.2f}x', 
                    xy=(k, sa_time), xytext=(-10, 15), 
                    textcoords='offset points', fontsize=16, alpha=0.9)
    
    plt.tight_layout()
    
    # Save plots if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        plt.savefig(save_dir / "runtime_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / "runtime_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_dir}")
    

    return fig, ax


def run_complete_timing_analysis(max_time_seconds=300):
    """
    Runs the complete timing analysis: experiment + save + plot.
    
    Args:
        max_time_seconds (int): Maximum time in seconds for brute force
    
    Returns:
        Path: Directory where results were saved
    """
    print("🚀 Starting complete timing analysis...")
    
    # Run experiment
    results = timing_experiment(max_time_seconds)
    
    # Save results
    save_dir = save_timing_results(results)
    
    # Create and save plots
    fig, ax = plot_timing_comparison(results_dict=results, save_dir=save_dir)
    
    # Display summary
    print("\n" + "="*60)
    print("TIMING ANALYSIS SUMMARY")
    print("="*60)
    print(f"K values tested: {results['K_values']}")
    print(f"Average speedup (BF/SA): {np.mean([bf/sa for bf, sa in zip(results['brute_force']['times'], results['sa_pause']['times'])]):.2f}x")
    print(f"Max speedup: {max([bf/sa for bf, sa in zip(results['brute_force']['times'], results['sa_pause']['times'])]):.2f}x")
    print(f"Results saved in: {save_dir}")
    
    plt.show()
    return save_dir


if __name__ == "__main__":
    # You can run either the single comparison or the complete timing analysis
    
    # Option 1: Single run comparison (original functionality)
    res = single_run_comparison(num_users=300, num_users_per_round=15,
                          brute_force_comparison=False, pivot_fill_comparison=True,
                          use_dummy_data=False,
                            real_data_path="data_gathering_exp/20,50_2025-07-13_11-20-37/epoch_29.pkl")
    
    # # Option 2: Complete timing analysis
    # run_complete_timing_analysis(max_time_seconds=600)
    
    # # Option 3: Plot existing results
    # results_path = "running_time_comparison/2025-07-07_16-33-47/timing_results.pkl"
    # fig, ax = plot_timing_comparison(results_path=results_path, save_dir = Path(results_path).parent)
    # plt.show()