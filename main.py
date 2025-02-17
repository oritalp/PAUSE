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



import utils



import shutil
from configurations import args_parser
from run_exp import run_exp


def main():
    args = args_parser()
    print(f"--full_exp is {args.full_exp}")
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_project_name = "PAUSE"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indicating_str = f"users={args.num_users}_alpha={args.alpha}_gamma={args.gamma}_eps_bar={args.epsilon_bar}_eta={args.epsilon_sum_deascent_coeff}_delta_f={args.delta_f}_dirichlet={args.dirichlet_coeff}_seed={args.seed}"
    if args.full_exp:
        exp_path = Path.cwd() / 'unified_results' / args.data / args.model / indicating_str / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path.mkdir(parents=True, exist_ok=True)
        if args.num_users <= 100:
            if args.i_i_d:
                methods=["sa_pause", 'pause brute', 'random', 'all users', "fastest ones"]
            else:
                methods=["sa_pause", 'pause brute', 'fraboni', 'all users', "fastest ones"]
        else:
            if args.i_i_d:
                methods = ["sa_pause", 'random', 'all users', "fastest ones"]
            else:
                methods = ["sa_pause", "fraboni", 'random', 'all users', "fastest ones"]

        paths_dict = {}
        for method in methods:
            args.method_choosing_users = method
            if args.wandb:
                # assure the previous run is finished
                if wandb.run is not None:
                    wandb.finish()                
                wandb.init( 
                    project=f"{wandb_project_name} - {indicating_str} - {start_time}",
                    name=method,
                    group=indicating_str,  # Groups all methods from same experiment together
                    config={
                        "exp_ID": indicating_str,
                        "model": args.model,
                        "num_users": args.num_users,
                        "num_users_per_round": args.num_users_per_round,
                        "global_epochs": args.global_epochs,
                        "max_seconds": args.max_seconds,
                        "epsilon_bar": args.epsilon_bar,
                        "epsilon_sum_deascent_coeff": args.epsilon_sum_deascent_coeff,
                        "delta_f": args.delta_f,
                        "data": args.data,
                        "alpha": args.alpha,
                        "gamma": args.gamma,
                        "method": method  # Add method to config for easier filtering
                    }
                )
            path_to_copy = Path(run_exp(args))
            print(f"finished running simulation with {method}")
            path_to_copy = path_to_copy.relative_to(Path.cwd())
            new_path = exp_path / method
            shutil.copytree(path_to_copy, new_path)
            paths_dict[method] = new_path / "last_model.pth.tar"

        if args.wandb:
            wandb.finish()
        

        print("finished running unified experiment with the following details: ",indicating_str, sep='\n')
        time.sleep(2) # sleep for 2 seconds to make sure that the models are saved
        utils.plot_graphs(paths_dict, path_to_save=exp_path, x_axis_time=True, print_graph=False)
        utils.plot_graphs(paths_dict, path_to_save=exp_path, x_axis_time=False, print_graph=False)

    else:
        if args.wandb:
            wandb.init(
                project=wandb_project_name,
                name=args.method_choosing_users + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                config={
                    "model": args.model,
                    "num_users": args.num_users,
                    "num_users_per_round": args.num_users_per_round,
                    "global_epochs": args.global_epochs,
                    "max_seconds": args.max_seconds,
                    "epsilon_bar": args.epsilon_bar,
                    "epsilon_sum_deascent_coeff": args.epsilon_sum_deascent_coeff,
                    "delta_f": args.delta_f,
                    "data": args.data,
                    "alpha": args.alpha,
                    "gamma": args.gamma,
                    "method": args.method_choosing_users
                }
            )
        run_exp(args)
        print(f"finished running simulation with {args.method_choosing_users} and the following details: ",indicating_str, sep='\n')

        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    main()

