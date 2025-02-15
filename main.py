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



import utils



import shutil
from configurations import args_parser
from run_exp import run_exp


def main():
    args = args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indicating_str = f"No_of_users={args.num_users}_alpha={args.alpha}_gamma={args.gamma}_epsilon_bar={args.epsilon_bar}_epsilon_sum_deascent_coeff={args.epsilon_sum_deascent_coeff}_delta_f={args.delta_f}"
    if args.full_exp:
        exp_path = Path.cwd() / 'unified_results' / args.data / args.model / indicating_str / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_path.mkdir(parents=True, exist_ok=True)
        methods = ["sa_pause",'pause brute', 'random', 'all users', "fastest ones"]
        paths_dict = {}
        for method in methods:
            args.method_choosing_users = method
            path_to_copy = Path(run_exp(args))
            print(f"finished running simulation with {method}")
            path_to_copy = path_to_copy.relative_to(Path.cwd())
            new_path = exp_path / method
            shutil.copytree(path_to_copy, new_path)
            paths_dict[method] = new_path / "last_model.pth.tar"
        

        print("finished running unified experiment with the following details: ",indicating_str, sep='\n')
        time.sleep(2) # sleep for 2 seconds to make sure that the models are saved
        utils.plot_graphs(paths_dict, path_to_save=exp_path, x_axis_time=True, print_graph=False)
        utils.plot_graphs(paths_dict, path_to_save=exp_path, x_axis_time=False, print_graph=False)

    else:
        run_exp(args)
        print(f"finished running simulation with {args.method_choosing_users} and the following details: ",indicating_str, sep='\n')


if __name__ == '__main__':
    main()

