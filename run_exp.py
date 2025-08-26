import gc
import sys
from statistics import mean

import torch
from tqdm import tqdm
import numpy as np
import itertools
from pathlib import Path
from scipy import special
import matplotlib.pyplot as plt
import datetime
import torch.linalg as LA
from torch.distributions.laplace import Laplace
from torchinfo import summary
import wandb


import time
import utils
import models
import learning_utils


def run_exp(args):
    base_path, textio, best_val_acc, path_best_model, last_model_path = utils.initializations(args)
    textio.cprint(str(args) if args.__class__.__name__ == 'Namespace' else str(vars(args)))
    args.exp_path = base_path

    # create the data loaders
    train_data, test_loader = utils.data(args)
    #input_var in the CNNs is the number of channels and in linear models is the size of the flatten pictures
    input_var, output, train_data = utils.data_arrangement(train_data, args)


    # model
    if args.model == 'mlp':
        global_model = models.FC3Layer(input_var, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input_var, output, args.data)
    elif args.model == 'cnn3':
        if args.data == 'cifar10':
            global_model = models.CNN3LayerCifar()
        else:
            global_model = models.CNN3LayerMnist()
    elif args.model == 'cnn5':
        if args.data == 'mnist' or args.data == 'fashion mnist':
            raise ValueError('CNN5 is not supported for MNIST type datasets')
        global_model = models.CNN5Layer(input_var, output)
    elif args.model == 'linear':
        global_model = models.Linear(input_var, output)
    elif args.model == 'mobilenetv2':
        if args.data == 'imagenet100':
            global_model = models.MobileNetV2(num_classes=output, pretrained=False)
        elif args.data in ['imagewoof', 'tiny_imagenet']:
            print(f"Creating MobileNetV2 for {args.data} with dynamic feature calculation...")
            
            # Create a temporary standard torchvision model to calculate feature size
            import torchvision.models as torchvision_models
            temp_model = torchvision_models.mobilenet_v2(weights=None)
            temp_model.eval()
            
            # Test with 64x64 input size (both ImageWoof and Tiny ImageNet use 64x64)
            with torch.no_grad():
                test_input = torch.randn(1, 3, 64, 64)
                # Get features before the final classifier
                features = temp_model.features(test_input)
                # MobileNetV2 uses adaptive average pooling
                pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                flattened_features = torch.flatten(pooled_features, 1)
                calculated_features = flattened_features.shape[1]
            
            print(f"🔧 MobileNetV2 backbone output for {args.data} (64x64): {flattened_features.shape}")
            print(f"🔧 Calculated in_features for final layer: {calculated_features}")
            
            # Now create the actual model using your custom MobileNetV2 class
            global_model = models.MobileNetV2(num_classes=output, pretrained=False)
            # Update the classifier with the dynamically calculated size
            global_model.mobilenet.classifier[1] = torch.nn.Linear(calculated_features, output)
            
            print(f"✅ MobileNetV2 configured for {args.data}: {calculated_features} -> {output} classes")
        else:
            raise ValueError('MobileNetV2 is currently only supported for ImageNet-100, ImageWoof, and Tiny ImageNet')

    textio.cprint(str(summary(global_model, verbose=0)).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'))
    global_model = global_model.to(args.device)
    print(f"global model's device: {next(global_model.parameters()).device}")

    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        textio.cprint("Using mixed precision training (AMP)")
    else:
        textio.cprint("Using standard precision training")

    train_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    local_models = utils.users_partition(global_model, train_data, args, i_i_d=args.i_i_d)
    utils.update_data_equility_partititon(local_models, args)

    #create fraboni distributions if args,choosing_users_method == "fraboni"
    if args.method_choosing_users == "fraboni":
        args.fraboni_distributions = utils.create_fraboni_probs(local_models, args)
        textio.cprint(f"fraboni distributions: {args.fraboni_distributions}")
        if args.fraboni_distributions.sum(axis=0).any() == 0:
            raise ValueError("one of the users has a zero probability in all the distributions using fraboni")

    choices_table = np.zeros((args.global_epochs, args.num_users))
    num_of_obs_arr = np.zeros((1,args.num_users))
    train_loss_list = []
    val_acc_list = []
    val_losses_list = []
    l1_norm_avg_deltha_theta_list = []
    global_epochs_time_list = []
    privacy_violations_list = []
    max_privacy_violation = 0


    time_counter = 0
    for global_epoch in tqdm(range(1, args.global_epochs+1)):
        """Part 1: Choosing Users"""
        for usr_idx in range(args.num_users):
            local_models[usr_idx].update_g(global_epoch)
            local_models[usr_idx].update_ucb(global_epoch)


        if args.alternative_privacy_reward:
            # for the alternative privacy reward we manage the update of the reward here instead of in the update_privacy_terms_and_violations
            variance_terms = np.array([local_models[usr_idx].compute_var_term(local_models[usr_idx].num_of_obs + 1) for usr_idx in range(args.num_users)])

            if (max(variance_terms) > 1) and (max(variance_terms) - min(variance_terms) != 0):
                variance_terms = variance_terms / max(variance_terms)
            
            one_minus_variance_terms = 1 - variance_terms

            for usr_idx in range(args.num_users):
                local_models[usr_idx].privacy_reward = one_minus_variance_terms[usr_idx]

        
        if args.choosing_users_verbose:
            textio.cprint(f"iteration: {global_epoch}")
        
        rounds_choise = utils.choose_users(local_models, args, global_epoch, textio,
                                            method=args.method_choosing_users)
        # create a version of rounds_choise without repetitions
        rounds_choise_no_rep = tuple(set(rounds_choise))

        if args.method_choosing_users == "fraboni":
            # if we're in fraboni, rounds_chice is a 2D array with the first row being the user index
            # and the second row being the number of times the user was chosen

            # count the number of times each user was chosen in rounds_choise
            rounds_choice_arr = np.array([range(args.num_users), np.zeros(args.num_users)])
            for usr_idx in rounds_choise:
                rounds_choice_arr[1, usr_idx] += 1
            
            # delete columns with zeros in their second row
            rounds_choise = rounds_choice_arr[:, rounds_choice_arr[1] != 0]
            #print(f"rounds_choise_arr for epoch No.{global_epoch}: {rounds_choise}")


        
        choices_table[global_epoch-1, rounds_choise_no_rep] = 1
        num_of_obs_arr[0,rounds_choise_no_rep] += 1
        num_slow_users = 0
        num_fast_users = 0
        for usr_idx in sorted(rounds_choise_no_rep):
            local_models[usr_idx].update_emp_avg()
            local_models[usr_idx].update_privacy_terms_and_violations()
            if local_models[usr_idx].privacy_violation > max_privacy_violation:
                max_privacy_violation = local_models[usr_idx].privacy_violation
            if args.choosing_users_verbose:
                textio.cprint(f"user {usr_idx}, emp_avg: {local_models[usr_idx].emp_avg}, h: {local_models[usr_idx].ucb_generalization}, ucb: {local_models[usr_idx].ucb},num_of_obs: {local_models[usr_idx].num_of_obs}, privacy reward: {local_models[usr_idx].privacy_reward}, g: {local_models[usr_idx].g}, curr_delay = {local_models[usr_idx].last_access_time}, data_size = {len(local_models[usr_idx].data_loader.dataset)}")
            if usr_idx < args.num_users//2:
                num_fast_users += 1
            else:
                num_slow_users += 1
            local_models[usr_idx].increase_num_of_obs()
        if args.choosing_users_verbose:
            textio.cprint(f"num of fast users chosen: {num_fast_users}, num of slow users chosen: {num_slow_users}")
        
        max_delay = max([local_models[i].last_access_time for i in rounds_choise_no_rep])
        
        # Add shared resource constraint latency penalty
        if args.shared_res_constraint:
            latency_penalty, shared_res_penalty = utils.compute_resource_constraint_penalty(rounds_choise_no_rep, args)
            max_delay += latency_penalty
            
            if args.shared_res_constraint_verbose:
                verbose_info = utils.get_resource_constraint_verbose_info(rounds_choise_no_rep, args)
                textio.cprint(f"Shared resource constraint latency penalty: {latency_penalty:.4f} seconds")
                textio.cprint(f"Shared resource constraint reward penalty: {shared_res_penalty:.4f}")
                if verbose_info['collisions']:
                    textio.cprint("Resource cluster collisions:")
                    for cluster_id, info in verbose_info['collisions'].items():
                        textio.cprint(f"  Cluster {cluster_id}: {info['count']} users {info['users']} (excess: {info['excess']})")
        
        if args.choosing_users_verbose:
            textio.cprint(f"max_delay = {max_delay:.2f} seconds")

        privacy_violations_list.append(max_privacy_violation)
        
        
        
        """Part 2: Training"""
        learning_utils.distribute_model(local_models, global_model)
        users_avg_loss_over_local_epochs = []


        for user_idx in rounds_choise_no_rep:
            user_loss = []
            for local_epoch in range(args.local_epochs):
                user = local_models[user_idx]
                train_loss = learning_utils.train_one_epoch(user, train_criterion, args, scaler)
                if args.lr_scheduler:
                    user.scheduler.step(train_loss)
                user_loss.append(train_loss)
            users_avg_loss_over_local_epochs.append(mean(user_loss))
        
        avg_loss_over_chosen_users_curr_global_epoch = mean(users_avg_loss_over_local_epochs)
        train_loss_list.append(avg_loss_over_chosen_users_curr_global_epoch)

        
        avg_deltha_theta = learning_utils.Fed_avg_models(local_models, global_model, rounds_choise, textio
                                                        ,args, snr_verbose = args.snr_verbose)
        

        val_acc, val_loss = learning_utils.test(test_loader, global_model, test_criterion, args)
        if global_epoch == 1:  # Only print on first epoch
            print("Debug: Checking class indices in validation set...")
            for i, (data, labels) in enumerate(test_loader):
                print(f"Batch {i}: label range = {labels.min().item()} to {labels.max().item()}")
                if i >= 2:  # Just check first few batches
                    break
        val_acc_list.append(val_acc) ; val_losses_list.append(val_loss)
        

        time_counter += max_delay
        textio.cprint((f"global epoch {global_epoch} has been done artifficialy in {max_delay:.2f} secs, the total time by now is {time_counter:.2f} \n with avg train loss {avg_loss_over_chosen_users_curr_global_epoch:.3f}, val loss {val_loss:.3f}, avg val acc {val_acc:.2f}%"))
        global_epochs_time_list.append(time_counter)
        gc.collect()


        if val_acc > best_val_acc and args.save_best_model:
            best_val_acc = val_acc
            torch.save({"model's state dict":global_model.state_dict(),
                        "train_loss_list": train_loss_list,
                        "val_acc_list": val_acc_list,
                        "val_losses_list": val_losses_list,
                        "global_epochs_time_list": global_epochs_time_list,
                        "num_of_users": args.num_users,
                        "num_of_users_per_round": args.num_users_per_round,
                        "privacy_violations_list": privacy_violations_list}
                        , path_best_model)
        
        
        with open(last_model_path, "wb") as f:
            torch.save({"train_loss_list": train_loss_list,
                    "val_acc_list": val_acc_list,
                    "val_losses_list": val_losses_list,
                    "global_epochs_time_list": global_epochs_time_list,
                    "num_of_obs_arr": num_of_obs_arr.reshape(-1),
                    "global_epoch": global_epoch,
                    "num_of_users": args.num_users,
                    "num_of_users_per_round": args.num_users_per_round,
                    "privacy_violations_list": privacy_violations_list}
                    , f)
            f.flush()
        if args.wandb:
            # Log metrics vs epochs
            wandb.log({
                "epoch": global_epoch,
                "time": time_counter,
                "train_loss": avg_loss_over_chosen_users_curr_global_epoch,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "max_privacy_violation": max_privacy_violation
            })
        
        if global_epoch % args.bar_plot_interval ==0:
            path_bar_plot = last_model_path.parent / "Number of times each user was chosen.png"
            utils.plot_layered_user_selections(choices_table[:global_epoch], path_bar_plot,
                                                args, interval=args.bar_plot_interval)
            

        if time_counter > args.max_seconds:
            break

    return base_path
