import utils
import torch
import torch.linalg as LA
from statistics import mean
import copy
import numpy as np
from torch.distributions.laplace import Laplace


def train_one_epoch(user, train_criterion ,args):
    user.model.train()
    losses_per_local_epoch = []
    if args.local_iterations is not None:
        local_iteration = 0
    for batch_idx, (data, label) in enumerate(user.data_loader):
        # send to device
        data, label = data.to(args.device), label.to(args.device)
        output = user.model(data)
        loss = train_criterion(output, label)

        user.opt.zero_grad()
        loss.backward()
        user.opt.step()

        losses_per_local_epoch.append(loss.item())

        if args.local_iterations is not None:
            local_iteration += 1
            if local_iteration == args.local_iterations:
                break
    return mean(losses_per_local_epoch)


def distribute_model(local_models, global_model):
    """Distribute the global model to local models in the beggining of each global epoch. This means that the
    global model is being averaged with the noise additions at the end of the previous global epoch."""
    for usr_idx in range(len(local_models)):
        local_models[usr_idx].model.load_state_dict(copy.deepcopy(global_model.state_dict()))


def Fed_avg_models(local_models, global_model, chosen_users_idxs, args, l1_norms_verbose = False, snr_verbose = False): 
    """this is a fed avg that averages according to the data length and quality for the non i.i.d case, in oppose
    to nataly's implementation"""
    #mean = lambda x: sum(x) / len(x)
    state_dict = copy.deepcopy(global_model.state_dict())
    data_length_sum = 0
    for j in chosen_users_idxs:
        data_length_sum += local_models[j].data_quality*len(local_models[j].data_loader.dataset)
    
    returned_delta_thetas = {}
    if l1_norms_verbose:
        l1_norms_arr = np.zeros((2,args.num_users))
        l1_norms_arr[0,:] = np.arange(args.num_users)

    users_delta_thetas = {}



    for key in state_dict.keys():
        delta_theta_average = (torch.zeros_like(state_dict[key])).type(torch.float32) 
        for user_idx in chosen_users_idxs:
            delta_theta = (local_models[user_idx].model.state_dict()[key] - state_dict[key]).type(torch.float32)
            if l1_norms_verbose:
                #checking yhe l1 norm of the delta theta over the last global epoch for each user,
                #this is for testing purposes and can be removed later on
                l1_norms_arr[1,user_idx] += LA.norm(delta_theta.flatten(), ord=1).detach().numpy()

            if snr_verbose:
                delta_theta_copy = copy.deepcopy(delta_theta)
                try:
                    users_delta_thetas[user_idx] = torch.cat((users_delta_thetas[user_idx], torch.abs(delta_theta_copy.flatten())), 0)
                except KeyError:
                    users_delta_thetas[user_idx] = torch.abs(delta_theta_copy.flatten())
  
            if args.privacy:
                #delta f is fixed as 10^-3 for now
                lap_noise = Laplace(torch.tensor([0.0]),
                                     torch.tensor(args.delta_f/local_models[user_idx].next_privacy_term))
                added_noise = lap_noise.sample(delta_theta.shape).squeeze(-1).to(args.device)
                delta_theta += added_noise


            

            

            delta_theta_average += (delta_theta * ((local_models[user_idx].data_quality*
                                      len(local_models[user_idx].data_loader.dataset))/data_length_sum))
        


        delta_theta_average = delta_theta_average.to(state_dict[key].dtype) #we cast it back to the original dtype
                                                                            #because for trainable parmeters that are int
                                                                            #like the number of batches in the batchnorm layer
                                                                            #the aggregation is made in float32 (even though 
                                                                            #we use momentum in BN layers so num_betaches_tracked
                                                                            #has no meaning for us and this is the only int parameter)
        returned_delta_thetas[key] = delta_theta_average       
        state_dict[key] += delta_theta_average

    if l1_norms_verbose:
        #filter only the len(chosen_users_idxs) largest l1 norms
        l1_norms_arr = l1_norms_arr[:,np.argsort(l1_norms_arr[1,:])[::-1]][:,:len(chosen_users_idxs)]
        for i in range(len(chosen_users_idxs)):
           print("user {}'s l1 norm is {}".format(l1_norms_arr[0,i], l1_norms_arr[1,i]))


    global_model.load_state_dict(copy.deepcopy(state_dict))

    if snr_verbose:
        for key in users_delta_thetas.keys():
            print(f"user No.{key} has been picked {local_models[user_idx].num_of_obs} and his deltas theta's mean is {torch.mean(users_delta_thetas[key])}, the added laplace noise var is {args.delta_f/local_models[user_idx].next_privacy_term}")
            #print(f"and the empirical var of deltas thetas is {torch.var(users_delta_thetas[key], correction=0)}")


    return returned_delta_thetas



def test(test_loader, model, criterion, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, label = data.to(args.device), label.to(args.device)  # send to device

        output = model(data)
        test_loss += criterion(output, label).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the class woth th highest predicted value
                                                    # (which also means the class with the highest 
                                                    # log softmax)
        correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss