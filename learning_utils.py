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


def Fed_avg_models(local_models, global_model, chosen_users_idxs, args, privacy = False): 
    """this is a fed avg that averages according to the data length and quality for the non i.i.d case, in oppose
    to nataly's implementation"""
    #mean = lambda x: sum(x) / len(x)
    state_dict = copy.deepcopy(global_model.state_dict())
    data_length_sum = 0
    for j in chosen_users_idxs:
        data_length_sum += local_models[j].data_quality*len(local_models[j].data_loader.dataset)
    
    returned_delta_thetas = {}
    l1_norms_arr = np.zeros((2,args.num_users))
    l1_norms_arr[0,:] = np.arange(args.num_users)


    for key in state_dict.keys():
        delta_theta_average = torch.zeros_like(state_dict[key])
        for user_idx in chosen_users_idxs:
            delta_theta = local_models[user_idx].model.state_dict()[key] - state_dict[key]
            #checking yhe l1 norm of the delta theta over the last global epoch for each user,
            #this is for testing purposes and can be removed later on
            #l1_norms_arr[1,user_idx] += LA.norm(delta_theta.flatten(), ord=1).detach().numpy()

            if privacy:
                #delta f is fixed as 2 for now
                lap_noise = Laplace(torch.tensor([0.0]), torch.tensor(2/local_models[user_idx].next_privacy_term))
                delta_theta += lap_noise.sample(delta_theta.shape).to(state_dict[key].dtype)
                #this might be a problem with parameters that supposed to be int 
                #like number of batches in the batchnorm layer but it has been ran and worked before

            delta_theta_average += (delta_theta * ((local_models[user_idx].data_quality*
                                      len(local_models[user_idx].data_loader.dataset))/data_length_sum)).to(state_dict[key].dtype)

        returned_delta_thetas[key] = delta_theta_average       
        state_dict[key] += delta_theta_average
    
    #filter only the len(chosen_users_idxs) largest l1 norms
    #l1_norms_arr = l1_norms_arr[:,np.argsort(l1_norms_arr[1,:])[::-1]][:,:len(chosen_users_idxs)]
    #for i in range(len(chosen_users_idxs)):
    #    print("user {}'s l1 norm is {}".format(l1_norms_arr[0,i], l1_norms_arr[1,i]))


    global_model.load_state_dict(copy.deepcopy(state_dict))
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