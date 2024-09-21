from ase.io import read
from ase.neighborlist import NeighborList, neighbor_list
import numpy as np
import torch 
from collections import Counter


def remove_infeasable( prob: torch.tensor, oxidised_atm: list , net_index : int ): 
    # prob :  dim( N_atm , 4)
    # return a flattened, normlaised tensor for correpsond to the FG probability distribution
    if net_index == 0 :  # EP is added
        prob = prob[:,:2]
    else:  # OH is added
        prob = prob[:,2:] 
    # if a carbon is oxidsed aldy, both the upper and lower side should be set to 0 
    prob[oxidised_atm,:] = 0    
    norm_factor  = torch.sum(prob)
    return (prob/ norm_factor).t().flatten()

def remove_infeasable_ppo( prob:torch.tensor, oxidised_atm: list , net_index : int ): 
    # prob :  dim( N_atm*4 , )
    # return a flattened, normlaised tensor for correpsond to the FG probability distribution
    dim =  prob.shape[0]
    if net_index == 0 :  # EP is added
        prob = prob[:int(dim/2)].clone()
    else:  # OH is added
        prob = prob[int(dim/2):].clone()
    # if a carbon is oxidsed aldy, both the upper and lower side should be set to 0 
    oxidised_index =  oxidised_atm + [i +int( dim/4) for i in oxidised_atm]
    prob[oxidised_index] = 0    
    scaled_prob = prob/prob.sum()
    return scaled_prob

def calc_loss(list_of_action_and_action_dist: list , reward: float ):
    """
    calculate loss/return 
    
    list_of_action_and_action_dist: [int, torch_distribution] 

    programme designed to maximize the reward, thus, positive 
    """    
    sum_log_prob = 0
    for action_and_action_dist in list_of_action_and_action_dist:
        sum_log_prob += action_and_action_dist[1].log_prob(action_and_action_dist[0])
    return -sum_log_prob * reward

def check_C_num ( config): 
    return  Counter(config.get_atomic_numbers())[6]

def nb_dict_construct( config:Atoms, 
     cutoff:dict = {(1,8): 0.98, (6,8): 1.70, (6,6): 1.85}) : 
    
    i,j = neighbor_list('i''j', config, cutoff= cutoff)
    nn_list = np.array((i, j)).T 

    # the above cutoffs are concordant with Zak's code on GO generation
    # need ot increase the expectancy length for C-O for the increase in 1500K simualiton 
    # use the 1.7 A to represent all of C=O, C-O-C, C-OH
    # 1.7 is really an arbitary choice, checked on a few configurations with ovito, appears to give a good return 
    neighbour_count = { i:[] for i in range( len(config))}
    for  nn_pair in nn_list: 
        neighbour_count[nn_pair[0]].append(nn_pair[1])
    return neighbour_count



