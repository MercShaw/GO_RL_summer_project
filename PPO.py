from network import single_head_network, two_head_output
import torch 
import torch.optim as optim
import numpy as np 
import time
import json
import os
from torch.distributions import Categorical
from scipy.stats import bernoulli
from ase.build import graphene_nanoribbon
from training_util import remove_infeasable_ppo,nb_dict_construct
from run_MD import set_up_md, run, collect_md_data
from single_point_ox_simplified import add_EP, add_OH


device =  torch.device("cuda")
torch.autograd.set_detect_anomaly(True)
class graphene_env: # this is the env
    def __init__(self, m:int, n:int, c_ratio: float, o_ratio:float, h_ratio:float, buckle_EP, buckle_OH, T : float = 300):
        self.m = m
        self.n = n
        self.graphene =  graphene_nanoribbon( self.m, self.n, type="armchair", saturated=False, sheet=True, vacuum=10) 
        self.n_O = round ( len(self.graphene) * o_ratio/c_ratio) # tot number of O to add
        self.n_H = round ( len(self.graphene) * h_ratio/c_ratio) # tot number of H to add
        self.n_OH = self.n_H # tot number of OH groups to add
        self.n_EP = self.n_O - self.n_OH # tot number of epoxide to add
        self.N_atom = len(self.graphene)
        self.nn_list = nb_dict_construct( self.graphene )
        self.buckle_OH = buckle_OH
        self.buckle_EP = buckle_EP

        self.temp = T
        self.MD_param = {
        'n_C': self.N_atom,
        'n_O': self.n_O,
        'n_H': self.n_H, 
        'T': self.temp}

        # internal state

        self.state_tensor = torch.zeros(self.N_atom*4, device= device)
        # this is the onehot embedding of the GO, 
        # segmented in (up_EP, down_EP, up_OH, down_OH )
        self.state = None
        self.oxidised_list =  []
        self.n_EP_left = self.n_EP
        self.n_OH_left = self.n_OH 

    def reset(self): # reset function,  initialise everything
        self.state_tensor = torch.zeros( self.N_atom*4, device= device)
        self.state = graphene_nanoribbon( self.m, self.n, type="armchair", saturated=False, sheet=True, vacuum=10) 
        self.oxidised_list = []
        self.n_EP_left = self.n_EP
        self.n_OH_left = self.n_OH 
        return self.state, self.state_tensor, self.oxidised_list, self.n_EP_left, self.n_OH_left
    
    def reward( self, iter):
        # this is the reward at the end of an episode
        # given by MD returned stress - strain curve
        simulaiton_name = set_up_md(self.state, iter, self.MD_param)
        run(self.state, simulaiton_name , self.temp)
        elastic_e = collect_md_data(simulaiton_name)
        reward  =  elastic_e  # let's just try this -  pure potential energy without any modification
        return reward
        #return (torch.rand(1)*10) .item()
    
    def step(self, action_index, net_index, iter): 
        # this is a single aciton to add OH / EP 
        final_reward = 0
        done = False
        if net_index == 0 : 
            self.state, self.oxidised_list, self.state_tensor , flag = add_EP(
                    action_index,
                    self.oxidised_list,
                    self.nn_list,
                    self.state,
                    self.N_atom, 
                    self.state_tensor, 
                    self.buckle_EP)
            if flag : 
                    self.n_EP_left -= 1

        elif net_index == 1 : 
            self.state, self.oxidised_list, self.state_tensor , flag = add_OH(
                    action_index,
                    self.oxidised_list,
                    self.nn_list,
                    self.state,
                    self.N_atom, 
                    self.state_tensor,
                    self.buckle_OH )
            if flag : 
                    self.n_OH_left -= 1
        #so in the step we will update the state  and the state tensor 

        if self.n_EP_left + self.n_OH_left == 0: 
            # the reward is only updated at the end of this episode
            final_reward = self.reward(iter) 
            done = True
        return self.state, self.state_tensor, self.oxidised_list, final_reward, done, {}
    
class PPO: # this is the agent 
    def __init__(self,  lr=3e-4, gamma=0.99, clip_epsilon=0.2, K_epochs=4,lam=0.95 ):
        self.policy =two_head_output(input_dim= 120, out_dim=120, hidden_dim= 800, n_layes=2).to(device)
        # use a Parallel MLP for reward prediction - as OH net and EP net not updates simultaneously
        # use the single branch in this parallel MLP as the 'critic' to give the state energy
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
   
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs

    def select_action(self,n_OH_left, n_EP_left, state_tensor, oxidised_carbon):
        # this will select the action based on a sigle state_tensor 
        # first use the bernouli distribution to select wich group to add
        net_index = bernoulli( n_OH_left/ (n_OH_left + n_EP_left)).rvs()
        # return 0 if n_EP_left is more, 1 if n_OH_left is more 
        # this matches the index of 2nd dimemsion of the onehot encoded state
        
        output,_ = self.policy(state_tensor)
        # the output should have the shape (N_atm*4, ), softmaxed 
        probability =  remove_infeasable_ppo(output, oxidised_carbon, net_index)
        # this function remove the actions that lead to repeat oxidaiton, remove the half that's not updated in each step and rescale them 
        # return (N_atm*2, ) 
        dist = Categorical(probability)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), net_index
        # action would be an integer in range ( 0, N_atm*4 ), indicating WHICH carbon, on WHICH side
        # net index is separate and not determined by this, but still returned
    def normalization ( self, tensor): 
        if len( tensor)> 3: 
            mean_reward = tensor.mean()
            std_reward = tensor.std()

            # Normalize rewards
            normalised_tensor = (tensor - mean_reward) / (std_reward + 1e-8)
            return normalised_tensor
        else: return tensor*0.1

    def compute_advantage(self, rewards, values, dones ):
        
        # Normalize rewards
        normalized_rewards = rewards/10 # this is to apply a scaling factor to convert the returns from 7.x to 0.7x for better performance
        normalized_values  = values/10 # as we applied to rewards same need ot be done on values
        # advanatge is defined using GAE -  see paper for more details 
        # rewards - torch.tensor of shape ( N_O, ),  reward collected in down a trajectory 
        # values - torch.tensor of shape ( N_O+1, ) , Critic predicted reward based on state,
        # dones -  torch.tensor of shape ( N_O, ), flags indicating if the current step is the final step, boolean 
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(normalized_rewards))):
            next_value = normalized_values[i + 1]
            delta = normalized_rewards[i] + (1 - dones[i]) * self.gamma * next_value - normalized_values[i]
            gae = delta + (1 - dones[i]) * self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + normalized_values[i])
        return torch.FloatTensor(advantages).to(device), torch.FloatTensor(returns).to(device)

    def update(self, memory):
        state_tensors, actions, log_probs, rewards, values, dones, filters = memory
        # implementation of th surrogate clipped objective function that PPO algorithm used to optimise 

        # all the above argument except for 'filter' are torch tnesor in shape of ( batch_size, feature_dim)
        # filters is list of len =  batch, with compoent being (ox_list, net_index) recorded for each step 
        # collected in thie trajectory in sample of minibatch

        advantages, returns =  self.compute_advantage(rewards, values, dones)
        
        for _ in range(self.K_epochs):
        
            output, state_values  = self.policy(state_tensors)
            # since input is batch, with shape ( N_O, N_atm*4), expect same dimension with the output 
            prob = torch.stack ( [remove_infeasable_ppo(output[i], filters[i][0], filters[i][1]) for i in range( output.shape[0])] ).to(device)
            # apply the filtering row wise on the prob minibatch

            
            dist = Categorical(prob)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (0.5 * (returns - state_values).pow(2)).mean()
            explore_exploit_trade_off = - (0.01 * entropy).mean()
            loss = actor_loss + critic_loss + explore_exploit_trade_off
            
            # update the branch accordingly 
            for params in self.policy.parameters(): 
                params.requires_grad = True
            # then switch of the half brunch that doesn;t requries update on 
            if net_index == 0 : 
                # if last net_index refers to an ep, only update the parameters in the 1st half 
                for param in self.policy.H_net.parameters(): 
                    param.requires_grad = False
            else:
                for param in self.policy.E_net.parameters(): 
                    param.requires_grad = False 
            
            # use the same optimizer to make sure both the actor and critc are updated simultaneously 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



agent =  PPO()
exisiting_model = '/policy_model/ppo_energy_policy.pth'#existing agent 

if exisiting_model in os.listdir('/policy_model'): 
   
    agent.policy.load(  torch.load(exisiting_model ) )

env = graphene_env(n=3, m=5, c_ratio=60, o_ratio=20, h_ratio=10, buckle_EP = np.arange(0.1, 0.16, 0.01),
    buckle_OH = np.arange(0.1, 0.36, 0.01), T= 1000 )

start_iter = 127
num_iterations=1500
batch_size=int(env.n_O/2) # this batch size should be the same as the number of steps 

data_store = {
    'episopde': [],
    'reward':[]
}

with open('PPO_energy_training.json', 'w') as file: 
    file.write( '[')
    for iter in range(start_iter, start_iter+ num_iterations):
        start_time = time.time( )
        env.state, env.state_tensor, env.oxidised_list,*_ =  env.reset()
        done = False
        memory = []
        reward = 0
        filters = []
        state_tensors = []
        values  =[]
        actions = []
        log_probs = []
        rewards = []
        dones = []
        #rewards, values, log_probs, actions, state_tensors, dones, filters memory

        while env.n_EP_left+ env.n_OH_left != 0:
            action, log_prob, net_index = agent.select_action(env.n_OH_left, env.n_EP_left, env.state_tensor, env.oxidised_list)
            next_state, next_state_tensor, env.oxidised_list, reward, done, _ = env.step(action, net_index, iter)

            state_tensors.append(env.state_tensor) 
            # for the input of policy network, need to flatten this 
            values.append( agent.policy(env.state_tensor)[1].item())
            # this would be the predicted value of the current state
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            filters.append( (env.oxidised_list, net_index)) 
            # this is also reuqired to see in the trajectory when making the decision which actions are filtered

            state_tensor = next_state_tensor
            if done :
                # update the agent at the end of the trajectory
                values.append(reward)
                state_tensor_batch  =  torch.stack(state_tensors).to(device)
                memory.append(( state_tensor_batch,
                                torch.LongTensor(actions).to(device),
                                torch.FloatTensor(log_probs).to(device),
                                torch.FloatTensor(rewards).to(device),
                                torch.FloatTensor(values).to(device),
                                torch.FloatTensor(dones).to(device),
                                filters) 
                                )
                state_tensors, actions, log_probs, rewards, values, dones, filters= [], [], [], [], [], [],[]
                agent.update(memory[-1])
                memory = []
        data_store['episopde'].append(iter)
        data_store['reward'].append(reward)
        

        if iter % 5 == 0:
            print( f'iteration:{iter} ----- {time.time()- start_time} s')
            json.dump( data_store, file)
            file.write(',\n')
            file.flush()
            torch.save(agent.policy.state_dict(), '/policy_model/ppo_energy_policy.pth')

print('done')
