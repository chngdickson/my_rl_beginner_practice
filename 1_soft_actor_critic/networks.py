import os
import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

# S, A -> Q
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, 
                 fc1_dims = 256, fc2_dims=256, 
                 name='critic', chkpt_dir ='tmp/sac'):
        super(CriticNetwork,self).__init__()

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.fc1 = nn.Linear(input_dims+n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state, action):
        # Returns action value Q
        action_value = self.fc1(torch.cat([state,action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        
        q = self.q(action_value)
        
        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# Value function dont care about any actions we take or are taking
# What value function care is
# V
class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork,self).__init__()
        
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.fc1 = nn.Linear(input_dims,fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        
        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        v = self.v(state_value)
        
        return v
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
        
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, 
                 fc1_dims=256, fc2_dims=256, n_actions=2,
                 discrete_or_cont='cont',
                 name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.discrete_or_cont = discrete_or_cont
        self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        
        self.reparam_noise = 1e-6 # Make sure we dont take log of 0 which is undefined.
        
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)
        
        
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1.0)
        
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        # If discrete, assign probability with softmax
        # If continuous, use gausssian
        mu, sigma = self.forward(state)
        
        #if self.discrete_or_cont == 'cont':
        probabilities = Normal(mu,sigma) 
        
        if reparameterize:
            actions = probabilities.rsample() #adding some noise
        else:
            actions = probabilities.sample()
        
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action, log_probs
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
    
    