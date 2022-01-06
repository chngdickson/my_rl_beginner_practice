import os
import torch 
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork
from collections import OrderedDict
import gym

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=8,
                 env:gym.Env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 layer1_size=257, layer2_size=256, batch_size=256, reward_scale=10):
        self.batch_size = batch_size
        ## replay buffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.gamma = gamma
        
        # Networks
        self.actor = ActorNetwork(alpha, input_dims,n_actions=n_actions,
                                  fc1_dims=layer1_size,fc2_dims=layer2_size,
                                  name='actor',
                                  max_action= env.action_space.high)
        self.critic_1 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta=beta, input_dims=input_dims, n_actions=n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      name='critic_2')
        self.value = ValueNetwork(beta=beta, input_dims=input_dims, 
                                  fc1_dims=layer1_size,fc2_dims=layer2_size,
                                  name='value')
        self.target_value = ValueNetwork(beta=beta, input_dims=input_dims,
                                         fc1_dims=layer1_size,fc2_dims=layer2_size,
                                         name='target_value')
        
        # Hyper Parameters : policy eval, reward_scaling, target_smoothing_coefficient
        self.reward_scale = reward_scale
        self.tau = tau
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action,reward, next_state,done)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        ######## ValueNetworks smoothing #########
        # Get Params 
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        
        # Get state dict
        target_value_state_dict = self.target_value.state_dict()
        value_params_state_dict = self.value.state_dict()
        
        for name in value_params_state_dict:
            value_params_state_dict[name] = tau*value_params_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
        
        # Update state dict        
        self.target_value.load_state_dict(value_params_state_dict)
        
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        
    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action,  dtype=torch.float).to(self.actor.device)
        reward = torch.tensor(reward,  dtype=torch.float).to(self.actor.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        
        # 1 Dim value and target
        value = self.value(state).view(-1) 
        target_value = self.target_value(next_state).view(-1)
        target_value[done] = 0.0
        
        
        
        """Value Loss"""
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy,q2_new_policy)
        critic_value = critic_value.view(-1)
        
        # Value loss
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5* F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()
        
        
        """Actor loss"""
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy,q2_new_policy)
        critic_value = critic_value.view(-1)
        
        # Actor loss 
        # Evaluation
        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        q_hat = self.reward_scale*reward + self.gamma*value
        q1_old_policy = self.critic_1.forward(state,action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy,q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)
        
        critic_loss = critic_1_loss+critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.update_network_parameters()
        
        