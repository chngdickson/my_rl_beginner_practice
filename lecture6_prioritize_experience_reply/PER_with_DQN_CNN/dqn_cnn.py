from collections import deque
from math import tau
import gym
import random
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
# Image preprocessing
from skimage import transform as ski_transform
# Plots
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import itertools
import pylab
import os
# Custom Lib
from prioritized_memory import Memory

class Net(nn.Module):
    def __init__(self,tau,num_actions):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(tau,32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136,512)
        self.fc2 = nn.Linear(512,num_actions)
    
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # Last layer doesn't have activation
        return x
    
    def save(self, file_name='breakout_CNN.pth'):
        # Create a folder to store models
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        # Save the file
        torch.save(self.state_dict(), file_name)
    
class Agent():
    def __init__(self, gym_env_name='Breakout-v0'):
        self.env = gym.make(gym_env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Hyper Params
        self.lr = 0.001
        self.tau = 4
        # Maybe i should use RMSProp
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Model Definition
        self.model = Net(tau = self.tau, num_actions=self.action_size).to(self.device)
        self.target_model = Net(tau = self.tau, num_actions=self.action_size).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr)
        
        # Memory
        self.memory = Memory(capacity= 10000)
        self.batch_size = 64
        self.train_start = 1000 # Start training model when this has 1000
        
        # New state type
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        
    
    def get_action(self,state):
        if np.random.random() <= self.epsilon or len(state)<self.tau:
            #print(len(state), self.tau, val_rand<=self.epsilon)
            return random.randrange(self.action_size)
        else:
            self.model.eval()
            with torch.no_grad():
                state = torch.FloatTensor(np.stack([state])).to(self.device)
                action = torch.argmax(self.model(state)).cpu().item()
                return action
    
    def preprocess_img_states(self,img):
        img_gray = np.dot(img[...,:3],[0.299,0.587,0.114])
        img_8484 = ski_transform.resize(img_gray,(84,84))
        img_rotate = np.moveaxis(img_8484,1,0)
        img_rotate = np.array(img_rotate,dtype="float64")
        return img_rotate
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_sample_in_replay_mem(self, state_buffer,action,next_state_buffer,reward,game_over):
        # S and S'

        state = torch.FloatTensor(np.stack([state_buffer]))
        next_state = torch.FloatTensor(np.stack([next_state_buffer]))

        cuda_state ,cuda_next_state = state.to(self.device), next_state.to(self.device)

        # Pred using model, max using target model, estimation
        pred_q = self.model(cuda_state).detach().cpu()
        pred_q = pred_q.view(-1,1)[action] #(1,4) -> (4,1)[action]
        
        
        if game_over:
            target_q = reward
        else:
            target_q = reward + self.gamma* torch.max(self.target_model(cuda_next_state).detach().cpu())
        
        errors = abs(pred_q - target_q).detach().item()
        #print(errors)
        self.memory.add(errors,(state,
                                torch.tensor(action),
                                next_state,
                                torch.tensor(reward),
                                torch.tensor(game_over)
                                )
                        )
        

    def train_model(self):
        # Set NN to training mode
        self.model.train()
        
        # Get samples and unpack
        sample_batches, sample_indices , imp_weights = self.memory.sample(n=self.batch_size)
        (states,actions,next_states,rewards,dones) = list(zip(*sample_batches))

        # Uses cat to stack already stacked images 
        # torch.Size([1, 10, 84, 84]) -> ([64, 10, 84, 84])
        
        batch_state = torch.cat(states,dim=0).to(self.device)
        batch_next_states = torch.cat(next_states,dim=0).to(self.device)
        
        batch_action = torch.stack(actions).unsqueeze(1).to(self.device)
        batch_rewards = torch.stack(rewards).to(self.device)
        batch_game_overs = torch.stack(dones).to(self.device)
        
        imp_weights = torch.FloatTensor(imp_weights).unsqueeze(1).to(self.device)

        # pred_q
        pred_q = self.model(batch_state).gather(1, batch_action)
        target_next_q = self.target_model(batch_next_states).detach().max(1)[0]
        # Target_MODEL
        # Target_q = R + yQ_ if not_game_over else Target_q = R
        batch_game_overs = (~batch_game_overs).long() # Convert from T/F -> 1/0, Then reversing it for computation
        target_q = (batch_rewards + (self.gamma*target_next_q)*batch_game_overs ).unsqueeze(1)
        
        # Error Handling
        assert pred_q.shape == target_q.shape, f"Shape current{pred_q.shape} != to target_q {target_q.shape}"
        assert imp_weights.shape == pred_q.shape, f"Shape of important_weights {imp_weights.shape} != pred_q{pred_q.shape}"
        
        # Update priority weights in memory CPU
        errors = torch.reshape(pred_q- target_q , (-1,)).cpu().data.numpy()
        for i in range(self.batch_size):
            self.memory.update_priority(sample_indices[i],errors[i])
        
        
        # Update NN
        self.optimizer.zero_grad()
        loss = (imp_weights * F.mse_loss(pred_q,target_q)).mean()
        loss.backward()
        self.optimizer.step()
    
    def training_loop(self, num_episodes):
        scores = []
        durations = []
        episodes_lst = []
        record = 0
        episodes = trange(num_episodes, desc="ep")
        for e in episodes:
            score = 0
            self.state_buffer.clear()
            self.next_state_buffer.clear()
            state = self.env.reset()
            state = self.preprocess_img_states(state)
            #print("What's my length, AM I RESETTED?: ",len(self.next_state_buffer))
            for t in itertools.count():
                self.env.render()
                
                action = self.get_action(self.state_buffer)

                next_state, reward, game_over, _ = self.env.step(action)
                next_state = self.preprocess_img_states(next_state)
                
                # Filling up buffer
                self.state_buffer.append(state)
                self.next_state_buffer.append(next_state)

                if len(self.next_state_buffer) == self.tau:
                    self.save_sample_in_replay_mem(self.state_buffer,action,self.next_state_buffer,reward,game_over)
                    # Train Model every timestep
                    if self.memory.tree.n_entries >= self.train_start:
                        self.train_model()
                
                score += reward
                state = next_state
                
                if game_over:
                    # Decay epsilon and update Target model After every Episode.
                    self.epsilon = max(0.01, self.epsilon*self.epsilon_decay)
                    self.update_target_model()
                    
                    
                    # Plottings
                    scores.append(score)
                    durations.append(t)
                    episodes_lst.append(e)
                    self.plot_durations(episodes_lst,scores)
                    if score >= record:
                        record = score
                        self.target_model.save()
                    episodes.set_description(desc=f"ep/t=[{e}/{num_episodes}]/{self.memory.tree.n_entries >= self.train_start} score = [{score}/{record}], epsilon=[{self.epsilon:.2f}]")
                    break

    # Maybe i should stick with images.
    def plot_durations(self, num_episodes, scores): # Update the plot at the end of each episode ## (I)
        pylab.plot(num_episodes,scores, 'b')
        pylab.savefig("./save_graph/DQNCNN_prio_replay.png")
        
        

something = Agent()
something.training_loop(100000)