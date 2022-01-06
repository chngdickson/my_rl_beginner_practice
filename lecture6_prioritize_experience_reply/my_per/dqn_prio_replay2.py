import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple,deque
import torch.nn.functional as F
import gym
import matplotlib_inline
import matplotlib.pyplot as plt
import pylab
from prioritized_memory import Memory
# For loop and while loop
from tqdm import tqdm, trange
import itertools

# Fk it, redo it.
"""
class PrioReplayBuffer:
    def __init__(self, maxlen):
        self.buffer= deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.max_beta = 1.0
        
        self.offset = 0.01
        self.alpha = 0.6
        self.alpha_decrement = 0.001
        self.min_alpha = 0.1
        
        
    def _calc_priority(self,error):
        return (np.abs(error)+self.offset)**self.alpha
    
    def add(self,experience, error):
        self.buffer.append(experience) # s,a,r,done,s_
        self.priorities.append(self._calc_priority(error)) # 
    
    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities,dtype="float32")**self.alpha # Pi^a
        sample_probabilities = scaled_priorities/sum(scaled_priorities) # Pi^a / sum P^a
        return sample_probabilities #Pr
    
    # W
    def get_importance_weight(self,probabilities):
        importance = (1/len(self.buffer) * 1/probabilities)**-self.beta # Wi = (N*Pr(i))^-beta
        importance_normalized = importance / max(importance) # W normalized
        importance_normalized = np.array(importance_normalized, dtype="float32").squeeze()
        #print(importance_normalized.shape)
        return importance_normalized
    
    def sample(self,batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer,dtype=object)[sample_indices]
        
        importance_weights = self.get_importance_weight(sample_probs[sample_indices])
        
        self.beta = min(self.max_beta,self.beta+self.beta_increment_per_sampling)
        self.alpha = max(self.min_alpha,self.alpha-self.alpha_decrement)
        
        return map(list, zip(*samples)), importance_weights , sample_indices
    
    def __len__(self):
        return len(self.buffer)
    
    # AT Right before MSELoss function
    def update_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e) + self.offset

memory = PrioReplayBuffer(100)
memory.add((0,2,3,5),0.9)
memory.add((4,4,4,4),0.4)
a,b,c=memory.sample(55,1)
print(list(zip(*a)))
"""


class DQN(nn.Module):
    def __init__(self,state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24, action_size)
        )
    def forward(self,x):
        return self.fc(x)





class Agent():
    def __init__(self, gym_env_name='CartPole-v1'):
        self.env = gym.make(gym_env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Hyper Params
        self.lr = 0.001
        # Maybe i should use RMSProp
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.99
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Model Definition
        self.model = DQN(self.state_size,self.action_size).to(self.device)
        self.target_model = DQN(self.state_size,self.action_size).to(self.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr = self.lr)
        #self.target_model = DQN(self.state_size,self.action_size).to(self.device)
        
        # Memory
        self.memory = Memory(capacity= 10000)
        #self.memory = PrioReplayBuffer(maxlen = 10000)
        self.batch_size = 64
        self.train_start = 1000 # Start training model when this has 1000
        print("ntg")
    
    def get_action(self,state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            self.model.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action = torch.argmax(self.model(state)).cpu().item()
                return action
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_sample_in_replay_mem(self, state,action,next_state,reward,game_over):
        # S and S'
        cuda_state ,cuda_next_state = torch.FloatTensor(state).to(self.device), torch.FloatTensor(next_state).to(self.device)
        
        # Pred using model, max using target model, estimation
        pred_q = self.model(cuda_state).detach().cpu()
        pred_q = pred_q[action]
        
        if game_over:
            target_q = reward
        else:
            target_q = reward + self.gamma* torch.max(self.target_model(cuda_next_state).detach().cpu())
        
        errors = abs(pred_q - target_q).detach().item()
        #print(errors)
        self.memory.add(errors,(torch.FloatTensor(state),
                                torch.tensor(action),
                                torch.FloatTensor(next_state),
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

        batch_state = torch.stack(states).to(self.device)
        batch_action = torch.stack(actions).unsqueeze(1).to(self.device)
        batch_next_states = torch.stack(next_states).to(self.device)
        batch_rewards = torch.stack(rewards).to(self.device)
        batch_game_overs = torch.stack(dones).to(self.device)
        
        imp_weights = torch.FloatTensor(imp_weights).unsqueeze(1).to(self.device)

        # pred_q
        #print(self.model(batch_state).shape)
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
            state = self.env.reset()
            
            for t in itertools.count():
                self.env.render()
                
                action = self.get_action(state)

                next_state, reward, game_over, _ = self.env.step(action)
                
                if game_over:
                    reward = -10
                    
                self.save_sample_in_replay_mem(state,action,next_state,reward,game_over)
                
                score += reward
                state = next_state
                
                # Train Model every timestep
                if self.memory.tree.n_entries >= self.train_start:
                    self.train_model()
                
                if game_over:
                    # Decay epsilon and update Target model After every Episode.
                    self.epsilon = max(0.001, self.epsilon*self.epsilon_decay)
                    self.update_target_model()
                    
                    
                    # Plottings
                    scores.append(score)
                    durations.append(t)
                    episodes_lst.append(e)
                    self.plot_durations(episodes_lst,scores)
                    if score >= record:
                        record = score
                    episodes.set_description(desc=f"ep=[{e}/{num_episodes}] score = [{score}/{record}], epsilon=[{self.epsilon:.2f}], alpha/beta = []")
                    break

    # Maybe i should stick with images.
    def plot_durations(self, num_episodes, scores): # Update the plot at the end of each episode ## (I)
        pylab.plot(num_episodes,scores, 'b')
        pylab.savefig("./save_graph/cartpole_prio_replay.png")
        
        

something = Agent()
something.training_loop(1000)

# Seems to be diverging 