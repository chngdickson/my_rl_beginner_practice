import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import pylab
import os

from collections import namedtuple
Experience = namedtuple('Experience',('state','action','reward','next_state','done'))

class EpsilonGreedyStrategyLinear():
    def __init__(self, start, end, final_eps = None, startpoint = 50000, kneepoint=1000000, final_knee_point = None):
    # compute epsilon in epsilon-greedy algorithm by linearly decrement
        self.start = start
        self.end = end
        self.final_eps = final_eps
        self.kneepoint = kneepoint
        self.startpoint = startpoint
        self.final_knee_point = final_knee_point

    def get_exploration_rate(self, current_step):
        if current_step < self.startpoint:
            return 1.
        mid_seg = self.end + \
                   np.maximum(0, (1-self.end)-(1-self.end)/self.kneepoint * (current_step-self.startpoint))
        if not self.final_eps:
            return mid_seg
        else:
            if self.final_eps and self.final_knee_point and (current_step<self.kneepoint):
                return mid_seg
            else:
                return self.final_eps + \
                       (self.end - self.final_eps)/(self.final_knee_point - self.kneepoint)*(self.final_knee_point - current_step)




# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class Sumtree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1) # store the priorities of memory
        self.tree[capacity - 1] = 1
        self.stored = [False] * (2*capacity - 1) # indicate whether this node is used to store
        # self.cur_point = 0 
        self.length = 0 # maximum length is capacity
        self.push_count = 0

    def update_node(self, index, change):
        # update sum tree from leaf node if the priority of leaf node changed
        parent = (index-1)//2
        self.tree[parent] += change
        self.stored[parent] = True
        if parent > 0:
            self.update_node(parent, change)

    def update(self, index_memory, p):
        # update sum tree from new priority
        index = index_memory + self.capacity - 1
        change = p - self.tree[index]
        self.tree[index] = p
        self.stored[index] = True
        self.update_node(index, change)

    def get_p_total(self):
        # return total priorities
        return self.tree[0]

    def get_p_min(self):
        return min(self.tree[self.capacity-1:self.length+self.capacity-1])

    def get_by_priority(self, index, s):
        # get index of node by priority s
        left_child = index*2 + 1
        right_child = index*2 + 2
        if left_child >= self.tree.shape[0]:
            return index
        if self.stored[left_child] == False:
            return self.get_by_priority(right_child, s-self.tree[left_child])
        if self.stored[right_child] == False:
            return self.get_by_priority(left_child, s)
        if s <= self.tree[left_child]:
            return self.get_by_priority(left_child, s)
        else:
            return self.get_by_priority(right_child, s-self.tree[left_child])

    def sample(self, s):
        # sample node by priority s, return the index and priority of experience
        self.stored[self.length + self.capacity - 2] = False # cannot sample the latest state
        index = self.get_by_priority(0, s)
        return index - self.capacity + 1, self.tree[index]

    def push(self):
        # push experience, the initial priority is the maximum priority in sum tree
        index_memory = self.push_count % self.capacity
        if self.length < self.capacity:
            self.length += 1
        self.update(index_memory, np.max(self.tree[self.capacity-1 : self.capacity+self.length-1]))
        self.push_count += 1

class ReplayMemory_economy_PER():
    # Memory replay with priorited experience replay
    device = torch.device("cpu")
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_startpoint=50000, beta_kneepoint = 1000000, error_epsilon=1e-5):
        self.capacity = capacity
        self.memory = []
        self.priority_tree = Sumtree(self.capacity) # store priorities
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increase = 1/(beta_kneepoint - beta_startpoint)
        self.error_epsilon = error_epsilon
        self.push_count = 0
        self.dtype = torch.uint8

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1
        # push new state to priority tree
        self.priority_tree.push()

    def sample(self, batch_size):
        # get indices of experience by priorities
        experience_index = []
        experiences = []
        priorities = []
        segment = self.priority_tree.get_p_total()/batch_size
        self.beta = np.min([1., self.beta + self.beta_increase])
        for i in range(batch_size):
            low = segment * i
            high = segment * (i+1)
            s = random.uniform(low, high)
            index, p = self.priority_tree.sample(s)
            experience_index.append(index)
            priorities.append(p)
            # get experience from index
            experiences.append(self.memory[index])
        # compute weight
        possibilities = priorities / self.priority_tree.get_p_total()
        min_possibility = self.priority_tree.get_p_min()
        weight = np.power(self.priority_tree.length * possibilities, -self.beta)
        max_weight = np.power(self.priority_tree.length * min_possibility, -self.beta)
        weight = weight/max_weight
        weight = torch.tensor(weight[:,np.newaxis], dtype = torch.float).to(ReplayMemory_economy_PER.device)
        return experiences, experience_index, weight

    def update_priority(self, index_list, TD_error_list):
        # update priorities from TD error
        # priorities_list = np.abs(TD_error_list) + self.error_epsilon
        priorities_list = (np.abs(TD_error_list) + self.error_epsilon) ** self.alpha
        for index, priority in zip(index_list, priorities_list):
            self.priority_tree.update(index, priority)

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.memory) >= replay_start_size) and (len(self.memory) >= batch_size + 3)
    
    

matplotlib.use('TkAgg')
def reward_plot(values, moving_avg_period=100):
    pylab.clf()
    pylab.title('Training...')
    pylab.xlabel('Episode')
    pylab.ylabel('Reward') 
    pylab.plot(values,label='Reward_per_ep')
    def get_moving_average(values, moving_avg_period):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= moving_avg_period:
            moving_avg = values.unfold(dimension=0, size=moving_avg_period, step=1) \
                .mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(moving_avg_period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()
    moving_avg = get_moving_average(values, moving_avg_period)
    pylab.plot(moving_avg, label=f'Avg_reward per {moving_avg_period} episodes')
    pylab.savefig("./save_graph/DQN_reward.png")
    #plt.close('all')
    
def loss_plot(values, moving_avg_period=100):
    pylab.clf()
    pylab.title('Training...')
    pylab.xlabel('Every 4 Timestep')
    pylab.ylabel('loss') 
    pylab.plot(values,label='loss_per_ep')
    def get_moving_average(values, moving_avg_period):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= moving_avg_period:
            moving_avg = values.unfold(dimension=0, size=moving_avg_period, step=1) \
                .mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(moving_avg_period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()
    moving_avg = get_moving_average(values, moving_avg_period)
    pylab.plot(moving_avg, label=f'Avg_loss per {moving_avg_period} episodes')
    pylab.savefig("./save_graph/DQN_loss.png")
    #plt.close('all')