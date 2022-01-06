import torch
from torch import nn
import matplotlib.pyplot as plt
import gym
import numpy as np
from skimage import transform as ski_transform

"""
Preprocess
 1. Convert RGB to GrayScale
 2. transform original size to 84x84
"""
env = gym.make('Breakout-v0')
state = env.reset()
plt.figure(figsize=(12,8))
plt.imshow(state)
plt.axis('off')
plt.show(block=False)


# Grayscale
def scale_lumininance(img):
    """RGB GrayScale Computation reduction"""
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

s_g = scale_lumininance(state)
plt.figure(figsize=(12,8))
plt.imshow(s_g, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show(block=False)

print(f'Original Image {state.shape}')
print(f'Gray Image {s_g.shape}')

# original dim -> 84x84
s_g84 = ski_transform.resize(s_g,(84,84))
print(s_g84.shape)

plt.figure(figsize=(12,8))
plt.imshow(s_g84, cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show(block=False)

def preprocess_img_states(img):
    img_gray = np.dot(img[...,:3],[0.299,0.587,0.114])
    img_8484 = ski_transform.resize(img_gray,(84,84))
    return np.moveaxis(img_8484,1,0) # Rotate image by 90 degree ccw
plt.figure(figsize=(12,8))
plt.imshow(preprocess_img_states(state), cmap=plt.get_cmap('gray'))
plt.axis('off')
plt.show(block=False)

print(preprocess_img_states(state).shape)

# Now we have tau which tells us how the ball is moving
# It essentially stacks images

from collections import deque

tau = 4
state_buffer = deque(maxlen=tau)
next_state_buffer = deque(maxlen=tau)


# Test CNN layering with tau
cnn = nn.Sequential(
    nn.Conv2d(tau,32,kernel_size=8,stride=4),
    nn.ReLU(),
    nn.Conv2d(32,64,kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64,64,kernel_size=3,stride=1),
    nn.ReLU()
    #FC Layer = ([1,64,7,7])
    # 3136
)

tau = 4
state_buffer = deque(maxlen=tau)
next_state_buffer = deque(maxlen=tau)
input_dim = (84,84)
[state_buffer.append(np.zeros(input_dim)) for i in range(tau)]
state_t = torch.FloatTensor(np.stack([state_buffer]))
output = cnn(state_t)
print(output.shape)
print(output.flatten().shape[0])

tau = 10
cnn = nn.Sequential(
    nn.Conv2d(tau,32,kernel_size=8,stride=4),
    nn.ReLU(),
    nn.Conv2d(32,64,kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64,64,kernel_size=3,stride=1),
    nn.ReLU()
    #FC Layer = ([1,64,7,7])
    # 3136
)
state_buffer = deque(maxlen=tau)
next_state_buffer = deque(maxlen=tau)
input_dim = (84,84)
[state_buffer.append(preprocess_img_states(state)) for i in range(tau)]
state_t = torch.FloatTensor(np.stack([state_buffer]))
print('state_tensor',state_t.shape)
output = cnn(state_t)
state_t_batch = []
[state_t_batch.append((state_t)) for i in range(64)]
state_batch = torch.cat(state_t_batch,dim=0)
print('batch',state_batch.shape)

print(state_t.shape)
print(output.shape)
print(output.flatten().shape[0])

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

cnn = Net(tau,4)
output = cnn(state_batch)
print(output.shape)