import torch
from torch import nn
import matplotlib.pyplot as plt
import gym
import numpy as np
from collections import deque
import torchvision.transforms as T
"""
Preprocess
 1. Convert RGB to GrayScale
 2. transform original size to 84x84
"""
env = gym.make('Breakout-v4')
state = env.reset()
plt.figure(figsize=(12,8))
plt.imshow(state)
plt.axis('off')
plt.show(block=False)


def transform_screen_data(env):
    screen = env.render('rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Use torchvision package to compose image transforms
    resize = T.Compose([
        T.ToPILImage()
        , T.Grayscale()
        , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
        , T.ToTensor()
    ])
    screen = resize(screen)
    return screen.unsqueeze(0)

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

print('original_shape',state.shape)
transformed_state = transform_screen_data(env)
print('transformed_state', transformed_state.shape)

states = deque(maxlen=4)
[states.append(transformed_state) for _ in range(4)]

state = torch.stack(list(states), dim=1).squeeze(2)
print('Each_state',state.shape)
cnn = Net(tau =4,num_actions=4)
output = cnn(state)
print('output_of_cnn',output.shape) # (1,4)


# Testing Many state
lst_of_state = []
[lst_of_state.append(state) for _ in range(10)]
batch_state = torch.cat(lst_of_state)
print('batch_state', batch_state.shape) #[10, 4, 84, 84] [batch K Height Width]
batch_output = cnn(batch_state)
print('batch_output_shape',batch_output.shape)

_, reward, done, lives = env.step(1)
print('num_lives?', lives)