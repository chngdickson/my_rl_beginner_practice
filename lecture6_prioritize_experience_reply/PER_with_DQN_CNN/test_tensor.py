import torch
import gym
tensor_1_4 = torch.FloatTensor([[-0.7945, -0.8377, -1.3094,  1.0669]])
print(tensor_1_4.shape) #torch.Size([1, 4])
print(tensor_1_4.view(4,1)[0]) # Get -0.

env = gym.make('Breakout-v0')
env.reset()
print(env.reward_range)
next_state, reward, game_over, _ = env.step(0)
print(_)