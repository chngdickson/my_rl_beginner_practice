import gym
import numpy as np
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("Breakout-v4")
print(env.observation_space)
print(env.action_space)

n_steps=8192
n_ep_steps = 1e6 # 100k timesteps
vec_env = make_vec_env("Breakout-v4", n_envs=32)
# model = PPO("CnnPolicy", 
#             vec_env, 
#             n_steps=n_steps//32, 
#             verbose=1, 
#             gamma=0.99,
#             batch_size=64,
#             target_kl=0.025,
#             device="cuda").learn(int(n_ep_steps))
# model.save("ppo_breakout")

for i in range(200):
    model = PPO.load("ppo_breakout",env=vec_env,target_kl=0.025,learning_rate=0.0001)
    model.learn(int(n_ep_steps))
    model.save("ppo_breakout")
    del model