from collections import namedtuple
from model import Dueling_DQN_2016_Modified
from ArariEnvManager import AtariEnvManager
from agent import Agent
from utils import ReplayMemory_economy_PER, EpsilonGreedyStrategyLinear, \
reward_plot, loss_plot, Experience
import torch
import numpy as np
import torch.nn.functional as F


gamma = 0.99
num_images_stacked = 4
env_mng = AtariEnvManager(game_env='Breakout-v4',
                          num_images_stacked=num_images_stacked,
                          is_use_additional_ending_criterion=True)
strategy = EpsilonGreedyStrategyLinear(start=1.0,end=0.1,
                                       final_eps=0.01,startpoint=50_000,
                                       kneepoint=1_000_000,
                                       final_knee_point=22_000_000)
agent = Agent(strategy, env_mng.n_actions)
memory = ReplayMemory_economy_PER(capacity=250_000)

policy_DQN = Dueling_DQN_2016_Modified(output_file_name="policy",
                                       n_k_stacked=num_images_stacked,
                                       n_actions=env_mng.n_actions,
                                       lr=0.001
                                       )
target_DQN = Dueling_DQN_2016_Modified(output_file_name="target",
                                       n_k_stacked=num_images_stacked,
                                       n_actions=env_mng.n_actions,
                                       lr=0.001
                                       )


def update_target_nn():
    target_DQN.load_state_dict(policy_DQN.state_dict())
    target_DQN.eval()
update_target_nn()
env_mng.print_action_meanings()


def calc_error_and_store_in_mem(state,action, reward, next_state,done):
    # device = policy_DQN.device
    # with torch.no_grad():
    #     next_q_value = target_DQN(next_state).detach()
    #     next_q_value[done] = 0.0
    
    # target_q = reward.to(device) + gamma * torch.max(next_q_value).view(1)
    
    # pred_q = (policy_DQN(state).gather(dim=1,index=action.unsqueeze(-1))).detach()
    # pred_q = pred_q.view(1)
    
    # #print(target_q.shape, pred_q.shape,reward.shape)
    # error = abs(pred_q.cpu()-target_q.cpu())
    # memory.add(error,Experience( 
    #                             state.cpu(),
    #                             action,
    #                             reward.cpu(),
    #                             next_state.cpu(),
    #                             torch.tensor(done))
    #            )
    memory.push(Experience(state.cpu(), action, reward.cpu(), next_state.cpu(),torch.tensor(done)))


def update_policy_with_PER():
    batch_size = 32
    device = policy_DQN.device
    experiences, experience_indices, imp_weights = memory.sample(batch_size=batch_size)
    batch = Experience(*zip(*experiences))
    
    imp_weights = torch.FloatTensor(imp_weights).reshape(batch_size).to(device)
    batch_states = torch.cat(batch.state).to(device)
    batch_actions = torch.stack(batch.action).to(device)
    batch_rewards = torch.stack(batch.reward).to(device)
    batch_next_states = torch.cat(batch.next_state).to(device)
    batch_dones = torch.stack(batch.done).to(device)
    
    """CAlculate predQ , target_qs(no_grad)"""
    pred_qs = policy_DQN(batch_states).gather(dim=1, index=batch_actions).reshape(batch_size)
    with torch.no_grad():
        next_qs = target_DQN(batch_next_states).detach().max(1)[0]
        next_qs[batch_dones]= 0.0
        target_qs = batch_rewards.reshape(batch_size) + gamma*next_qs
    
    """Update memory and policy NN"""
    policy_DQN.optimizer.zero_grad()
    TD_errors = torch.abs(pred_qs-target_qs).detach().cpu()
    memory.update_priority(experience_indices,TD_errors)
    
    loss = torch.mean(imp_weights.detach()*F.mse_loss(pred_qs,target_qs))
    #print(loss)
    loss.backward()
    policy_DQN.optimizer.step()
    
    assert pred_qs.shape == target_qs.shape, f"Shape current {pred_qs.shape} != to target_q {target_qs.shape}"
    return loss.cpu()

if __name__ == "__main__":
    avg_loss_lst = []
    reward_hist = []
    max_reward = -5
    for episode in range(150000):
        env_mng.reset()
        state = env_mng.get_state()
        total_reward = 0
        
        while True:
            env_mng.render()
            
            # S, A , R
            action = agent.select_action(state,policy_net=policy_DQN)
            #print(type(action))
            reward = torch.sign(env_mng.step(action))
            
            total_reward +=reward
            # S'
            next_state = env_mng.get_state()
            
            # TODO: memory.add
            calc_error_and_store_in_mem(state,
                                        action,
                                        reward,
                                        next_state,
                                        env_mng.done)
            # Update Policy neural network Every 4 iterations
            if agent.current_step % 4 == 0 and memory.capacity>=200:
                avg_loss = update_policy_with_PER()
                avg_loss_lst.append(avg_loss)
            
            state = next_state
            if agent.current_step %2500 ==0:
                update_target_nn()
                target_DQN.save_checkpoint()
                
            if env_mng.done:
                if total_reward > max_reward:
                    max_reward=total_reward
                    policy_DQN.save_checkpoint()
                reward_hist.append(total_reward)
                reward_plot(reward_hist,100)
                loss_plot(avg_loss_lst,100)
                break
        
        