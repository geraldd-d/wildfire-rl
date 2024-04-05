from models.DQN import DQN, DQNAgent
from env import constants
from env import forestfire
from env import astar
from env import environment
import torch
import numpy as np
import random
from collections import deque
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import profiler
import boardVisualizer
import pygame

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    sim = forestfire.ForestFire()
    episodes = 10000
    learn_start = 1000
    state = sim.env.get_state()
    state_tensor = torch.tensor(state).float()
    state_tensor = state_tensor.permute(2, 0, 1)
    state_size = state_tensor.shape  # (3, 10, 10)
    action_size = sim.action_space  # 5
    agent = DQNAgent(state_size, action_size, sim)
    agent.collect_memories(100)
    done = False
    batch_size = constants.METADATA["DQN"]["batch_size"]
    rewards = []

    with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
        state = sim.reset()
        state_tensor = torch.tensor(state).float()
        state_tensor = state_tensor.permute(2, 0, 1)
        state = state_tensor
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = sim.step(action)
            next_state_tensor = torch.tensor(next_state).float()
            next_state_tensor = next_state_tensor.permute(2, 0, 1)
            next_state = next_state_tensor
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > learn_start:
                agent.replay(batch_size)
    print(prof.key_averages().table(sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"))

    for e in range(episodes):
        state = sim.reset()
        state_tensor = torch.tensor(state).float()
        state_tensor = state_tensor.permute(2, 0, 1)
        state = state_tensor
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = sim.step(action)
            next_state_tensor = torch.tensor(next_state).float()
            next_state_tensor = next_state_tensor.permute(2, 0, 1)    
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print("episode: {}/{}, e: {:.2}, reward: {}".format(e, episodes, agent.epsilon, total_reward))
                rewards.append(round(total_reward,3))
                break
            if len(agent.memory) > learn_start:
                agent.replay(batch_size)
        if e % constants.METADATA["DQN"]["target_update"] == 0:
            agent.update_target_model()
    plt.plot(rewards)
    plt.title('Cumulative Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.show()
    torch.save(agent.model, "model.pt")

if __name__ == "__main__":
    main()
