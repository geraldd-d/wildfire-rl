import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from env.constants import METADATA 
from env.forestfire import ForestFire
from boardVisualizer import BoardVisualizer
import pygame

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(input_size[0], 32, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(),
            # Convolutional Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            # Convolutional Layer 3
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.LeakyReLU()
        )
        conv_out_size = self._get_conv_out(input_size)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),  # 0.25 dropout
            nn.Linear(512, output_size)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.unsqueeze(0) 
        conv_out = self.conv_layers(x).view(x.size()[0], -1)  # Flatten the output for the fully connected layer
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_size, action_size, sim):
        self.METADATA = METADATA
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.METADATA["DQN"]["memory_size"])
        self.gamma = self.METADATA["DQN"]["gamma"]
        self.epsilon = self.METADATA["DQN"]["max_epsilon"]
        self.epsilon_min = self.METADATA["DQN"]["min_epsilon"]
        self.epsilon_decay = self.METADATA["DQN"]["epsilon_decay"]
        self.learning_rate = self.METADATA["DQN"]["alpha"]
        self.batch_size = self.METADATA["DQN"]["batch_size"]
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.sim = sim
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(state)
        act_values = act_values.cpu()
        return np.argmax(act_values.numpy()[0])
    
    def replay(self, batch_size):
        """Trains the network using randomly sampled experiences from the memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)))
            target_f = self.model(state)
            target_f[0][action] = target
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_decay)

    def update_target_model(self):
        """Updates the target model weights to match the current model's weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def collect_memories(self, num_of_episodes=100, perform_baseline=False):
        bv = BoardVisualizer(self.sim)
        if not num_of_episodes:
            return
        # Wipe internal memory
        self.memory = deque(maxlen=self.METADATA["DQN"]["memory_size"])
        success_count = 0
        episode = 0

        # While memory is not filled up
        while True:
            total_reward = 0
            memories = list()
            done = False

            state = self.sim.reset()
            state_tensor = torch.tensor(state).float()
            state_tensor = state_tensor.permute(2, 0, 1)
            state = state_tensor

            while not done:
                # Choose an action
                action = self.choose_randomwalk_action()

                # Observe sprime and reward
                sprime, reward, done, _ = self.sim.step(action)
                sprime_state_tensor = torch.tensor(sprime).float()
                sprime = sprime_state_tensor.permute(2, 0, 1)

                # Collect memories
                memories.append((state, action, reward, sprime, done))
                state = sprime
                total_reward += reward
                bv.screen.fill((0, 0, 0))  # Fill the screen with black
                bv.draw_grid()
                pygame.display.flip()
                bv.clock.tick(3)
                # Only if we contained the fire, we collect the memories
                if not perform_baseline and reward >= self.METADATA["REWARDS"]["success"]:
                    success_count += 1
                    # Store successful experience in memory
                    for state, action, reward, sprime, done in memories:
                        self.remember(state, action, reward, sprime, done)
                    # Set done to true
                    done = True
                    # Collect logging info and return
                    if success_count == num_of_episodes:
                        return
                episode += 1
        pygame.quit()

    def choose_randomwalk_action(self, avoid_fire=True):
        # It can happen in SARSA to ask for an action when agent has died.
        # However, that action is never looked at and is irrelevant
        if not self.sim.env.agents:
            return 0

        width, height = self.sim.env.WIDTH, self.sim.env.HEIGHT
        agent_x, agent_y = self.sim.env.agents[0].get_location()
        mid_x, mid_y = (int(width / 2), int(height / 2))

        # Loop to try to avoid choosing actions that lead to death
        count = 0
        while True:
            # The chosen action should always make the agent go around the fire
            if agent_x >= mid_x and agent_y > mid_y:
                possible_actions = [1, 2]
            if agent_x > mid_x and agent_y <= mid_y:
                possible_actions = [1, 3]
            if agent_x <= mid_x and agent_y < mid_y:
                possible_actions = [0, 3]
            if agent_x < mid_x and agent_y >= mid_y:
                possible_actions = [0, 2]

            # Choose randomly from valid actions
            action = np.random.choice(possible_actions)

            if not avoid_fire:
                break

            # Break when it is a safe move or when we have tried too often
            fire_at_loc = self.sim.env.agents[0].fire_in_direction(METADATA["ACTIONS"][action])
            if not fire_at_loc or count > 10:
                break
            count += 1

        return action