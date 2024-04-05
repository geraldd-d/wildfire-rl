from env.constants import METADATA
from env.environment import World, Agent
from env.utility import layers, type_map


class ForestFire:
    def __init__(self):
        self.env = World()
        self.layers = layers
        self.action_space = len(METADATA["ACTIONS"])

    def step(self, action):
        penalty = 0
        if action <= 3:
            self.env.agents[0].move(METADATA["ACTIONS"][action])
            new_loc = self.env.agents[0].get_location()
            if not self.env.is_burnable(new_loc):
                penalty = -0.01
            if self.env.agents[0].toggle_dig:
                self.env.agents[0].dig()
        self.update()

        return [self.env.get_state(), round(self.env.get_reward()+penalty, 3), not self.env.RUNNING, {}]
    
    def reset(self):
        self.env.reset()
        return self.env.get_state()
    
    # handle fire spread, agent death
    def update(self):
        burning_cells = self.env.get_burning_cells().copy()
        for cell in burning_cells:
            # if still burning
            if self.env.reduce_fuel(cell):
                # spread fire
                neighbours = self.env.get_neighbours(cell)
                for neighbour in neighbours:
                    if self.env.is_burnable(neighbour):
                        self.env.transfer_heat(cell, neighbour)
        
        self.env.agents = [agent for agent in self.env.agents if not agent.is_dead()]
        if not self.env.agents or not burning_cells:
            self.env.RUNNING = False
