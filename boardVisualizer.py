from env import constants
from env import forestfire
from env import environment
from env import utility
import numpy as np
import random
import pygame

class BoardVisualizer:
    def __init__(self, sim, cell_size=20):
        self.sim = sim
        self.env = sim.env
        self.cell_size = cell_size
        self.colors = utility.color_map
        pygame.init()
        self.type_layer = self.env.get_type_layer()
        self.screen = pygame.display.set_mode((self.type_layer.shape[0] * cell_size, self.type_layer.shape[1] * cell_size))
        self.clock = pygame.time.Clock()
    
    def draw_grid(self):
        for x in range(self.type_layer.shape[0]):
            for y in range(self.type_layer.shape[1]):
                if len(self.env.agents) and self.env.agents[0].get_location() == (x, y):
                    color = self.colors["agent"]
                else:
                    cell_type = utility.type_map[self.env.get_type_layer()[x][y]]
                    color = self.colors.get(cell_type, (255, 255, 255))  # Default to white if type is unknown
                pygame.draw.rect(self.screen, color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
