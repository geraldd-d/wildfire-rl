import math
import numpy as np
import random as r
import copy
from env.constants import METADATA
from env.astar import astar
from env.utility import layers, type_map
from env.utility import get_agent_location, get_fire_location, circle_points
import matplotlib.pyplot as plt
from env.utility import PerlinNoiseFactory


WIDTH = METADATA["SIMULATION"]["width"]
HEIGHT = METADATA["SIMULATION"]["height"]
WIND = METADATA["SIMULATION"]["wind"]

"""
    Create a map of the environment
    Structure of environment = Width x Height x (Attributes)
    i.e Width x Height x (Grid Type, Threshold, Heat, Fire Mobility, Agent Mobility, Agent Present)
"""

def create_map():
    PNF = PerlinNoiseFactory(2, octaves = 3)
    type = np.zeros((WIDTH, HEIGHT))
    threshold = np.full((WIDTH, HEIGHT), METADATA["SIMULATION"]["grass"]["threshold"])
    heat = np.full((WIDTH, HEIGHT), METADATA["SIMULATION"]["grass"]["heat"])
    fire_mobility = np.ones((WIDTH, HEIGHT))
    agent_mobility = np.ones((WIDTH, HEIGHT))
    agent_present = np.zeros((WIDTH, HEIGHT))
    elevation = np.zeros((WIDTH, HEIGHT))
    for i in range(WIDTH):
        for j in range(HEIGHT):
            elevation[i, j] = PNF(i/WIDTH, j/HEIGHT)
    fuel = np.full((WIDTH, HEIGHT), 20)
    color = np.full((WIDTH, HEIGHT), METADATA["SIMULATION"]["COLOR"]['grass'])
    
    return np.dstack((type, threshold, heat, fire_mobility, agent_mobility, agent_present, elevation, fuel, color))

# reset all layers to default, random river near the middle
def reset_map(env, make_river=False):
    env[:, :, layers["type"]] = type_map["grass"]
    env[:, :, layers["heat"]] = METADATA["SIMULATION"]["grass"]["heat"]
    env[:, :, layers["threshold"]] = METADATA["SIMULATION"]["grass"]["threshold"]
    env[:, :, layers["fire_mobility"]] = 1
    env[:, :, layers["agent_mobility"]] = 1
    env[:, :, layers["agent_present"]] = 0
    env[:, :, layers["color"]] = METADATA["SIMULATION"]["COLOR"]['grass']
    env[:, :, layers["fuel"]] = METADATA["SIMULATION"]["grass"]["fuel"]
    if make_river:

        # River distance from border
        d = range(1,4)
        (fx, fy) = get_fire_location(WIDTH, HEIGHT)

        # River starts anywhere on x-axis
        river_x = r.choice((range(WIDTH)))

        # River starts near top, away from border
        river_y = r.choice(d)

        # Next river grid is randomly selected downwards, and must not be on a fire grid or outside bounds
        while river_y < (HEIGHT - r.choice(d)):
            env[river_x, river_y, layers["type"]] = type_map["water"]
            env[river_x, river_y, layers["fire_mobility"]] = np.inf
            env[river_x, river_y, layers["agent_mobility"]] = np.inf
            env[river_x, river_y, layers["color"]] = METADATA["SIMULATION"]["COLOR"]['water']
            new_river_y = river_y + 1
            new_river_x = (river_x + r.choice([-1, 0, 1]))
            while not r.choice(d) <= new_river_x < (WIDTH - r.choice(d)) and not (new_river_x, new_river_y) == (fx, fy):
                new_river_x = (river_x + r.choice([-1, 0, 1]))
        
            (river_x, river_y) = (new_river_x, new_river_y)

class World:
    def __init__(self):
        self.env = create_map()
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.DEPTH = self.get_state().shape[2]
        self.wind_speed = WIND[0]
        self.agents = []
        self.burning_cells = set()
        self.fire_at_border = False
        self.RUNNING = True
        self.initial_agent_location = get_agent_location(self.WIDTH, self.HEIGHT)
        self.reset()
    
    def reset(self):
        self.RUNNING = True
        reset_map(self.env)
        self.burning_cells = set()
        self.set_fire_to(get_fire_location(self.WIDTH, self.HEIGHT))
        self.agents = [
            Agent(self, self.initial_agent_location)
        ]
        self.fire_at_border = False
    
    # GETTERS 
        
    def get_state(self):
            return np.dstack((self.env[:, :, layers['agent_present']],
                            self.env[:, :, layers['type']] == type_map['fire'],
                            self.env[:, :, layers['type']] == type_map['grass'],
                            self.env[:, :, layers['fire_mobility']] != np.inf))
        
    def is_burnable(self, location):
        x, y = location
        return self.env[x, y, layers['type']] == type_map['grass']
    
    def is_fire(self, location):
        x, y = location
        return self.env[x, y, layers['type']] == type_map['fire']
    
    def get_neighbours(self, cell):
        x, y = cell
        neighbours = []
        for pos in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            if 0 <= pos[0] < self.WIDTH and 0 <= pos[1] < self.HEIGHT and self.is_burnable(pos):
                neighbours.append(pos)
        return neighbours
    
    def get_reward(self):
        # check if fire has path to border
        """
        collect list of border points, and list of burning cells
        for each burning cell, check if it has a path to any border cell. when out of border cells, reset the border cells list and pop burning cell
        when no burning cells left, fire contained.
        """
        if len(self.burning_cells):
            burning = self.burning_cells.copy()
            grid = self.env[:, :, layers["fire_mobility"]]
            border = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
            start = burning.pop()
            end = border.pop()
            path = astar(grid, start, end)
            while not path:
                # no borders left to path to, and no burning cells to path with
                if not len(border) and not len(burning):
                    self.RUNNING = False
                    grass_cells = np.count_nonzero(self.env[:, :, layers["type"]] == type_map["grass"])
                    print("Remaining grass: ", grass_cells)
                    print("Dirt: ", np.count_nonzero(self.env[:, :, layers["type"]] == type_map["dirt"]))
                    print("Fire Contained")
                    return (not self.fire_at_border)*METADATA["REWARDS"]["success"] + grass_cells/(WIDTH*HEIGHT) * METADATA["REWARDS"]["success"]
                # no borders left to path to, but burning cells left to path with
                elif not len(border):
                    border = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
                    start = burning.pop()
                    end = border.pop()
                    path = astar(grid, start, end)
                # borders left to path to
                else:
                    end = border.pop()
                    path = astar(grid, start, end)
        if not self.agents:
            self.RUNNING = False
            return METADATA["REWARDS"]["death"]
        
        # if fire is gone, return reward on grass cells
        if not self.burning_cells:
            self.RUNNING = False
            grass_cells = np.count_nonzero(self.env[:, :, layers["type"]] == type_map["grass"])
            print("Remaining grass: ", grass_cells)
            print("Dirt: ", np.count_nonzero(self.env[:, :, layers["type"]] == type_map["dirt"]))
            return (not self.fire_at_border)*METADATA["REWARDS"]["success"] + grass_cells/(WIDTH*HEIGHT) * METADATA["REWARDS"]["success"]
        
        return METADATA["REWARDS"]["default"]
    
    # get wind angle
    """
    cell a -------->  cell b
    \
     \
      \
    wind direction
    find angle of wind direction from vector ab
    let east be 0 rad, north be pi/2 rad, west be pi rad, south be 3pi/2 rad
    """
    def get_wind_angle(self, source, target):
        sx, sy = source
        tx, ty = target
        vector = (tx - sx, ty - sy)
        w_vector = WIND[1]
        angle = np.arccos(np.dot(vector, w_vector)/(np.linalg.norm(vector)*np.linalg.norm(w_vector)))
        return angle
    
    def get_burning_cells(self):
        return self.burning_cells
    
    # get elevation change
    def get_elevation_change(self, source, target):
        sx, sy = source
        tx, ty = target
        return abs(self.env[tx, ty, layers["elevation"]] - self.env[sx, sy, layers["elevation"]])

    # get grid type
    def get_type(self, location):
        return self.env[location[0], location[1], layers["type"]]

    # get agent mobility
    def get_agent_mobility(self, location):
        return self.env[location[0], location[1], layers["agent_mobility"]]

    # SETTERS

    def set_fire_to(self, location):
        self.env[location[0], location[1], layers['type']] = type_map['fire']
        if self.env[location[0], location[1], layers['heat']] < self.env[location[0], location[1], layers['threshold']]:
            self.env[location[0], location[1], layers['heat']] = self.env[location[0], location[1], layers['threshold']] + 0.1
        self.burning_cells.add(location)
        self.env[location[0], location[1], layers['color']] = METADATA["SIMULATION"]["COLOR"]['fire']
        if location[0] == 0 or location[0] == self.WIDTH - 1 or location[1] == 0 or location[1] == self.HEIGHT - 1:
            self.fire_at_border = True
    
    # set fire mobility
    def set_fire_mobility(self, location, mobility):
        self.env[location[0], location[1], layers["fire_mobility"]] = mobility

    # set agent mobility
    def set_agent_mobility(self, location, mobility):
        self.env[location[0], location[1], layers["agent_mobility"]] = mobility

    # set agent present
    def set_agent_present(self, location, present):
        self.env[location[0], location[1], layers["agent_present"]] = present

    # set grid type
    def set_type(self, location, type):
        self.env[location[0], location[1], layers["type"]] = type
        self.env[location[0], location[1], layers["color"]] = METADATA["SIMULATION"]["COLOR"][type_map[type]]

    # FIRE SPREAD

    # transfer of heat
    def transfer_heat(self, source, target):
        sx, sy = source
        tx, ty = target

        angle = self.get_wind_angle(source, target)

        # depending on direction, wind multiplier from 0.5 to 1.5 times of wind speed
        heat_transfer = self.wind_speed * self.env[sx, sy, layers["heat"]] * (0.5* np.cos(angle) + 1)

        self.env[tx, ty, layers["heat"]] += heat_transfer
        if self.env[tx, ty, layers["heat"]] > self.env[tx, ty, layers["threshold"]]:
            self.set_fire_to((tx, ty))
    
    # reduce fuel of cell
    def reduce_fuel(self, cell):
        x, y = cell
        self.env[x, y, layers["fuel"]] -= 1
        if self.env[x, y, layers["fuel"]] <= 0:
            self.burning_cells.remove(cell)
            self.env[x, y, layers["type"]] = type_map["burnt"]
            self.env[x, y, layers["fire_mobility"]] = np.inf
            self.env[x, y, layers["color"]] = METADATA["SIMULATION"]["COLOR"]['burnt']
            return False # cell burnt out
        return True
    
    def render(self):
        plt.imshow(self.env[:, :, layers['color']], cmap='gray', interpolation='nearest')
        plt.colorbar()  # Optionally add a colorbar to show the mapping from grayscale values to colors
        plt.show()
    
    
class Agent:


    def __init__(self, world, location):
        self.world = world
        self.location = location
        self.dead = False
        self.toggle_dig = True
        self.dig() # start on a safe square

    def get_location(self):
        return self.location
    
    def is_dead(self):
        if self.dead or self.world.is_fire(self.location):
            self.dead = True
            self.world.set_agent_present(self.location, 0)
            return True
        return False
    
    def dig(self):
        if self.world.get_type(self.location) == type_map["grass"]:
            self.world.set_type(self.location, type_map["dirt"])
            self.world.set_fire_mobility(self.location, np.inf)

    def move(self, direction):
        x, y = self.location
        new_x, new_y = x + direction[0], y + direction[1]
        if 0 <= new_x < self.world.WIDTH and 0 <= new_y < self.world.HEIGHT and self.world.get_agent_mobility((new_x, new_y)) != np.inf:
            self.world.set_agent_present(self.location, 0)
            self.world.set_agent_present((new_x, new_y), 1)
            self.location = (new_x, new_y)
            if self.world.is_fire(self.location):
                self.dead = True
                self.world.set_agent_present(self.location, 0)
    
    def set_location(self, location):
        self.location = location
        self.world.set_agent_present(self.location, 1)
    
    def fire_in_direction(self, direction):
        x, y = self.location + direction
        return self.world.is_fire((x,y)) if 0 <= x < self.world.WIDTH and 0 <= y < self.world.HEIGHT else False
