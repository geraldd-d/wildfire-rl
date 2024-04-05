import numpy as np
from colour import Color
SIZE = 10 # Size of Simulation
SPEED = 1 # Speed of Agent

def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Environment Attributes
METADATA = {
    # Rewards
    "REWARDS": {
        "default": 0,
        "death": -1,
        "success": 1,
        "elevation_penalty": 10
    },
    # Actions
    "ACTIONS": {
        0: np.array([0, -1]), # N
        1: np.array([0, 1]), # S
        2: np.array([-1, 0]), # W
        3: np.array([1, 0]), # E
    },
    # Simulation
    "SIMULATION": {
        "width": SIZE,
        "height": SIZE,
        "speed": SPEED,
        "wind": [0.15, (2,2)], # Strength, Direction Vector
        "rivers" : False,
        "grass" : {
            "heat": 0.3,
            "threshold": 3,
            "fuel": 20
        },
        "COLOR": {
            "fire": grayscale(Color("red")),
            "agent": grayscale(Color("pink")),
            "grass": grayscale(Color("green")),
            "water": grayscale(Color("blue")),
            "dirt": grayscale(Color("brown")),
            "burnt": grayscale(Color("black"))
        },
    },
    # DQN
    "DQN": {
        "memory_size": 20000,
        "max_epsilon": 0.95,
        "min_epsilon": 0.01,
        "epsilon_decay": 0.0000035,
        "gamma" : 0.999,
        "batch_size" : 128,
        "alpha" : 0.0005,
        "target_update" : 50,
    },

}