import numpy as np
from colour import Color
SIZE = 40 # Size of Simulation

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
        "wind": [0.15, (2,2)], # Strength, Direction Vector
        "rivers" : False,
        "grass" : {
            "heat": 0.3,
            "threshold": 3,
            "fuel": 20
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