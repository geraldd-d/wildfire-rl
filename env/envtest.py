import unittest
from env.forestfire import ForestFire
from env.constants import METADATA
from env.utility import type_map

class TestForestFireSimulation(unittest.TestCase):
    def setUp(self):
        # Setup for each test
        self.simulation = ForestFire()

    def test_initialization(self):
        # Test environment and agent initialization
        self.assertEqual(self.simulation.env.WIDTH, METADATA["SIMULATION"]["width"])
        self.assertEqual(self.simulation.env.HEIGHT, METADATA["SIMULATION"]["height"])
        self.assertTrue(self.simulation.env.agents[0].get_location() is not None)

    def test_agent_movement(self):
        # Test agent can move without errors
        start_location = self.simulation.env.agents[0].get_location()
        self.simulation.step(0)  # Assuming 0 corresponds to a movement action
        end_location = self.simulation.env.agents[0].get_location()
        self.assertNotEqual(start_location, end_location)

    def test_fire_spreading(self):
        starting_burning_cells = self.simulation.env.get_burning_cells().copy()
        for _ in range(10):
            self.simulation.update()
        ending_burning_cells = self.simulation.env.get_burning_cells()
        self.assertNotEqual(starting_burning_cells, ending_burning_cells)

    def test_containment(self):
        # Test that the simulation ends when the fire is contained
        fire_cells = list(self.simulation.env.get_burning_cells())
        self.assertEqual(len(fire_cells), 1)
        neighbours = self.simulation.env.get_neighbours(fire_cells[0])
        for neighbour in neighbours:
            self.simulation.env.set_type(neighbour, type_map["water"])
        self.simulation.step(0)
        self.assertFalse(self.simulation.env.RUNNING)

    def test_agent_death(self):
        # Test that the simulation ends when the agent dies
        burning_cells = list(self.simulation.env.get_burning_cells())
        self.simulation.env.agents[0].set_location(burning_cells[0])
        self.simulation.step(5)  # 5 corresponds to nothing
        self.assertFalse(self.simulation.env.RUNNING)
        self.assertTrue(len(self.simulation.env.agents) == 0)
if __name__ == '__main__':
    unittest.main()