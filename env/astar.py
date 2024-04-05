import numpy as np
from env.constants import SIZE
import time
# astar for fire pathfinding to border (true if can reach border)


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = self.h = self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    

def astar(env, start, end):
    WIDTH = HEIGHT = SIZE
    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list = []
    closed_list = []
    open_list.append(start_node)

    while open_list:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_pos = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_pos[0] > (WIDTH - 1) or node_pos[0] < 0 or node_pos[1] > (HEIGHT - 1) or node_pos[1] < 0:
                continue
            if env[node_pos[0]][node_pos[1]] == np.inf:
                continue
            new_node = Node(current_node, node_pos)
            children.append(new_node)
        for child in children:
            if child in closed_list:
                continue
            child.g = current_node.g + 1
            # heuristic: manhattan (no diag movement)
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            if child in open_list:
                continue
            open_list.append(child)
        
    return None