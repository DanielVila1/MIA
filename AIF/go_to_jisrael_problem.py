import argparse
import logging
import sys

from search import Problem, BreadthFirstGraphSearch, DepthFirstGraphSearch, AstarGraphSearch
from map import Map


class GoToJisraelProblem(Problem):

    def __init__(self, problem_file, h=0):
        self.map = Map(problem_file)
        super().__init__(State(*self.map.initial), State(*self.map.goal))
        
        self.table = self.map.table
        self.h = eval(f"self.h{h}")

    def actions(self, state):

        # Check if can move
        can_move = True

        if state[2] == 0 and state[0] == 0:                     # /\
            logging.debug(f"Cannot go up.")
            can_move = False
        
        elif state[2] == 1 and state[1] == self.map.size_y - 1:   # >
            logging.debug("Cannot go right.")
            can_move = False

        elif state[2] == 2 and state[0] == self.map.size_x - 1:   # \/
            logging.debug("Cannot go down.")
            can_move = False

        elif state[2] == 3 and state[1] == 0:                   # <
            logging.debug("Cannot go left.")
            can_move = False

        # Return possible actions
        if can_move:
            return ["ROTATE_L", "ROTATE_R", "MOVE"]
        else:
            return ["ROTATE_L", "ROTATE_R"]

    def result(self, state, action):
        x, y, orientation = state

        if action == "ROTATE_L":    return State(x, y, (orientation - 1) % 4)
        elif action == "ROTATE_R":  return State(x, y, (orientation + 1) % 4)
        elif action == "MOVE":
            if orientation == 0:    return State(x - 1, y, 0)
            elif orientation == 1:  return State(x, y + 1, 1)
            elif orientation == 2:  return State(x + 1, y, 2)
            elif orientation == 3:  return State(x, y - 1, 3)

    def goal_test(self, state):
        x1, y1, _ = state
        x2, y2, _ = self.goal
        return x1 == x2 and y1 == y2

    def path_cost(self, c, state1, action, state2):
        if action == "MOVE":
            x, y, _ = state2
            return c + self.table[x][y]
        else:
            return c + 1

    def heuristic_cost(self, state):
        return self.h(state)

    def h0(self, state):
        """No heuristic function."""
        return super().heuristic_cost(state)

    def h1(self, state):
        """Manhattan distance heuristic function."""
        return abs(state.x - self.goal[0]) + abs(state.y - self.goal[1])

    def h2(self, state):
        """Manhattan distance + rotations heuristic function."""

        # Manhattan distance
        goal_x, goal_y, _ = self.goal
        manhattan_distance = abs(state.x - goal_x) + abs(state.y - goal_y)

        # rotations needed to align one orientation, x, with the other, y
        r = lambda x,y : max(abs(x - y) % 2, abs(x - y) % 3)

        north_south = (state.x < goal_x) * 2  # 0: North, 2: South
        ver_rotations = r(state.orientation, north_south) * (state.x != goal_x)

        west_east = (state.y > goal_y) * 2 + 1  # 1: East, 3: West
        hor_rotations = r(state.orientation, west_east) * (state.y != goal_y)

        total_rotations = min(ver_rotations + hor_rotations, 2)
        return manhattan_distance + total_rotations


class State(tuple):

    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

    def __new__(cls, x, y, orientation):
        return super(State, cls).__new__(cls, tuple((x, y, orientation)))

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.format_orientation(self.orientation)})"

    def format_orientation(self, orientation):
        if      orientation == 0: return 'North'
        elif    orientation == 1: return 'East'
        elif    orientation == 2: return 'South'
        elif    orientation == 3: return 'West'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve the Jisrael problem.')
    parser.add_argument('file', type=str, help='file with the terrain, the initial and the goal positions')
    parser.add_argument('--algorithm', type=str, default="BreadthFirst", help='[DepthFirst, BreadthFirst, Astar]')
    parser.add_argument('--h', type=int, default=0, help='0: no heuristic function, 1: Manhattan distance, 2: Manhattan distance + rotations')
    parser.add_argument('--debug', default=False, action='store_true', help='true to print the search process')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if args.debug else logging.INFO, format='%(message)s')

    # Load problem
    problem = GoToJisraelProblem(args.file, args.h)

    # Solve problem with an specific algorithm
    search = eval(f"{args.algorithm}GraphSearch(problem)")
    found, node, frontier, explored = search.execute()

    logging.info("#########################################")
    logging.info("# PATH")
    logging.info("#########################################")

    if not found:
        logging.info("The goal node could not be found. The path to the last examined node is shown.")
    node.print_path()

    logging.info("#########################################")
    logging.info("# ANALYTICS")
    logging.info("#########################################")
    logging.info(f"Path depth: {node.depth}")
    logging.info(f"Path cost: {node.path_cost}")
    logging.info(f"Frontier size: {len(frontier)}")
    logging.info(f"Explored list size: {len(explored)}")
