"""
Code adapted from AIMA project: https://github.com/aimacode/aima-python
"""

from collections import deque
import logging
from utils import is_in, memoize, PriorityQueue


# ##############################################################################
# Problem representation
# ##############################################################################

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def heuristic_cost(self, state):
        """Return the heuristic cost of a state. The heuristic must be
        admissible to ensure correctness. The default method returns 0."""
        return 0

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0, heuristic_cost=None, using_heuristic=False):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.heuristic_cost = heuristic_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        self.using_heuristic = using_heuristic

    def __repr__(self):
        if self.using_heuristic:
            return f"({self.depth}, {self.path_cost}, {self.action}, {self.heuristic_cost}, {self.state})"
        return f"({self.depth}, {self.path_cost}, {self.action}, {self.state})"

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem, is_repeated):
        """List the nodes reachable  (and not already created) in one step from this node."""
        logging.debug(f"Expanding {self}")
        children = [self.child_node(problem, action)
                    for action in problem.actions(self.state)]
        logging.debug(f"New successors: {', '.join(str(child) for child in children if not is_repeated(child))}")
        return children

    def child_node(self, problem, action):
        """Return the reachable node from this node and a specific action."""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state), 
            problem.heuristic_cost(next_state), using_heuristic=self.using_heuristic)
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def print_path(self):
        """Print the nodes forming the path from the root to this node."""
        path = self.path()

        logging.info(path[0])
        for node in path[1:]:
            logging.info(node.action)
            logging.info(node)


    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


# ##############################################################################
# Search algorithms
# ##############################################################################

class GraphSearch():
    """Abstract class that implements main loop of the search process and prints 
    the information related to the search such the state of the frontier of the
    explored list."""

    def __init__(self, problem):
        self.problem = problem
        self.frontier = []
        self.explored = set()
        self.node = None            # current Node in the search
        self.finished = False       # True if a goal node is found
        self.created_nodes = 0      # total number of nodes created

    def initialize(self):
        """Prepare data structures (frontier) and execute the corresponding checks."""
        raise NotImplementedError

    def execute(self):
        """Main loop of the search process."""

        # First computation of the algorithm
        self.initialize()
        if self.finished:  # the goal node is the initial node
            return True, self.node, [], set()

        # Main loop of the algorithm
        iteration = 1
        while self.frontier:
            logging.debug(f"# Iteration {iteration}")
            iteration += 1

            self.search()
            logging.debug(f"Created nodes: {self.created_nodes}")
            logging.debug(f"Frontier (size={len(self.frontier)}): {self.frontier}")
            logging.debug(f"Explored list (size={len(self.explored)}): {self.explored}")

            if self.finished:
                return True, self.node, self.frontier, self.explored

        logging.debug(f"Cannot find a solution.")
        return False, self.node, self.frontier, self.explored

    def search(self):
        """Execute the actions and checks of the specific search algorithm."""
        raise NotImplementedError

    def check(self, node):
        """Check if the node is a goal node."""
        if self.problem.goal_test(node.state):
            logging.debug(f"Solution found: {node}")
            self.finished = True
        return self.finished

    def is_repeated_node(self, node):
        """Check if the node has already been created."""
        raise NotImplementedError()


class DepthFirstGraphSearch(GraphSearch):

    def initialize(self):
        self.frontier = [(Node(self.problem.initial,using_heuristic=False))]  # Stack

    def search(self):

        # This algorithm finishes when a goal node is EXPANDED
        self.node = self.frontier.pop()
        if self.check(self.node):
            return

        self.explored.add(self.node.state)

        successors = [child for child in self.node.expand(self.problem, self.is_repeated_node)]
        self.created_nodes += len(successors)
        self.frontier.extend(child for child in successors if not self.is_repeated_node(child))

    def is_repeated_node(self, node):
        return node.state in self.explored or node in self.frontier


class BreadthFirstGraphSearch(GraphSearch):

    def initialize(self):
        self.node = Node(self.problem.initial,using_heuristic=False)
        self.check(self.node)
        self.frontier = deque([self.node])

    def search(self):
        self.node = self.frontier.popleft()
        self.explored.add(self.node.state)

        # To be consistent with the order of DepthFirst searching (MOVE > ROTATE_R > ROTATE_L), the list is reversed.
        # This is because the actions returned by the problem are [ROTATE_R, ROTATE_L, MOVE], but DepthFirst
        # uses a Stack so, when popping an element, the one from the right has more priority.
        successors = self.node.expand(self.problem, self.is_repeated_node)
        successors = list(reversed(successors))
        self.created_nodes += len(successors)

        for child in successors:
            if not self.is_repeated_node(child):

                # This algorithm finishes when a goal node is FOUND
                if self.check(child):
                    self.node = child
                    return

                self.frontier.append(child)

    def is_repeated_node(self, node):
        return node.state in self.explored or node in self.frontier


class BestFirstGraphSearch(GraphSearch):

    def __init__(self, problem, f=None):
        super().__init__(problem)
        self.f = lambda n: n.path_cost

    def initialize(self):
        self.f = memoize(self.f, 'f')

        self.node = Node(self.problem.initial, heuristic_cost=self.problem.heuristic_cost(self.problem.initial), using_heuristic=True)
        self.frontier = PriorityQueue('min', self.f)
        self.frontier.append(self.node)

    def search(self):
        self.node = self.frontier.pop()
        if self.check(self.node):
            return

        self.explored.add(self.node.state)

        successors = self.node.expand(self.problem, self.is_repeated_node)
        self.created_nodes += len(successors)

        for child in successors:

            if child.state not in self.explored and child not in self.frontier:
                self.frontier.append(child)

            elif child in self.frontier and self.f(child) < self.frontier[child]:
                del self.frontier[child]
                self.frontier.append(child)

    def is_repeated_node(self, node):
        if node in self.frontier:
            return self.f(node) >= self.frontier[node]
        return node.state in self.explored


class AstarGraphSearch(BestFirstGraphSearch):

    def __init__(self, problem):
        super().__init__(problem)
        self.f = lambda n: n.path_cost + problem.heuristic_cost(n.state)
