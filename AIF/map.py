import argparse
import logging
import os
from random import randint


class Map:
    """Implements input and output operations to read a map from a file,
    generate a new one or save it in file."""

    def __init__(self, file=None):
        if file is not None:
            self.load(file)

    def __repr__(self):
        return "\n\t".join([str(x) for x in self.table])

    def load(self, file):
        """Load the map from a file."""

        if not os.path.isfile(file):
            logging.error(f"Cannot load file from '{file}'")
            return 

        # Load the data
        with open(file, 'r', encoding="utf-8-sig") as f:
            lines = " ".join(f.readlines())
            elements = [int(x) for x in lines.split()]

            # Load the size
            self.size_x, self.size_y  = elements[0:2]

            # Load the table as a list of lists
            self.table = []
            for x in range(self.size_x):
                r1 = 2 + self.size_y * x
                r2 = 2 + self.size_y * (x + 1)
                self.table.append(elements[r1:r2])

            # Load the initial and goal positions
            self.initial = tuple(elements[2 + self.size_x * self.size_y:5 + self.size_x * self.size_y])
            self.goal = tuple(elements[5 + self.size_x * self.size_y:8 + self.size_x * self.size_y])

    def save(self, file):
        """Save the map in a file."""

        with open(file, 'w', encoding="utf-8-sig") as f:

            # Write dimensions
            f.write(f"{self.size_x} {self.size_y}\n")

            # Write terrain
            table_str = str(self).replace('\t', "").replace('[', "").replace(']', "").replace(',', "")
            f.write(table_str)

            # Write initial and goal positions
            f.write(f"\n{self.initial[0]} {self.initial[1]} 0")
            f.write(f"\n{self.goal[0]} {self.goal[1]} 0")

    def generate(self, size_x, size_y, initial=None, goal=None, min_cost=1, max_cost=5, output=None):
        """Generate a new random map with the parameters given."""

        # Set the size of the table
        self.size_x, self.size_y = size_x, size_y
        
        # Establish the initial and goal positions
        self.initial = initial
        if self.initial is None:
            self.initial = (randint(0, size_x - 1), randint(0, size_y - 1))

        self.goal = goal
        if self.goal is None:
            self.goal = (randint(0, size_x - 1), randint(0, size_y - 1))

        # Generate the terrain
        self.table = [[randint(min_cost, max_cost) for y in range(size_y)] for x in range(size_x)]

        # Save the map in a file
        if output is not None:
            self.save(output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a new map.')
    parser.add_argument('--dims', type=int, nargs='+', default=[3,3], help='width and height of the map (default 3 3)')
    parser.add_argument('--initial', type=int, nargs='+', default=None, help='initial position (default random)')
    parser.add_argument('--goal', type=int, nargs='+', default=None, help='goal position (default random)')   
    parser.add_argument('--min-cost', type=int, default=1, help='minimum cost of the terrain (default 1)') 
    parser.add_argument('--max-cost', type=int, default=5, help='maximum cost of the terrain (default 5)') 
    parser.add_argument('--output', type=str, default='./output.txt', help='file to save the map (default ./output.txt') 
    args = parser.parse_args()

    map = Map()
    map.generate(*args.dims, args.initial, args.goal, args.min_cost, args.max_cost, args.output)