from pygame import K_DOWN, K_UP, K_RIGHT, K_LEFT
import random
from tetris.constants import rows, cols
import numpy as np


class Tile(object):
    """Tile structures: I, S, Z, O, T, J, L.
    Use the following matrix to define each tile

     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    """

    def __init__(self, *args, **kwargs):
        self.x = cols // 2  # TODO: use random to start x position
        self.y = 0
        self.name = random.choice(["I", "S", "Z", "O", "T", "J", "L"])
        self.data = getattr(self, self.name)
        self.rotate_order = 0
        self.locked = False
        self.event_control_map = {
            K_UP: self.rotate,
            K_DOWN: self.down,
            K_LEFT: self.left,
            K_RIGHT: self.right,
        }

    def caliberation(self):
        x_max = np.max(self.pos[:, 0])
        x_min = np.min(self.pos[:, 0])
        y_max = np.max(self.pos[:, 1])
        if x_max > (cols - 1):
            self.x -= x_max - cols + 1
        elif x_min < 0:
            self.x += 0 - x_min
        if y_max > (rows - 1):
            self.y -= y_max - rows + 1
            self.locked = True

    @property
    def pos(self):
        """Position of grids."""
        structure = self.data.get("structure")
        structure = structure[self.rotate_order % len(structure)]
        return np.array(
            [
                [self.x + (num % 4), self.y + (num // 4), self.data.get("color")]
                for num in structure
            ], dtype=object
        )

    def rotate(self):
        self.rotate_order += 1
        self.rotate_order %= len(self.data.get("structure"))

    def down(self):
        if self.y < rows:
            self.y += 1

    def right(self):
        if self.x < cols:
            self.x += 1

    def left(self):
        if self.x >= 0:
            self.x -= 1

    @property
    def I(self):
        return {
            "structure": np.array([[0, 1, 2, 3], [1, 5, 9, 13]]),
            "color": np.array([34, 56, 68]),
        }

    @property
    def S(self):
        return {
            "structure": np.array([[1, 5, 6, 10], [1, 2, 4, 5]]),
            "color": np.array([96, 110, 118]),
        }

    @property
    def Z(self):
        return {
            "structure": np.array([[2, 5, 6, 9], [0, 1, 5, 6]]),
            "color": np.array([130, 156, 129]),
        }

    @property
    def O(self):
        return {
            "structure": np.array([[0, 1, 4, 5]]),
            "color": np.array([169, 161, 151]),
        }

    @property
    def T(self):
        return {
            "structure": np.array(
                [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9]]
            ),
            "color": np.array([200, 162, 166]),
        }

    @property
    def J(self):
        return {
            "structure": np.array(
                [[1, 5, 8, 9], [0, 4, 5, 6], [1, 2, 5, 9], [4, 5, 6, 10]]
            ),
            "color": np.array([152, 147, 176]),
        }

    @property
    def L(self):
        return {
            "structure": np.array(
                [[0, 4, 8, 9], [4, 5, 6, 8], [0, 1, 5, 9], [2, 4, 5, 6]]
            ),
            "color": np.array([151, 165, 198]),
        }