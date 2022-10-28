import random
from tetris.constants import cols
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
        self.x = cols // 2
        self.y = 0
        self.name = random.choice(["I", "S", "Z", "O", "T", "J", "L"])
        self.data = getattr(self, self.name)
        self.rotate_order = 0
        self.locked = False

    @property
    def pos_and_color(self):
        """Position and color in grids."""
        structure = self.data.get("structure")
        structure = structure[self.rotate_order % len(structure)]
        return np.array(
            [
                [self.x + (num % 4), self.y + (num // 4), self.data.get("color")]
                for num in structure
            ], dtype=object
        )

    @property
    def I(self):
        return {
            "structure": np.array([[0, 1, 2, 3], [1, 5, 9, 13]]),
            "color": (34, 56, 68),
        }

    @property
    def S(self):
        return {
            "structure": np.array([[1, 5, 6, 10], [1, 2, 4, 5]]),
            "color": (96, 110, 118),
        }

    @property
    def Z(self):
        return {
            "structure": np.array([[2, 5, 6, 9], [0, 1, 5, 6]]),
            "color": (130, 156, 129),
        }

    @property
    def O(self):
        return {
            "structure": np.array([[0, 1, 4, 5]]),
            "color": (169, 161, 151),
        }

    @property
    def T(self):
        return {
            "structure": np.array(
                [[1, 4, 5, 6], [1, 5, 6, 9], [4, 5, 6, 9], [1, 4, 5, 9]]
            ),
            "color": (200, 162, 166),
        }

    @property
    def J(self):
        return {
            "structure": np.array(
                [[1, 5, 8, 9], [0, 4, 5, 6], [1, 2, 5, 9], [4, 5, 6, 10]]
            ),
            "color": (152, 147, 176),
        }

    @property
    def L(self):
        return {
            "structure": np.array(
                [[1, 5, 9, 10], [4, 5, 6, 8], [0, 1, 5, 9], [2, 4, 5, 6]]
            ),
            "color": (151, 165, 198),
        }
