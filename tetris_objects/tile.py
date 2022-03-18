from .config import tile_size, play_height, play_width
import random


class Tile:
    """
    Tile structure
    I, S, Z, O, T, J, L
    use the following matrix to define each tile
    >>> 0  1  2  3
    ... 4  5  6  7
    ... 8  9 10 11
    ...12 13 14 15
    """
    tiles = {'I': I, 'S': S, 'Z': Z, 'O': O, 'T': T, 'J': J, 'L': L}
    colors = {'I': (34, 56, 68), 'S': (96, 110, 118), 'Z': (130, 156, 129), 'O': (169, 161, 151),
              'T': (200, 162, 166), 'J': (152, 147, 176), 'L': (151, 165, 198)}

    def __init__(self):
        self.x = tile_size * (play_width // tile_size // 2 - 1)
        self.y = 0
        self.choice = None
        self.tile = None
        self.color = None
        self.rotate = 0

    @property
    def I(self):
        return {'structure': [(0, 1, 2, 3), (1, 5, 9, 13)], 'color': []}

    @property
    def S(self):
        return [(1, 5, 6, 10), (1, 2, 4, 5)]

    @property
    def Z(self):
        return [(2, 5, 6, 9), (0, 1, 5, 6)]

    @property
    def O(self):
        return [(0, 1, 4, 5)]

    @property
    def T(self):
        return [(1, 4, 5, 6), (1, 5, 6, 9), (4, 5, 6, 9), (1, 4, 5, 9)]

    @property
    def J(self):
        return [(1, 5, 8, 9), (0, 4, 5, 6), (1, 2, 5, 9), (4, 5, 6, 10)]

    @property
    def L(self):
        return [(0, 4, 8, 9), (4, 5, 6, 8), (0, 1, 5, 9), (6, 8, 9, 10)]
