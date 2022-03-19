import pygame
import random


class Tile(object):
    """Tile structures: I, S, Z, O, T, J, L.
    Use the following matrix to define each tile

     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15

    Usage:
    >>> tile = Tile(params)
    ... tile.data
    """

    def __init__(self, *args, **kwargs):
        self.x = start_x
        self.y = 0
        self.name = random.choice(['I', 'S', 'Z', 'O', 'T', 'J', 'L'])
        self.data = getattr(self, self.name)
        self.rotate = 0
        self.locked = False

    def pos(self):
        """Position of grids.
        """
        structure = self.data.get('structure')
        structure = structure[self.rotate % len(structure)]
        pos = []
        for num in structure:
            if num // 4 == 0:
                pos.append((self.x + (num % 4), self.y))
            elif num // 4 == 1:
                pos.append((self.x + (num % 4), self.y + 1))
            elif num // 4 == 2:
                pos.append((self.x + (num % 4), self.y + 2))
            elif num // 4 == 3:
                pos.append((self.x + (num % 4), self.y + 3))
        return pos

    @property
    def I(self):
        return {
            'structure': [(0, 1, 2, 3), (1, 5, 9, 13)],
            'color': (34, 56, 68)
        }

    @property
    def S(self):
        return {
            'structure': [(1, 5, 6, 10), (1, 2, 4, 5)],
            'color': (96, 110, 118)
        }

    @property
    def Z(self):
        return {
            'structure': [(2, 5, 6, 9), (0, 1, 5, 6)],
            'color': (130, 156, 129)
        }

    @property
    def O(self):
        return {'structure': [(0, 1, 4, 5)], 'color': (169, 161, 151)}

    @property
    def T(self):
        return {
            'structure': [(1, 4, 5, 6), (1, 5, 6, 9), (4, 5, 6, 9),
                          (1, 4, 5, 9)],
            'color': (200, 162, 166)
        }

    @property
    def J(self):
        return {
            'structure': [(1, 5, 8, 9), (0, 4, 5, 6), (1, 2, 5, 9),
                          (4, 5, 6, 10)],
            'color': (152, 147, 176)
        }

    @property
    def L(self):
        return {
            'structure': [(0, 4, 8, 9), (4, 5, 6, 8), (0, 1, 5, 9),
                          (6, 8, 9, 10)],
            'color': (151, 165, 198)
        }


class Teris(object):
    """Setup basic windows
    """

    def __init__(self, *args, **kwargs) -> None:
        pygame.init()
        self.load_params(kwargs)
        self.window = pygame.display.set_mode(
            kwargs['window']['width'], kwargs['window']['height'])
        self.background = pygame.Surface(
            self.play_width, self.play_width)
        self.background.fill((255, 255, 255))
        self.grids_color = [[(255, 255, 255) for _ in range(self.cols)]
                            for _ in range(self.rows)]
        self.tile = Tile(self.tile_size, self.play_width)
        self.clock = pygame.time.Clock()

    def update(self, tile):
        pass

    def get_new_tile(self):
        self.tile = Tile(self.start_x)

    def load_params(self, **kwargs):
        self.tile_size = kwargs['tile_size']
        self.play_width = kwargs['play']['width']
        self.play_height = kwargs['play']['height']
        self.top_x = (kwargs['window']['width'] - self.play_width) // 2
        self.top_y = (kwargs['window']['height'] - self.play_height) // 2
        self.cols = self.play_width // self.tile_size
        self.rows = self.play_height // self.tile_size
        self.start_x = self.play_width // self.tile_size // 2 - 1
