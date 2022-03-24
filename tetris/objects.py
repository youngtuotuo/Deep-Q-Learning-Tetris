import pygame
import sys
from pygame import QUIT, KEYDOWN, K_ESCAPE, K_q, K_p, K_DOWN, K_UP, K_RIGHT, K_LEFT
import random
from tetris.constants import (
    tile_size,
    window_width,
    rows,
    cols,
    window_height,
    play_width,
    play_height,
    tl_x,
    tl_y,
)


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
        self.x = 7  # TODO: use random to start x position
        self.y = 0
        self.name = random.choice(["I", "S", "Z", "O", "T", "J", "L"])
        self.data = getattr(self, self.name)
        self.rotate_order = 0
        self.locked = False

    @property
    def pos(self):
        """Position of grids."""
        structure = self.data.get("structure")
        structure = structure[self.rotate_order % len(structure)]
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

    def roate(self):
        # TODO: rotation method here or in Tetris object
        self.rotate_order += 1
        self.rotate_order %= len(self.data.get("structure"))

    @property
    def I(self):
        return {"structure": [(0, 1, 2, 3), (1, 5, 9, 13)], "color": (34, 56, 68)}

    @property
    def S(self):
        return {"structure": [(1, 5, 6, 10), (1, 2, 4, 5)], "color": (96, 110, 118)}

    @property
    def Z(self):
        return {"structure": [(2, 5, 6, 9), (0, 1, 5, 6)], "color": (130, 156, 129)}

    @property
    def O(self):
        return {"structure": [(0, 1, 4, 5)], "color": (169, 161, 151)}

    @property
    def T(self):
        return {
            "structure": [(1, 4, 5, 6), (1, 5, 6, 9), (4, 5, 6, 9), (1, 4, 5, 9)],
            "color": (200, 162, 166),
        }

    @property
    def J(self):
        return {
            "structure": [(1, 5, 8, 9), (0, 4, 5, 6), (1, 2, 5, 9), (4, 5, 6, 10)],
            "color": (152, 147, 176),
        }

    @property
    def L(self):
        return {
            "structure": [(0, 4, 8, 9), (4, 5, 6, 8), (0, 1, 5, 9), (6, 8, 9, 10)],
            "color": (151, 165, 198),
        }


class Tetris(object):
    """Setup basic windows"""

    def __init__(self, *args, **kwargs) -> None:
        pygame.init()
        pygame.key.set_repeat(1, 95)
        self.window = pygame.display.set_mode((window_width, window_height))
        self.window.fill((0, 0, 0))
        self.playground = pygame.Surface((play_width, play_height))
        self.playground.fill((255, 255, 255))
        self.tile = Tile()
        self.clock = pygame.time.Clock()
        self.run = True
        self.pause = False

    def update(self):
        """Update game states."""
        pass

    def get_new_tile(self):
        self.tile = Tile()

    def display(self):
        """Display the game window."""
        # TODO: render tiles
        self.playground.fill((255, 255, 255))
        self.render_tile()
        self.render_grids()
        self.window.blit(self.playground, (tl_x, tl_y))
        pygame.display.update()

    def render_tile(self):
        color = self.tile.data.get('color')
        for col, row in self.tile.pos:
            pygame.draw.rect(self.playground, color, ((col - 1) * tile_size, row * tile_size, tile_size, tile_size), 0)

    def render_grids(self):
        for row in range(1, rows):
            y = tile_size * row
            pygame.draw.line(
                self.playground,
                color=(100, 100, 100),
                start_pos=(0, y),
                end_pos=(play_width, y),
            )
        for col in range(1, cols):
            x = tile_size * col
            pygame.draw.line(
                self.playground,
                color=(100, 100, 100),
                start_pos=(x, 0),
                end_pos=(x, play_height),
            )

    def pause_game(self):
        self.pause = True
        while self.pause:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.run = False
                if event.type == KEYDOWN:
                    key = event.key
                    if key == K_p:
                        self.pause = False
                    if key == K_ESCAPE or key == K_q:
                        self.run = False

    def quit(self):
        pygame.quit()
        sys.exit()

    def control(self):
        """Main keyboard listen method.
        q, esc -> quit
        arrow keys -> control
        space -> lock to bottom
        """
        # TODO: control
        for event in pygame.event.get():
            if event.type == QUIT:
                self.run = False
            if event.type == KEYDOWN:
                key = event.key
                if key == K_ESCAPE or key == K_q:
                    self.run = False
                if key == K_p:
                    self.pause_game()
                if key == K_DOWN:
                    if self.tile.y < rows:
                        self.tile.y += 1
                elif key == K_UP:
                    if self.tile.y > 0:
                        self.tile.y -= 1
                elif key == K_RIGHT:
                    if self.tile.x < cols:
                        self.tile.x += 1
                elif key == K_LEFT:
                    if self.tile.x > 0:
                        self.tile.x -= 1
