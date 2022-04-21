import pygame
import sys
from tetris.tile import Tile
from pygame import (
    QUIT,
    KEYDOWN,
    K_ESCAPE,
    K_q,
    K_p,
    K_n,
    K_r,
    K_UP,
    K_DOWN,
    K_RIGHT,
    K_LEFT,
)
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
    fall_speed,
)


class Tetris(object):
    """Main Tetris Game."""

    def __init__(self, *args, **kwargs) -> None:
        pygame.init()
        pygame.key.set_repeat(95, 95)
        self.window = pygame.display.set_mode((window_width, window_height))
        self.window.fill((0, 0, 0))
        self.playground = pygame.Surface((play_width, play_height))
        self.tile = Tile()
        self.tiles_grid = [[(0, 0, 0) for _ in range(cols)] for _ in range(rows)]
        self.clock = pygame.time.Clock()
        self.time = 0
        self.run = True
        self.pause = False

    def play(self):
        self.control()
        self.clear_and_move_down()
        self.display()

    def control(self):
        """Main keyboard listen method.
        q, esc -> quit
        arrow keys -> control
        space -> lock to bottom
        """
        self.tile_fall(fall_speed)
        self.key_down_detection()

    def clear_and_move_down(self):
        for row in range(rows - 1, -1, -1):
            if (0, 0, 0) not in self.tiles_grid[row]:
                self.tiles_grid.pop(row)
                self.tiles_grid.insert(0, [(0, 0, 0) for _ in range(cols)])

    def tile_lock(self):
        self.tile.locked = True
        for x, y, color in self.tile.pos_and_color:
            self.tiles_grid[y][x] = color
        self.get_new_tile()

    def tile_fall(self, fall_speed):
        self.clock.tick()
        self.time += self.clock.get_rawtime()
        if self.time > fall_speed:
            self.time = 0
            self.tile.y += 1
            if not self.feasible:
                self.tile.y -= 1
                self.tile_lock()

    @property
    def feasible(self) -> bool:
        """The feasible property."""
        empty = [
            (col, row)
            for col in range(cols)
            for row in range(rows)
            if self.tiles_grid[row][col] == (0, 0, 0)
        ]
        for x, y, _ in self.tile.pos_and_color:
            if (x, y) not in empty:
                return False
        return True

    def key_down_detection(self):
        """Detect single key press"""
        if pygame.event.get(QUIT):
            self.run = False

        # for single key down press
        events = pygame.event.get(KEYDOWN)
        for event in events:
            key = event.key
            if key == K_ESCAPE or key == K_q:
                self.run = False
            if key == K_p:
                self.pause_game()
            if key == K_n:
                self.get_new_tile()
            if key == K_r:
                self.tiles_grid = [
                    [(0, 0, 0) for _ in range(cols)] for _ in range(rows)
                ]

            if key == K_DOWN:
                self.tile.y += 1
                if not self.feasible:
                    self.tile.y -= 1
                    self.tile_lock()

            elif key == K_UP:
                self.tile.rotate_order += 1
                self.tile.rotate_order %= len(self.tile.data.get("structure"))
                if not self.feasible:
                    self.tile.rotate_order -= 1

            elif key == K_RIGHT:
                self.tile.x += 1
                if not self.feasible:
                    self.tile.x -= 1
            elif key == K_LEFT:
                self.tile.x -= 1
                if not self.feasible:
                    self.tile.x += 1

    def render_grids_and_tiles(self):
        """Draw static tiles -> draw current tile -> draw grid lines"""
        # render locked tiles
        for row, row_array in enumerate(self.tiles_grid):
            for col, color in enumerate(row_array):
                pygame.draw.rect(
                    self.playground,
                    color,
                    (col * tile_size, row * tile_size, tile_size, tile_size),
                    0,
                )
        # render current tile
        for col, row, color in self.tile.pos_and_color:
            pygame.draw.rect(
                self.playground,
                color,
                (col * tile_size, row * tile_size, tile_size, tile_size),
                0,
            )

        # render grid lines
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

    def get_new_tile(self):
        self.tile = Tile()

    def display(self):
        """Display the game window."""
        self.playground.fill((0, 0, 0))
        self.render_grids_and_tiles()
        self.window.blit(self.playground, (tl_x, tl_y))
        pygame.display.update()

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
                    if key == K_r:
                        self.tiles_grid = [
                            [(0, 0, 0) for _ in range(cols)] for _ in range(rows)
                        ]
                    if key == K_n:
                        self.get_new_tile()
                    if key == K_ESCAPE or key == K_q:
                        self.run = False
                        self.pause = False

    def quit(self):
        pygame.quit()
        sys.exit()
