import pygame
import sys
from tetris.tile import Tile
import numpy as np
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
        self.font = pygame.font.Font(
            "/home/tuo/.local/share/fonts/Hack Regular Nerd Font Complete.ttf", 25
        )

        self.cleared_rows = 0

    def clear_and_move_down(self):
        """Clear complete row and move the whole board down"""
        for row in range(rows - 1, -1, -1):
            if (0, 0, 0) not in self.tiles_grid[row]:
                self.cleared_rows += 1
                self.tiles_grid.pop(row)
                self.tiles_grid.insert(0, [(0, 0, 0) for _ in range(cols)])

    def reset(self):
        self.tiles_grid = [[(0, 0, 0) for _ in range(cols)] for _ in range(rows)]
        self.cleared_rows = 0

    def step(self, action):
        """This method is used for model output.
        """
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

            if key == K_r:
                self.tiles_grid = [
                    [(0, 0, 0) for _ in range(cols)] for _ in range(rows)
                ]

        # Down
        if action == 0:
            self.tile.y += 1
            if not self.collision:
                self.tile.y -= 1
                self.tile.locked = True
                for x, y, color in self.tile.pos_and_color:
                    self.tiles_grid[y][x] = color
                self.check_game_over()
        # Up
        elif action == 1:
            self.tile.rotate_order += 1
            self.tile.rotate_order %= len(self.tile.data.get("structure"))
            if not self.collision:
                self.tile.rotate_order -= 1
        # Right
        elif action == 2:
            self.tile.x += 1
            if not self.collision:
                self.tile.x -= 1
        # Left
        elif action == 3:
            self.tile.x -= 1
            if not self.collision:
                self.tile.x += 1

        self.clear_and_move_down()
        done = self.tile.locked and self.tile.y < 1

        return self.reward, done

    @property
    def touch_ceiling(self):
        return (self.tile.locked and self.tile.y < 1)

    @property
    def n_actions(self):
        return 4

    @property
    def reward(self) -> int:
        return 1 + (self.cleared_rows ** 2) * cols

    @property
    def window_array(self):
        return pygame.surfarray.array2d(self.window)

    @property
    def states(self):
        """The model input. Represent current board's condition."""
        entropy, heights = self.entropy_and_heights
        return self.cleared_rows, self.holes, entropy, heights

    @property
    def entropy_and_heights(self):
        """
        Entropy means how upside down the board is.
        Heights means the total tile height of each column.
        """
        mask = np.array(
            [[cell != (0, 0, 0) for cell in row] for row in self.tiles_grid]
        )
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), rows)
        heights = rows - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        entropy = np.sum(diffs)
        return entropy, total_height

    @property
    def holes(self):
        holes = 0
        for col in zip(*self.tiles_grid):
            row = 0
            while row < rows and col[row] == (0, 0, 0):
                row += 1
            holes += len([x for x in col[row + 1 :] if x == (0, 0, 0)])
        return holes

    def tile_fall(self, fall_speed=fall_speed):
        self.clock.tick()
        self.time += self.clock.get_rawtime()
        if self.time > fall_speed:
            self.time = 0
            self.tile.y += 1
            if not self.collision:
                self.tile.y -= 1
                self.tile.locked = True
                for x, y, color in self.tile.pos_and_color:
                    self.tiles_grid[y][x] = color
                self.check_game_over()

    @property
    def collision(self) -> bool:
        """The collision property."""
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
                if not self.collision:
                    self.tile.y -= 1
                    self.tile.locked = True
                    for x, y, color in self.tile.pos_and_color:
                        self.tiles_grid[y][x] = color
                    self.check_game_over()

            elif key == K_UP:
                self.tile.rotate_order += 1
                self.tile.rotate_order %= len(self.tile.data.get("structure"))
                if not self.collision:
                    self.tile.rotate_order -= 1

            elif key == K_RIGHT:
                self.tile.x += 1
                if not self.collision:
                    self.tile.x -= 1

            elif key == K_LEFT:
                self.tile.x -= 1
                if not self.collision:
                    self.tile.x += 1

    def check_game_over(self):
        if self.tile.locked and self.tile.y < 1:
            self.reset()
        else:
            self.get_new_tile()

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
        self.window.fill((0, 0, 0))
        self.render_grids_and_tiles()
        self.window.blit(self.playground, (tl_x, tl_y))
        rows, holes, entropy, heights = self.states
        text = self.font.render(
            f"rows: {rows}, holes: {holes}, entropy: {entropy}, heights: {heights}",
            True,
            (127, 127, 127),
        )
        self.window.blit(text, (0, 0))
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
