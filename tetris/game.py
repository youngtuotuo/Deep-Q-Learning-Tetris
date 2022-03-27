import pygame
import sys
import numpy as np
from tetris.tile import Tile
from pygame import QUIT, KEYDOWN, K_ESCAPE, K_q, K_p, K_n
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
    control_threshold,
)


class Tetris(object):
    """Main Tetris Game."""

    def __init__(self, *args, **kwargs) -> None:
        pygame.init()
        self.window = pygame.display.set_mode((window_width, window_height))
        self.window.fill((0, 0, 0))
        self.playground = pygame.Surface((play_width, play_height))
        self.tile = Tile()
        self.lock_pos = np.array([])
        # NOTE: Remember to remove L
        self.tile.data = self.tile.L
        self.clock = pygame.time.Clock()
        self.time = 0
        self.control_tick = 0
        self.run = True
        self.pause = False

    def play(self):
        self.control()
        self.display()

    def control(self):
        """Main keyboard listen method.
        q, esc -> quit
        arrow keys -> control
        space -> lock to bottom
        """
        # TODO: feasible tile region
        self.tile_fall(fall_speed)
        self.single_key_down_detection()
        self.repeat_keys_detection()
        self.tile.caliberation()

    def display(self):
        """Display the game window."""
        self.playground.fill((0, 0, 0))
        self.render_current_tile()
        self.render_grids()
        self.window.blit(self.playground, (tl_x, tl_y))
        pygame.display.update()

    def get_new_tile(self):
        self.tile = Tile()

    def tile_fall(self, fall_speed):
        self.clock.tick()
        self.time += self.clock.get_rawtime()
        if self.time > fall_speed:
            self.time = 0
            # self.tile.y += 1

    def render_current_tile(self):
        for col, row, color in self.tile.pos:
            pygame.draw.rect(
                self.playground,
                color,
                (col * tile_size, row * tile_size, tile_size, tile_size),
                0,
            )
        if self.tile.locked:
            self.lock_pos = (
                self.tile.pos
                if len(self.lock_pos) == 0
                else np.append(self.lock_pos, self.tile.pos, axis=0)
            )
            self.get_new_tile()

    def render_grids(self):
        for col, row, color in self.lock_pos:
            pygame.draw.rect(
                self.playground,
                color,
                (col * tile_size, row * tile_size, tile_size, tile_size),
                0,
            )

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
                        self.pause = False

    def quit(self):
        pygame.quit()
        sys.exit()

    def trigger_movement(self, func):
        self.control_tick += 1
        if self.control_tick > control_threshold:
            self.control_tick = 0
            func()

    def single_key_down_detection(self):
        """Detect single key press"""
        if pygame.event.get(QUIT):
            self.run = False
        # for single key down press
        events = pygame.event.get(KEYDOWN)
        for event in events:
            if event.key == K_ESCAPE or event.key == K_q:
                self.run = False
            if event.key == K_p:
                self.pause_game()
            if event.key == K_n:
                self.get_new_tile()
            if event.key in self.tile.event_control_map:
                self.control_tick = 0
                self.tile.event_control_map[event.key]()

    def repeat_keys_detection(self):
        """Detect repeat key press"""
        keys = pygame.key.get_pressed()
        for event in self.tile.event_control_map:
            if keys[event]:
                self.trigger_movement(self.tile.event_control_map[event])
                # the following break make sure trigger only one repeat key press
                break
