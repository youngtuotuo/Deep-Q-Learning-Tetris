from tetris.game import Tetris
from tetris.constants import fall_speed
import pygame
import numpy as np


def main():
    tetris = Tetris()
    while tetris.run:
        tetris.tile_fall()
        tetris.key_down_detection()
        tetris.clear_and_move_down()
        tetris.display()
    tetris.quit()


if __name__ == "__main__":
    main()
