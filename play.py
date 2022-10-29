from tetris.game import Tetris
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
