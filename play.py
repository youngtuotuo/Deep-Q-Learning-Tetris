from tetris.game import Tetris
import pygame
import numpy as np


def main():
    tetris = Tetris()
    while tetris.run:
        tetris.play()
        # Convert the window in black color(2D) into a matrix
        window_pixel_matrix = np.array(pygame.surfarray.array3d(tetris.window))
        print(window_pixel_matrix.shape)
    tetris.quit()



if __name__ == "__main__":
    main()
