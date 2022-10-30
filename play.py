from tetris.game import Tetris
import torch


def main():
    tetris = Tetris()
    while tetris.run:
        tetris.tile_fall()
        print(torch.tensor(tetris.binary, dtype=torch.int8))
        tetris.key_down_detection()
        tetris.clear_and_move_down()
        tetris.display()
    tetris.quit()



if __name__ == "__main__":
    main()
