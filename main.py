from tetris.game import Tetris


def main():
    tetris = Tetris()
    while tetris.run:
        tetris.play()
    tetris.quit()
    quit()


if __name__ == '__main__':
    main()
