from tetris.objects import Tetris


def main():
    tetris = Tetris()
    while tetris.run:
        tetris.listen_event()
        tetris.display()
    tetris.quit()


if __name__ == '__main__':
    main()
