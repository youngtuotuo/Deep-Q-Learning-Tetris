import pygame


class Teris(object):
    def __init__(self, params) -> None:
        pygame.init()
        self.load_params(params)
        window = pygame.display.set_mode((window_width, window_height))
        background = pygame.Surface((play_width, play_height))
        background.fill((255, 255, 255))
        grids_color = [[(255, 255, 255) for _ in range(cols)]
                       for _ in range(rows)]

    def process(self):
        pass

    def load_params(self, params):
        self.tile_size, self.window_width, self.window_height, self.play_width, self.play_height = \
            params['tile_size'], params['window']['width'], params['window']['height'], params['play']['width'], params['play']['height']
        self.top_x, self.top_y = (
            self.window_width - self.play_width) // 2, (self.window_height - self.play_height) // 2
        self.cols, self.rows = self.play_width // self.tile_size, self.play_height // self.tile_size
