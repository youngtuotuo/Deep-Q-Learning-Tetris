import pygame
import random
from pygame.locals import *
from pprint import pprint

import numpy as np

import copy

from tetris_binary_grids import store_grids
from tetris_actions import store_actions

'''Tile structure'''
# I, S, Z, O, T, J, L
# use the following matrix to define each tile
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
I = [(0,1,2,3), (1,5,9,13)]
S = [(1,5,6,10), (1,2,4,5)]
Z = [(2,5,6,9), (0,1,5,6)]
O = [(0,1,4,5)]
T = [(1,4,5,6), (1,5,6,9), (4,5,6,9), (1,4,5,9)]
J = [(1,5,8,9), (0,4,5,6), (1,2,5,9), (4,5,6,10)]
L = [(0,4,8,9), (4,5,6,8), (0,1,5,9), (6,8,9,10)]
tiles = {'I':I, 'S':S, 'Z':Z, 'O':O, 'T':T, 'J':J, 'L':L}
colors = {'I':(34,56,68), 'S':(96,110,118), 'Z':(130,156,129), 'O':(169,161,151),
          'T':(200,162,166), 'J':(152,147,176), 'L':(151,165,198)}

'''Tindow structure'''
tile_size = 24
window_width, window_height = 700, 700
play_width, play_height = 360, 480
top_x, top_y = (window_width - play_width) // 2, (window_height - play_height) // 2
cols, rows = play_width // tile_size, play_height // tile_size

'''Initialize necessary things'''
pygame.init()
window = pygame.display.set_mode((window_width, window_height))
background = pygame.Surface((play_width, play_height))
background.fill((255,255,255))
grids_color = [[(255,255,255) for col in range(cols)] for row in range(rows)]

'''Tile object'''
class Tile:
    def __init__(self):
        self.x = tile_size * (play_width // tile_size // 2 - 1)
        self.y = 0
        self.choice = random.choice(['I', 'S', 'Z', 'O', 'T', 'J', 'L'])
        self.tile = tiles[self.choice]
        self.color = colors[self.choice]
        self.rotate = 0

'''data object that is convenient to call'''
class data:
    def __init__(self):
        self.collect_action = []
        self.collect_binary = []
        self.has_data = False

'''Check tile position, used to lock'''
def valid(obj:Tile, grids_color:list) -> bool:
    whites = [(col, row) for col in range(cols) for row in range(rows) if grids_color[row][col] == (255,255,255)]
    tile_pos = tile_grid_pos(obj)
    for pos in tile_pos:
        if pos not in whites:
            return False
    return True

'''Get the tile's grid positions used to update grid'''
def tile_grid_pos(obj:Tile) -> list:
    tile = obj.tile[obj.rotate % len(obj.tile)]
    pos = []
    for index in tile:
        if index // 4 == 0:
            pos.append((obj.x // tile_size + (index % 4), obj.y // tile_size))
        elif index // 4 == 1:
            pos.append((obj.x // tile_size + (index % 4), obj.y // tile_size + 1))
        elif index // 4 == 2:
            pos.append((obj.x // tile_size + (index % 4), obj.y // tile_size + 2))
        elif index // 4 == 3:
            pos.append((obj.x // tile_size + (index % 4), obj.y // tile_size + 3))
    return pos

values = {'binary_grids':[], 'actions':[], 'score':[], 'current_binary':[]}
RIGHT, LEFT, UP, DOWN, NOTHING = 0, 1, 2, 3, 4
cnn_data = data()

def main(wrapper, values):
    global tile_size, top_x, top_y, play_height, play_width, grids_color
    global RIGHT, LEFT, UP, DOWN, NOTHING
    '''Assign variables before using'''
    run = True
    time = 0
    clock = pygame.time.Clock()
    obj = Tile()
    locked = False
    lock_pos = {}
    speed_data = 180
    score = 0
    count_frequency = 0
    while run:
        clock.tick()
        time += clock.get_rawtime()
        grids_color = [[(255,255,255) for col in range(cols)] for row in range(rows)]
        binary_grids = [[[0] for col in range(cols)] for row in range(rows)] 
        ''' Draw current tile on binary girds '''
        for pos in tile_grid_pos(obj):
            binary_grids[pos[1]][pos[0]] = [1]

        '''Update grids color by locked information'''
        for pos in lock_pos:
            try:
                grids_color[pos[1]][pos[0]] = lock_pos[pos]
                binary_grids[pos[1]][pos[0]] = [1]
            except:
                pass
            
        '''Make tiles falling'''
        if time > speed_data:
            time = 0
            obj.y += tile_size
            if not valid(obj, grids_color):
                obj.y -= tile_size
                locked = True
            
        '''Keyboard control'''
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    quit()

                elif event.key == K_RIGHT:
                    obj.x += tile_size
                    if count_frequency % 7 == 0:
                        cnn_data.collect_binary.append(binary_grids)
                        cnn_data.collect_action.append([RIGHT])
                    if not valid(obj, grids_color):
                        obj.x -= tile_size
                            
                elif event.key == K_LEFT:
                    obj.x -= tile_size
                    if count_frequency % 7 == 0:
                        cnn_data.collect_binary.append(binary_grids)
                        cnn_data.collect_action.append([LEFT])
                    if not valid(obj, grids_color):
                        obj.x += tile_size

                elif event.key == K_DOWN:
                    obj.y += tile_size
                    if count_frequency % 15 == 0:
                        cnn_data.collect_binary.append(binary_grids)
                        cnn_data.collect_action.append([DOWN])
                    if not valid(obj, grids_color):
                        obj.y -= tile_size

                elif event.key == K_UP:
                    pygame.key.set_repeat(1, 95)
                    obj.rotate += 1
                    obj.rotate %= len(obj.tile)
                    if count_frequency % 3 == 0:
                        cnn_data.collect_binary.append(binary_grids)
                        cnn_data.collect_action.append([UP])
                    if not valid(obj, grids_color):
                        obj.rotate -= 1

                # elif event.key == K_r:
                    # main(wrapper, values)

                elif event.key == K_p:
                    pprint(values['binary_grids'])
                    print()
                    print('score:', values['score'])
                    print()
                    print('action:', values['actions'])
                    print()
                    print('count RIGHT:', values['actions'].count([RIGHT]))
                    print('count LEFT:', values['actions'].count([LEFT]))
                    print('count UP:', values['actions'].count([UP]))
                    print('count DOWN:', values['actions'].count([DOWN]))
                    # print('count NOTHING:', values['actions'].count([NOTHING]))


                # if event.key == K_SPACE:
                #     pygame.key.set_repeat(1, 95)
                #     cnn_data.collect_binary.append(binary_grids)
                #     cnn_data.collect_action.append([SPACE])
                #     while valid(obj, grids_color):
                #         obj.y += tile_size
                #     else:
                #         obj.y -= tile_size

            
        count_frequency += 1
        # if count_frequency % 500 == 0:
        #     cnn_data.collect_binary.append(binary_grids)
        #     cnn_data.collect_action.append([NOTHING])

        if count_frequency == 100000:
            count_frequency = 1

        '''Update girds color'''
        for pos in tile_grid_pos(obj):
            try:
                grids_color[pos[1]][pos[0]] = obj.color
            except:
                pass

        '''The following will be activate when tiles need to be locked'''
        if locked:
            for pos in tile_grid_pos(obj):
                lock_pos[pos] = obj.color
            obj = Tile()
            locked = False

            '''Clear the filled rows'''
            for row in range(len(grids_color) - 1, -1, -1):
                if (255,255,255) not in grids_color[row]:
                    for col in range(cols):
                        try:
                            del lock_pos[(col, row)]
                        except:
                            continue
                    score += 10

            '''Move rest tiles down'''
            stage = 0
            set_y = {pos[1] for pos in lock_pos}
            row_rest_y = sorted(list(set_y), reverse=True)
            for rest_y in row_rest_y:
                for pos in list(lock_pos):
                    if pos[1] == rest_y:
                        new_pos = (pos[0], 19 - stage)
                        lock_pos[new_pos] = lock_pos.pop(pos)
                stage += 1

        values['binary_grids'] = cnn_data.collect_binary
        values['actions'] = cnn_data.collect_action
        values['score'] = score

        window.fill((0,0,0))
        background.fill((255,255,255))
        '''Draw grids color and gird'''
        for col in range(cols):
            for row in range(rows):
                pygame.draw.rect(background, grids_color[row][col], (col * tile_size, row * tile_size, tile_size, tile_size), 0)
                pygame.draw.rect(background, (80,80,80), (col * tile_size, row * tile_size, tile_size, tile_size), 1)

        font = pygame.font.SysFont('JetBrains Mono', 30)
        text_score_and_data_amount = font.render('Score: ' + str(score) + ', ' + 'Amount of data: ' + str(len(values['actions'])), False, (255,0,0))
        window.blit(text_score_and_data_amount, (5,5))
        text_count_action = font.render('RIGHT: ' + str(values['actions'].count([RIGHT])) + ', ' +
                                        'LEFT: ' + str(values['actions'].count([LEFT])) + ', ' +
                                        'UP: ' + str(values['actions'].count([UP])) + ', ' +
                                        'DOWN: ' + str(values['actions'].count([DOWN])), False, (255,0,0))
        window.blit(text_count_action, (5,25))
        text_instacne = font.render('There are ' + str(len(store_actions)) + ' instances.', False, (255,255,255))
        window.blit(text_instacne, (5, 45))

        window.blit(background, (top_x, top_y))
        pygame.display.update()

        for x, y in list(lock_pos):
            if y < 1:
                run = False

        if len(values['actions']) % 500 == 0 and len(values['actions']) > 0:
            run = False
    
    with open('tetris_binary_grids.py', 'w') as file:
        for binary_grid in values['binary_grids']:
            store_grids.append(binary_grid)
        file.write('store_grids = ' + str(store_grids))

    with open('tetris_actions.py', 'w') as file:
        for action in values['actions']:
            store_actions.append(action)
        file.write('store_actions = ' + str(store_actions))
    
    wrapper.gameover(values)

games_count = 0

def cnn_main(wrapper,values):    
    global tile_size, top_x, top_y, play_height, play_width, grids_color
    global games_count, RIGHT, LEFT, UP, DOWN, NOTHING
    '''Assign variables before using'''
    run = True
    time = 0
    clock = pygame.time.Clock()
    obj = Tile()
    locked = False
    lock_pos = {}
    speed_data = 180
    score = 0
    count_frequency = 0
    while run:
        clock.tick()
        time += clock.get_rawtime()
        grids_color = [[(255,255,255) for col in range(cols)] for row in range(rows)]
        binary_grids = [[[0] for col in range(cols)] for row in range(rows)] 
        ''' Draw current tile on binary girds '''
        for pos in tile_grid_pos(obj):
            try:
                binary_grids[pos[1]][pos[0]] = [1]
            except:
                continue
        
        '''Update grids color by locked information'''
        for pos in lock_pos:
            try:
                grids_color[pos[1]][pos[0]] = lock_pos[pos]
                binary_grids[pos[1]][pos[0]] = [1]
            except:
                pass
            
        '''Make tiles falling'''
        if time > speed_data:
            time = 0
            obj.y += tile_size
            if not valid(obj, grids_color):
                obj.y -= tile_size
                locked = True

        '''cnn response'''
        values['current_binary'] = binary_grids
        response = wrapper.control(values)
        if response == RIGHT:
            if count_frequency % 30 == 0:
                obj.x += tile_size
                cnn_data.collect_binary.append(binary_grids)
                cnn_data.collect_action.append([RIGHT])
                if not valid(obj, grids_color):
                    obj.x -= tile_size

        elif response == LEFT:
            if count_frequency % 30 == 0:
                obj.x -= tile_size
                cnn_data.collect_binary.append(binary_grids)
                cnn_data.collect_action.append([LEFT])
                if not valid(obj, grids_color):
                    obj.x += tile_size

        elif response == DOWN:
            if count_frequency % 30 == 0:
                obj.y += tile_size
                cnn_data.collect_binary.append(binary_grids)
                cnn_data.collect_action.append([DOWN])
                if not valid(obj, grids_color):
                    obj.y -= tile_size

        elif response == UP:
            if count_frequency % 30 == 0:
                obj.rotate += 1
                obj.rotate %= len(obj.tile)
                cnn_data.collect_binary.append(binary_grids)
                cnn_data.collect_action.append([UP])
                if not valid(obj, grids_color):
                    obj.rotate -= 1

        # elif response == NOTHING:
        #     if count_frequency % 4000 == 0:
        #         cnn_data.collect_binary.append(binary_grids)
        #         cnn_data.collect_action.append([NOTHING])
        
        count_frequency += 1
        if count_frequency == 100000:
            count_frequency = 1

        # elif response == SPACE:
        #     cnn_data.collect_binary.append(binary_grids)
        #     cnn_data.collect_action.append([SPACE])
        #     while valid(obj, grids_color):
        #         obj.y += tile_size
        #     else:
        #         obj.y -= tile_size

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    quit()

        '''Update girds color'''
        for pos in tile_grid_pos(obj):
            try:
                grids_color[pos[1]][pos[0]] = obj.color
            except:
                pass

        '''When tiles need to be locked'''
        if locked:
            for pos in tile_grid_pos(obj):
                lock_pos[pos] = obj.color
            obj = Tile()
            locked = False

            '''Clear the filled rows'''
            for row in range(len(grids_color) - 1, -1, -1):
                if (255,255,255) not in grids_color[row]:
                    '''the variable for showing if clear or not'''
                    for col in range(cols):
                        try:
                            del lock_pos[(col, row)]
                        except:
                            continue
                    score += 10
            
            '''Move rest tiles down'''
            stage = 0
            set_y = {pos[1] for pos in lock_pos}
            row_rest_y = sorted(list(set_y), reverse=True)
            for rest_y in row_rest_y:
                for pos in list(lock_pos):
                    if pos[1] == rest_y:
                        new_pos = (pos[0], 19 - stage)
                        lock_pos[new_pos] = lock_pos.pop(pos)
                stage += 1

        values['binary_grdis'] = cnn_data.collect_binary
        values['actions'] = cnn_data.collect_action
        values['score'] = score

        window.fill((0,0,0))
        background.fill((255,255,255))
        '''Draw grids color and gird'''
        for col in range(cols):
            for row in range(rows):
                pygame.draw.rect(background, grids_color[row][col], (col * tile_size, row * tile_size, tile_size, tile_size), 0)
                pygame.draw.rect(background, (80,80,80), (col * tile_size, row * tile_size, tile_size, tile_size), 1)
        
        font = pygame.font.SysFont('JetBrains Mono', 30)
        text_score_and_data_amount = font.render('Score: ' + str(score) + ', ' + 'Amount of data: ' + str(len(values['actions'])) + ', ' +
                                                 'Games Count: ' + str(games_count) + ', ' + 'Response:' + str(response), False, (255,255,255))
        window.blit(text_score_and_data_amount, (5,5))
        text_count_action = font.render('RIGHT: ' + str(values['actions'].count([RIGHT])) + ', ' +
                                        'LEFT: ' + str(values['actions'].count([LEFT])) + ', ' +
                                        'UP: ' + str(values['actions'].count([UP])) + ', ' +
                                        'DOWN: ' + str(values['actions'].count([DOWN])), False, (255,255,255))
        window.blit(text_count_action, (5,25))
        text_instacne = font.render('There are ' + str(len(store_actions)) + ' instances.', False, (255,255,255))
        window.blit(text_instacne, (5,45))

        window.blit(background, (top_x, top_y))
        pygame.display.update()

        '''check gave over'''
        for x, y in list(lock_pos):
            if y < 1:
                run = False
    if score > 0:
        with open('tetris_binary_grids.py', 'w') as file:
            for binary_grid in values['binary_grids']:
                store_grids.append(binary_grid)
            file.write('store_grids = ' + str(store_grids))

        with open('tetris_actions.py', 'w') as file:
            for action in values['actions']:
                store_actions.append(action)
            file.write('store_actions = ' + str(store_actions))

    games_count += 1
    if games_count == 500:
        wrapper.gaveover(values)
    wrapper.cnn_gameover(values)

if __name__ == '__main__':
    main(values)