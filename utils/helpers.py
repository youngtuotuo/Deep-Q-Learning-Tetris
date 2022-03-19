def valid(obj: Tile, grids_color: list) -> bool:
    '''Check tile position, used to lock'''
    whites = [(col, row) for col in range(cols) for row in range(rows)
              if grids_color[row][col] == (255, 255, 255)]
    tile_pos = tile_grid_pos(obj)
    for pos in tile_pos:
        if pos not in whites:
            return False
    return True


def tile_grid_pos(obj: Tile) -> list:
    '''Get the tile's grid positions used to update grid'''
    tile = obj.tile[obj.rotate % len(obj.tile)]
    pos = []
    for index in tile:
        if index // 4 == 0:
            pos.append((obj.x // tile_size + (index % 4), obj.y // tile_size))
        elif index // 4 == 1:
            pos.append(
                (obj.x // tile_size + (index % 4), obj.y // tile_size + 1))
        elif index // 4 == 2:
            pos.append(
                (obj.x // tile_size + (index % 4), obj.y // tile_size + 2))
        elif index // 4 == 3:
            pos.append(
                (obj.x // tile_size + (index % 4), obj.y // tile_size + 3))
    return pos
