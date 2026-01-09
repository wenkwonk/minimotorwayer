import numpy as np

import minimotorwayer_config
import minimotorwayer_utils

#BOARD INIT

#BOARD INITIALIZATION

class Cell:
    #represents a single cell on the board
    def __init__(self, row, col, color, type):
        self.row = row
        self.col = col
        self.color = color
        self.type = type

    def __repr__(self):
        return f"cell({self.row},{self.col}) color={self.color}"

def initialize_board(rows):
    #takes screenshot and initializes empty board
    board = [[] for row in range(rows)]
    return board

#BOARD INDEXING AND SETUP

def color_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y, threshold):
    #reads screenshot and assigns each cell a color group
    colors = []
    shot = minimotorwayer_utils.get_screenshot_array()
    for i in range(rows):
        for j in range(cols):
            #computing cell region of interest
            roi = minimotorwayer_utils.get_roi(shot, board_TL, cell_size, i, j, scale_X, scale_Y, minimotorwayer_config.Sampling.color_sampling_ratio)
            r, g, b = np.mean(roi, axis=(0, 1))
            matched = False
            #checking similarity with existing colors
            for idx, c in enumerate(colors):
                if (abs(c[0] - r) < threshold and 
                    abs(c[1] - g) < threshold and 
                    abs(c[2] - b) < threshold):
                    board[i].append(Cell(i, j, idx, None))
                    matched = True
                    break
            #assign new color id
            if not matched:
                colors.append([float(r), float(g), float(b)])
                board[i].append(Cell(i, j, len(colors) - 1, None))
    return board

def type_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y):
    #assigns cell types via pattern recognition
    rows, cols = len(board), len(board[0])
    color_counts = {}
    #count frequencies of each color id
    for cell_row in board:
        for cell in cell_row:
            color_counts[cell.color] = color_counts.get(cell.color, 0) + 1
    #mark common colors as environment
    for key in list(color_counts.keys()):
        if color_counts[key] > rows * cols / minimotorwayer_config.Detection.ev_tile_ratio:
            color_counts[key] = 'ev'
        else:
            color_counts[key] = 'hs'
    #assign types
    for cell_row in board:
        for cell in cell_row:
            cell.type = color_counts[cell.color]
    shot = minimotorwayer_utils.get_screenshot_array()
    #groups of houses are classified into objectives using a 3/4 majority rule
    for i in range(rows - 1):
        for j in range(cols - 1):
            TL = board[i][j]
            TR = board[i][j + 1]
            BL = board[i + 1][j]
            BR = board[i + 1][j + 1]
            block = [TL, TR, BL, BR]
            #count colors among house tiles only
            freq = {}
            for c in block:
                if c.type == 'hs':
                    freq[c.color] = freq.get(c.color, 0) + 1
            #find a majority color if exists
            majority_color = None
            for color_ID, count in freq.items():
                if count >= 3:
                    majority_color = color_ID
                    break
            #if majority found mark block as objective
            if majority_color is not None:
                for c in block:
                    c.type = 'ob'
                    c.color = majority_color
                #finding exit corner
                corner_strip_colors = []
                #all possible locations of carpark + exit
                for strip in minimotorwayer_config.Sampling.cp_strips:
                    corner_row = TL.row + strip[0][0]
                    corner_col = TL.col + strip[0][1]
                    cp1_row = TL.row + strip[1][0]
                    cp1_col = TL.col + strip[1][1]
                    cp2_row = TL.row + strip[2][0]
                    cp2_col = TL.col + strip[2][1]
                    if not (0 <= corner_row < rows and 0 <= corner_col < cols
                            and 0 <= cp1_row < rows and 0 <= cp1_col < cols
                            and 0 <= cp2_row < rows and 0 <= cp2_col < cols):
                        continue
                    corner_ROI = minimotorwayer_utils.get_roi(shot, board_TL, cell_size, corner_row, corner_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
                    cp1_ROI = minimotorwayer_utils.get_roi(shot, board_TL, cell_size, cp1_row, cp1_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
                    cp2_ROI = minimotorwayer_utils.get_roi(shot, board_TL, cell_size, cp2_row, cp2_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
                    #skip empty ROIs
                    if corner_ROI.size == 0 or cp1_ROI.size == 0 or cp2_ROI.size == 0:
                        continue
                    r1, g1, b1 = np.mean(corner_ROI, axis=(0, 1))
                    r2, g2, b2 = np.mean(cp1_ROI, axis=(0, 1))
                    r3, g3, b3 = np.mean(cp2_ROI, axis=(0, 1))
                    avg_RGB = ((r1 + r2 + r3) / 3.0, (g1 + g2 + g3) / 3.0, (b1 + b2 + b3) / 3.0)
                    corner_strip_colors.append(((corner_row, corner_col), avg_RGB))
                #pick the brightest
                if corner_strip_colors:
                    coord_avgs = {coord: np.array(rgb) for (coord, rgb) in corner_strip_colors}
                    brightness = {coord: float(rgb.sum()) for coord, rgb in coord_avgs.items()}
                    target_corner = max(brightness, key = brightness.get)
                    #assign the target cell and its two carparks
                    target_cell = board[target_corner[0]][target_corner[1]]
                    target_cell.type = 'ta'
                    target_cell.color = majority_color
                    #find the matching strip to get carpark coords
                    for strip in minimotorwayer_config.Sampling.cp_strips:
                        corner_row = TL.row + strip[0][0]
                        corner_col = TL.col + strip[0][1]
                        if (corner_row, corner_col) == target_corner:
                            cp1_row = TL.row + strip[1][0]; cp1_col = TL.col + strip[1][1]
                            cp2_row = TL.row + strip[2][0]; cp2_col = TL.col + strip[2][1]
                            if 0 <= cp1_row < rows and 0 <= cp1_col < cols:
                                board[cp1_row][cp1_col].type = 'cp'
                                board[cp1_row][cp1_col].color = majority_color
                            if 0 <= cp2_row < rows and 0 <= cp2_col < cols:
                                board[cp2_row][cp2_col].type = 'cp'
                                board[cp2_row][cp2_col].color = majority_color
                            break
    return board

print("BOARD INIT LOADED")