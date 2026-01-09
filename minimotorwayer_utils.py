import pyautogui
import numpy as np

#UTILS

#SCREEN CAPTURE UTILITIES

def get_screenshot_array():
    #gets screenshot as numpy array and removes alpha if needed
    img = np.array(pyautogui.screenshot())
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def get_screen_scaling():
    #computes screenshot-to-screen scaling factors
    screen_width, screen_height = pyautogui.size()
    screenshot = get_screenshot_array()
    screenshot_height, screenshot_width = screenshot.shape[:2]
    scale_X = int(screenshot_width / screen_width)
    scale_Y = int(screenshot_height / screen_height)
    return (screen_width, screen_height), (scale_X, scale_Y)

#CELL GEOMETRY

def get_cell_pixel_TL(board_TL, cell_size, row, col):
    #returns the top-left pixel of a cell
    return [board_TL[0] + cell_size * col, board_TL[1] + cell_size * row]

def get_cell_pixel_center(cellTL, cell_size, scale_X, scale_Y):
    #computes pixel center of a cell for clicking
    return [cellTL[0] + (cell_size / scale_X), cellTL[1] + (cell_size / scale_Y)]

def get_roi(shot, board_TL, cell_size, row, col, scale_X, scale_Y, ratio):
    tl = get_cell_pixel_TL(board_TL, cell_size, row, col)
    fullW = cell_size * scale_X
    fullH = cell_size * scale_Y
    w = int(fullW / ratio)
    h = int(fullH / ratio)
    x = int(tl[0] * scale_X + (fullW - w) / 2)
    y = int(tl[1] * scale_Y + (fullH - h) / 2)
    ROI = shot[y:y+h, x:x+w]
    return ROI

#BOARD VISUALIZATION

def print_board(board):
    #printing board cell 
    print('COMPUTER VISUALIZATION')
    for row in board:
        print([f'{cell.color}{cell.type[0]}' for cell in row])
    return board

print("UTILITIES LOADED")