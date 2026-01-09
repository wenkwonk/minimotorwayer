import pyautogui

import minimotorwayer_utils

#ACTIONS

#MOUSE ACTIONS

def clear_all_roads(board_TL, rows, cols, cell_size, scale_X, scale_Y):
    #clears every cell by right-click dragging over whole board
    pyautogui.PAUSE = 0
    coords = [(i, j) for i in range(rows) for j in range(cols)]
    first = coords[0]
    #starting at first cell
    tl = minimotorwayer_utils.get_cell_pixel_TL(board_TL, cell_size, first[0], first[1])
    center = minimotorwayer_utils.get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
    pyautogui.moveTo(*center)
    pyautogui.mouseDown(button='right')
    #dragging across every cell
    for r, c in coords:
        tl = minimotorwayer_utils.get_cell_pixel_TL(board_TL, cell_size, r, c)
        center = minimotorwayer_utils.get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
        pyautogui.moveTo(*center)
    pyautogui.mouseUp(button='right')
    pyautogui.PAUSE = 1

def place_roads(board_TL, coordListList, cell_size, scale_X, scale_Y):
    #places roads by dragging across provided cell list
    pyautogui.PAUSE = 0
    for coordList in coordListList:
        first = coordList[0]
        tl = minimotorwayer_utils.get_cell_pixel_TL(board_TL, cell_size, first[0], first[1])
        center = minimotorwayer_utils.get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
        #starting drag at first cell without placing
        pyautogui.moveTo(*center)
        pyautogui.sleep(0.25)
        pyautogui.mouseDown()
        #dragging to each cell in list in a straight line
        for r, c in coordList:
            tl = minimotorwayer_utils.get_cell_pixel_TL(board_TL, cell_size, r, c)
            center = minimotorwayer_utils.get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
            pyautogui.moveTo(*center)
        pyautogui.mouseUp()
        pyautogui.sleep(0.25)
    pyautogui.PAUSE = 1

print("ACTIONS LOADED")