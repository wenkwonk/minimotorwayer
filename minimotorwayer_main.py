import pyautogui

import minimotorwayer_config
import minimotorwayer_utils
import minimotorwayer_board_detection
import minimotorwayer_board_init
import minimotorwayer_pathfinding
import minimotorwayer_actions

#GLOBAL SETUP

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1

#MAIN BOT

def miniMotorwaysBot():
    #initializes bot and reads board layout
    (screenW, screenH), (scale_X, scale_Y) = minimotorwayer_utils.get_screen_scaling()
    rows, cols, board_TL, cell_size = minimotorwayer_board_detection.find_border()
    board = minimotorwayer_board_init.initialize_board(rows)
    minimotorwayer_board_init.color_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y, minimotorwayer_config.Detection.color_similarity_threshold)
    minimotorwayer_board_init.type_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y)
    minimotorwayer_utils.print_board(board)
    paths = minimotorwayer_pathfinding.find_optimal_paths(board)
    minimotorwayer_actions.place_roads(board_TL, paths, cell_size, scale_X, scale_Y)

#CALL

if __name__ == "__main__":
    miniMotorwaysBot()
