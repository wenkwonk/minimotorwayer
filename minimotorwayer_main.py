import pyautogui
import cv2
import numpy as np
import math
import heapq
import minimotorwayer_config

#GLOBAL SETUP

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1

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

#BOARD DETECTION

def capture_contours_after_click():
    #takes screenshot right after holding click to capture game window state
    sw, sh = pyautogui.size()
    pyautogui.moveTo(sw / 2, sh / 2, duration=0.5)
    pyautogui.mouseDown()
    screenshot = get_screenshot_array()
    pyautogui.mouseUp()

    #converts screenshot to gray then extracts contours
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def find_border():
    #getting screen scaling
    (screenW, screenH), (scale_X, scale_Y) = get_screen_scaling()
    #taking screenshot after click to capture game window
    img, contours = capture_contours_after_click()
    #list to track common rectangle sizes (likely cells)
    contour_stats = [[1, 1, 1, 1, 0]]  # [x,y,w,h,freq]
    #iterating all contours to group cell-like rectangles
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        raw_area = cv2.contourArea(c)
        #filtering uneven or small shapes
        if raw_area / area < minimotorwayer_config.Detection.min_contour_squareness or area < minimotorwayer_config.Detection.min_contour_area:
            continue
        matched = False
        for saved in contour_stats:
            prev_area = saved[2] * saved[3]
            #checks if contour area matches existing cell size
            if prev_area * 0.975 <= area <= prev_area * 1.025:
                saved[4] += 1
                matched = True
                break
        #adds new cell size grouping
        if not matched:
            contour_stats.append([x, y, w, h, 1])
    #chooses most common rectangle as cell size
    cell_contour = max(contour_stats, key = lambda x: x[4])
    cell_size = cell_contour[3]
    #collecting only true cell-shaped contours
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        raw_area = cv2.contourArea(c)
        if raw_area / area >= minimotorwayer_config.Detection.min_contour_squareness and area >= minimotorwayer_config.Detection.min_contour_area:
            valid.append((x, y, w, h))
    #computing board top-left cell
    top_left = [min(v[0] for v in valid), min(v[1] for v in valid), cell_size]
    #computing board bottom-right cell
    bot_right = [max(v[0] for v in valid),max(v[1] for v in valid),cell_size]
    #DEBUG: visualizing contours and detected board region
    if minimotorwayer_config.Debug.mode == True:
        if top_left and bot_right:
            #unpacking cell positions
            x1, y1, ts1 = bot_right
            x2, y2, ts2 = top_left
            #drawing all found contours
            cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
            #drawing rectangles around extreme cells
            cv2.rectangle(img, (x1, y1), (x1 + ts1, y1 + ts1), (0, 0, 255), 2)
            cv2.rectangle(img, (x2, y2), (x2 + ts2, y2 + ts2), (0, 0, 255), 2)
            #showing result
            cv2.imshow("Detected Board", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            pyautogui.sleep(3)
        else:
            print("no suitable board area detected")
    #calculating full board pixel dimensions
    board_bot_right = [bot_right[0] + cell_size, bot_right[1] + cell_size]
    pixel_width = board_bot_right[0] - top_left[0]
    pixel_height = board_bot_right[1] - top_left[1]
    #estimating number of rows/cols
    rows = pixel_height / cell_size
    cols = pixel_width / cell_size
    #computing corrected cell_size because borders weren't in contour
    cell_under = (cols * 100000) % 100000 / (math.floor(cols) * 1000)
    true_cell_size = cell_size + cell_size * (cell_under / 100)
    #printing board summary
    print(f"board detected: {math.floor(rows)} rows x {math.floor(cols)} cols")
    print(f"board top-left pixels: {top_left[0]/scale_X}, {top_left[1]/scale_Y}")
    print(f"adjusted cell size: {true_cell_size/scale_X}")
    return (math.floor(rows), math.floor(cols), [top_left[0] / scale_X, top_left[1] / scale_Y], true_cell_size / scale_X)

#CELL GEOMETRY

def get_cell_pixel_TL(board_TL, cell_size, row, col):
    #returns the top-left pixel of a cell
    return [board_TL[0] + cell_size * col, board_TL[1] + cell_size * row]

def get_cell_pixel_center(cellTL, cell_size, scale_X, scale_Y):
    #computes pixel center of a cell for clicking
    return [cellTL[0] + (cell_size / scale_X), cellTL[1] + (cell_size / scale_Y)]

def get_ROI(shot, board_TL, cell_size, row, col, scale_X, scale_Y, ratio):
    tl = get_cell_pixel_TL(board_TL, cell_size, row, col)
    fullW = cell_size * scale_X
    fullH = cell_size * scale_Y
    w = int(fullW / ratio)
    h = int(fullH / ratio)
    x = int(tl[0] * scale_X + (fullW - w) / 2)
    y = int(tl[1] * scale_Y + (fullH - h) / 2)
    ROI = shot[y:y+h, x:x+w]
    return ROI

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
    shot = get_screenshot_array()
    for i in range(rows):
        for j in range(cols):
            #computing cell region of interest
            roi = get_ROI(shot, board_TL, cell_size, i, j, scale_X, scale_Y, minimotorwayer_config.Sampling.color_sampling_ratio)
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
    shot = get_screenshot_array()
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
                strips = [[(-1, -1), (0, -1), (+1, -1)],
                          [(-1, -1), (-1, 0), (-1, +1)],
                          [(-1, +2), (0, +2), (+1, +2)],
                          [(-1, +2), (-1, +1), (-1, 0)],
                          [(+2, -1), (+1, -1), (0, -1)],
                          [(+2, -1), (+2, 0), (+2, +1)],
                          [(+2, +2), (+1, +2), (0, +2)],
                          [(+2, +2), (+2, +1), (+2, 0)],]
                for strip in strips:
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
                    corner_ROI = get_ROI(shot, board_TL, cell_size, corner_row, corner_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
                    cp1_ROI = get_ROI(shot, board_TL, cell_size, cp1_row, cp1_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
                    cp2_ROI = get_ROI(shot, board_TL, cell_size, cp2_row, cp2_col, scale_X, scale_Y, minimotorwayer_config.Sampling.cp_sampling_ratio)
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
                    for strip in strips:
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

def print_board(board):
    #printing board cell 
    print('COMPUTER VISUALIZATION')
    for row in board:
        print([f'{cell.color}{cell.type[0]}' for cell in row])
    return board

# PATHFINDING

def heuristic(a, b):
    #cell distances
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (minimotorwayer_config.Pathfinding.diagonal_cost - 2) * min(dx, dy)

def astar(board, start, goal, color):
    rows, cols = len(board), len(board[0])
    start_row, start_col = start
    goal_row, goal_col = goal
    pq = []
    heapq.heappush(pq, (0, 0, (start_row, start_col), [(start_row, start_col)]))
    goal_cost = {(start_row, start_col): 0}
    while pq:
        f, g, (row, col), path = heapq.heappop(pq)
        if (row, col) == (goal_row, goal_col):
            return path
        for dr, dc in minimotorwayer_config.Pathfinding.directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                cell = board[new_row][new_col]
                #allow placing on the goal (ta with matching color)
                if (new_row, new_col) == (goal_row, goal_col):
                    pass
                #allow staying on start only at the beginning
                elif cell.type == "hs":
                    if (new_row, new_col) != (start_row, start_col):
                        continue  #cant place through any other house
                #allow placing only on ev tiles
                elif cell.type != "ev":
                    continue
                #block wrong colored objective tiles
                if cell.type == "ta" and (new_row, new_col) != (goal_row, goal_col) and cell.color != color:
                    continue
                #compute step cost
                if dr != 0 and dc != 0:
                    #diagonal
                    step_cost = minimotorwayer_config.Pathfinding.diagonal_cost
                else:
                    step_cost = 1
                new_goal = g + step_cost
                if (new_row, new_col) not in goal_cost or new_goal < goal_cost[(new_row, new_col)]:
                    goal_cost[(new_row, new_col)] = new_goal
                    h = heuristic((new_row, new_col), (goal_row, goal_col))
                    heapq.heappush(pq, (new_goal + h, new_goal, (new_row, new_col), path + [(new_row, new_col)]))
    return None

def find_optimal_paths(board):
    moves = []
    rows, cols = len(board), len(board[0])
    #locate houses
    houses = []
    for i in range(rows):
        for j in range(cols):
            if board[i][j].type == "hs":
                houses.append((i, j, board[i][j].color))
    #locate targets by color
    targets = {}
    for i in range(rows):
        for j in range(cols):
            if board[i][j].type == "ta":
                tcolor = board[i][j].color
                targets[tcolor] = (i, j)
    #run A*
    for hr, hc, color in houses:
        if color not in targets:
            print(f"No target for color {color}, skipping.")
            continue
        goal = targets[color]
        path = astar(board, (hr, hc), goal, color)
        if path:
            moves.append([[r, c] for (r, c) in path])
        else:
            print(f"No path found for house at {hr},{hc}")
    return moves
    
#MOUSE ACTIONS

def clear_all_roads(board_TL, rows, cols, cell_size, scale_X, scale_Y):
    #clears every cell by right-click dragging over whole board
    pyautogui.PAUSE = 0
    coords = [(i, j) for i in range(rows) for j in range(cols)]
    first = coords[0]
    #starting at first cell
    tl = get_cell_pixel_TL(board_TL, cell_size, first[0], first[1])
    center = get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
    pyautogui.moveTo(*center)
    pyautogui.mouseDown(button='right')
    #dragging across every cell
    for r, c in coords:
        tl = get_cell_pixel_TL(board_TL, cell_size, r, c)
        center = get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
        pyautogui.moveTo(*center)
    pyautogui.mouseUp(button='right')
    pyautogui.PAUSE = 1

def place_roads(board_TL, coordListList, cell_size, scale_X, scale_Y):
    #places roads by dragging across provided cell list
    pyautogui.PAUSE = 0
    for coordList in coordListList:
        first = coordList[0]
        tl = get_cell_pixel_TL(board_TL, cell_size, first[0], first[1])
        center = get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
        #starting drag at first cell without placing
        pyautogui.moveTo(*center)
        pyautogui.sleep(0.25)
        pyautogui.mouseDown()
        #dragging to each cell in list in a straight line
        for r, c in coordList:
            tl = get_cell_pixel_TL(board_TL, cell_size, r, c)
            center = get_cell_pixel_center(tl, cell_size, scale_X, scale_Y)
            pyautogui.moveTo(*center)
        pyautogui.mouseUp()
        pyautogui.sleep(0.25)
    pyautogui.PAUSE = 1

#MAIN BOT

def miniMotorwaysBot():
    #initializes bot and reads board layout
    (screenW, screenH), (scale_X, scale_Y) = get_screen_scaling()
    rows, cols, board_TL, cell_size = find_border()
    board = initialize_board(rows)
    color_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y, minimotorwayer_config.Detection.color_similarity_threshold)
    type_index_board(board, board_TL, rows, cols, cell_size, scale_X, scale_Y)
    print_board(board)
    paths = find_optimal_paths(board)
    place_roads(board_TL, paths, cell_size, scale_X, scale_Y)

#CALL

if __name__ == "__main__":
    miniMotorwaysBot()
