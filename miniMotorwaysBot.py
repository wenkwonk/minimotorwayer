import pyautogui
import cv2
import numpy as np
import math
import heapq

#GLOBAL SETUP

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1

#SCREEN CAPTURE UTILITIES

def getScreenshotArray():
    #gets screenshot as numpy array and removes alpha if needed
    img = np.array(pyautogui.screenshot())
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def getScreenScaling():
    #computes screenshot-to-screen scaling factors
    screenW, screenH = pyautogui.size()
    ss = getScreenshotArray()
    ssH, ssW = ss.shape[:2]
    scaleX = int(ssW / screenW)
    scaleY = int(ssH / screenH)
    return (screenW, screenH), (scaleX, scaleY)

#BOARD DETECTION

def captureContoursAfterClick():
    #takes screenshot right after holding click to capture game window state
    sw, sh = pyautogui.size()
    pyautogui.moveTo(sw / 2, sh / 2, duration=0.5)
    pyautogui.mouseDown()
    screenshot = getScreenshotArray()
    pyautogui.mouseUp()

    #converts screenshot to gray then extracts contours
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def findBorder():
    #getting screen scaling
    (screenW, screenH), (scaleX, scaleY) = getScreenScaling()
    #taking screenshot after click to capture game window
    img, contours = captureContoursAfterClick()
    #list to track common rectangle sizes (likely cells)
    contourStats = [[1, 1, 1, 1, 0]]  # [x,y,w,h,freq]
    #iterating all contours to group cell-like rectangles
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        rawArea = cv2.contourArea(c)
        #filtering uneven or small shapes
        if rawArea / area < 0.975 or area < 5000:
            continue
        matched = False
        for saved in contourStats:
            prevArea = saved[2] * saved[3]
            #checks if contour area matches existing cell size
            if prevArea * 0.975 <= area <= prevArea * 1.025:
                saved[4] += 1
                matched = True
                break
        #adds new cell size grouping
        if not matched:
            contourStats.append([x, y, w, h, 1])
    #chooses most common rectangle as cell size
    cellContour = max(contourStats, key=lambda x: x[4])
    cellSize = cellContour[3]
    #collecting only true cell-shaped contours
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        rawArea = cv2.contourArea(c)
        if rawArea / area >= 0.975 and area >= 5000:
            valid.append((x, y, w, h))
    #computing board top-left cell
    topLeft = [min(v[0] for v in valid), min(v[1] for v in valid), cellSize]
    #computing board bottom-right cell
    botRight = [max(v[0] for v in valid),max(v[1] for v in valid),cellSize]
    #visualizing contours and detected board region
    # if topLeft and botRight:
    #     #unpacking cell positions
    #     x1, y1, ts1 = botRight
    #     x2, y2, ts2 = topLeft
    #     #drawing all found contours
    #     cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    #     #drawing rectangles around extreme cells
    #     cv2.rectangle(img, (x1, y1), (x1 + ts1, y1 + ts1), (0, 0, 255), 2)
    #     cv2.rectangle(img, (x2, y2), (x2 + ts2, y2 + ts2), (0, 0, 255), 2)
    #     #showing result
    #     cv2.imshow("Detected Board", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     pyautogui.sleep(3)
    # else:
    #     print("no suitable board area detected")
    #calculating full board pixel dimensions
    boardBotRight = [botRight[0] + cellSize, botRight[1] + cellSize]
    pixelW = boardBotRight[0] - topLeft[0]
    pixelH = boardBotRight[1] - topLeft[1]
    #estimating number of rows/cols
    rows = pixelH / cellSize
    cols = pixelW / cellSize
    #computing corrected cellSize because borders weren't in contour
    cellUnder = (cols * 100000) % 100000 / (math.floor(cols) * 1000)
    truecellSize = cellSize + cellSize * (cellUnder / 100)
    #printing board summary
    print(f"board detected: {math.floor(rows)} rows x {math.floor(cols)} cols")
    print(f"board top-left pixels: {topLeft[0]/scaleX}, {topLeft[1]/scaleY}")
    print(f"adjusted cell size: {truecellSize/scaleX}")
    return (math.floor(rows), math.floor(cols), [topLeft[0] / scaleX, topLeft[1] / scaleY], truecellSize / scaleX)

#CELL GEOMETRY

def getcellPixelTL(boardTL, cellSize, row, col):
    #returns the top-left pixel of a cell
    return [boardTL[0] + cellSize * col, boardTL[1] + cellSize * row]

def getcellPixelCenter(cellTL, cellSize, scaleX, scaleY):
    #computes pixel center of a cell for clicking
    return [cellTL[0] + (cellSize / scaleX), cellTL[1] + (cellSize / scaleY)]

def getROI(shot, boardTL, cellSize, row, col, scaleX, scaleY, ratio):
    tl = getcellPixelTL(boardTL, cellSize, row, col)
    fullW = cellSize * scaleX
    fullH = cellSize * scaleY
    w = int(fullW / ratio)
    h = int(fullH / ratio)
    x = int(tl[0] * scaleX + (fullW - w) / 2)
    y = int(tl[1] * scaleY + (fullH - h) / 2)
    roi = shot[y:y+h, x:x+w]
    return roi

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

def initializeBoard(rows):
    #takes screenshot and initializes empty board
    board = [[] for row in range(rows)]
    return board

#BOARD INDEXING AND SETUP

def colorIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY, threshold):
    #reads screenshot and assigns each cell a color group
    colors = []
    shot = getScreenshotArray()
    for i in range(rows):
        for j in range(cols):
            #computing cell region of interest
            roi = getROI(shot, boardTL, cellSize, i, j, scaleX, scaleY, 2)
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

def typeIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY):
    #assigns cell types via pattern recognition
    rows, cols = len(board), len(board[0])
    colorCounts = {}
    #count frequencies of each color id
    for cellRow in board:
        for cell in cellRow:
            colorCounts[cell.color] = colorCounts.get(cell.color, 0) + 1
    #mark common colors as environment
    for key in list(colorCounts.keys()):
        if colorCounts[key] > rows * cols / 15:
            colorCounts[key] = 'ev'
        else:
            colorCounts[key] = 'hs'
    #assign types
    for cellRow in board:
        for cell in cellRow:
            cell.type = colorCounts[cell.color]
    shot = getScreenshotArray()
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
            majorityColor = None
            for colorID, count in freq.items():
                if count >= 3:
                    majorityColor = colorID
                    break
            #if majority found mark block as objective
            if majorityColor is not None:
                for c in block:
                    c.type = 'ob'
                    c.color = majorityColor
                #finding exit corner
                cornerStripColors = []
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
                    cornerRow = TL.row + strip[0][0]
                    cornerCol = TL.col + strip[0][1]
                    cp1Row = TL.row + strip[1][0]
                    cp1Col = TL.col + strip[1][1]
                    cp2Row = TL.row + strip[2][0]
                    cp2Col = TL.col + strip[2][1]
                    if not (0 <= cornerRow < rows and 0 <= cornerCol < cols
                            and 0 <= cp1Row < rows and 0 <= cp1Col < cols
                            and 0 <= cp2Row < rows and 0 <= cp2Col < cols):
                        continue
                    roiCorner = getROI(shot, boardTL, cellSize, cornerRow, cornerCol, scaleX, scaleY, 3)
                    roiCP1 = getROI(shot, boardTL, cellSize, cp1Row, cp1Col, scaleX, scaleY, 3)
                    roiCP2 = getROI(shot, boardTL, cellSize, cp2Row, cp2Col, scaleX, scaleY, 3)
                    #skip empty ROIs
                    if roiCorner.size == 0 or roiCP1.size == 0 or roiCP2.size == 0:
                        continue
                    r1, g1, b1 = np.mean(roiCorner, axis=(0, 1))
                    r2, g2, b2 = np.mean(roiCP1, axis=(0, 1))
                    r3, g3, b3 = np.mean(roiCP2, axis=(0, 1))
                    avgRGB = ((r1 + r2 + r3) / 3.0, (g1 + g2 + g3) / 3.0, (b1 + b2 + b3) / 3.0)
                    cornerStripColors.append(((cornerRow, cornerCol), avgRGB))
                #pick the brightest
                if cornerStripColors:
                    coordAverages = {coord: np.array(rgb) for (coord, rgb) in cornerStripColors}
                    brightness = {coord: float(rgb.sum()) for coord, rgb in coordAverages.items()}
                    targetCorner = max(brightness, key=brightness.get)
                    #assign the target cell and its two carparks
                    targetCell = board[targetCorner[0]][targetCorner[1]]
                    targetCell.type = 'ta'
                    targetCell.color = majorityColor
                    #find the matching strip to get carpark coords
                    for strip in strips:
                        cornerRow = TL.row + strip[0][0]
                        cornerCol = TL.col + strip[0][1]
                        if (cornerRow, cornerCol) == targetCorner:
                            cp1Row = TL.row + strip[1][0]; cp1Col = TL.col + strip[1][1]
                            cp2Row = TL.row + strip[2][0]; cp2Col = TL.col + strip[2][1]
                            if 0 <= cp1Row < rows and 0 <= cp1Col < cols:
                                board[cp1Row][cp1Col].type = 'cp'
                                board[cp1Row][cp1Col].color = majorityColor
                            if 0 <= cp2Row < rows and 0 <= cp2Col < cols:
                                board[cp2Row][cp2Col].type = 'cp'
                                board[cp2Row][cp2Col].color = majorityColor
                            break
    return board

def printBoard(board):
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
    return dx + dy + (1.414 - 2) * min(dx, dy)

def astar(board, start, goal, color):
    rows, cols = len(board), len(board[0])
    startRow, startCol = start
    goalRow, goalCol = goal
    pq = []
    heapq.heappush(pq, (0, 0, (startRow, startCol), [(startRow, startCol)]))
    goalCost = {(startRow, startCol): 0}
    directions = [(-1, -1), (-1, 0), (-1, +1),
                  (0, -1),            (0, +1),
                  (+1, -1), (+1, 0), (+1, +1)]
    while pq:
        f, g, (row, col), path = heapq.heappop(pq)
        if (row, col) == (goalRow, goalCol):
            return path
        for dr, dc in directions:
            newRow, newCol = row + dr, col + dc
            if 0 <= newRow < rows and 0 <= newCol < cols:
                cell = board[newRow][newCol]
                #allow stepping on the goal (ta with matching color)
                if (newRow, newCol) == (goalRow, goalCol):
                    pass
                #allow staying on start only at the beginning
                elif cell.type == "hs":
                    if (newRow, newCol) != (startRow, startCol):
                        continue  #cant walk through any other house
                #allow walking only on ev tiles
                elif cell.type != "ev":
                    continue
                #block wrong colored objective tiles
                if cell.type == "ta" and (newRow, newCol) != (goalRow, goalCol) and cell.color != color:
                    continue
                #compute step cost
                if dr != 0 and dc != 0:
                    #diagonal
                    stepCost = 1.414
                else:
                    stepCost = 1
                newGoal = g + stepCost
                if (newRow, newCol) not in goalCost or newGoal < goalCost[(newRow, newCol)]:
                    goalCost[(newRow, newCol)] = newGoal
                    h = heuristic((newRow, newCol), (goalRow, goalCol))
                    heapq.heappush(pq, (newGoal + h, newGoal, (newRow, newCol), path + [(newRow, newCol)]))
    return None

def findOptimalPaths(board):
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

def clearAllRoads(boardTL, rows, cols, cellSize, scaleX, scaleY):
    #clears every cell by right-click dragging over whole board
    pyautogui.PAUSE = 0
    coords = [(i, j) for i in range(rows) for j in range(cols)]
    first = coords[0]
    #starting at first cell
    tl = getcellPixelTL(boardTL, cellSize, first[0], first[1])
    center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
    pyautogui.moveTo(*center)
    pyautogui.mouseDown(button='right')
    #dragging across every cell
    for r, c in coords:
        tl = getcellPixelTL(boardTL, cellSize, r, c)
        center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
        pyautogui.moveTo(*center)
    pyautogui.mouseUp(button='right')
    pyautogui.PAUSE = 1

def placeRoads(boardTL, coordListList, cellSize, scaleX, scaleY):
    #places roads by dragging across provided cell list
    pyautogui.PAUSE = 0
    for coordList in coordListList:
        first = coordList[0]
        tl = getcellPixelTL(boardTL, cellSize, first[0], first[1])
        center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
        #starting drag at first cell without placing
        pyautogui.moveTo(*center)
        pyautogui.sleep(0.25)
        pyautogui.mouseDown()
        #dragging to each cell in list in a straight line
        for r, c in coordList:
            tl = getcellPixelTL(boardTL, cellSize, r, c)
            center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
            pyautogui.moveTo(*center)
        pyautogui.mouseUp()
        pyautogui.sleep(0.25)
    pyautogui.PAUSE = 1

#MAIN BOT

def miniMotorwaysBot():
    #initializes bot and reads board layout
    (screenW, screenH), (scaleX, scaleY) = getScreenScaling()
    rows, cols, boardTL, cellSize = findBorder()
    board = initializeBoard(rows)
    colorIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY, 55) #LAST VARIABLE IS COLOR DETECTION THRESHOLD, CHANGE AS NEEDED
    typeIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY)
    printBoard(board)
    paths = findOptimalPaths(board)
    placeRoads(boardTL, paths, cellSize, scaleX, scaleY)

#CALL

if __name__ == "__main__":
    miniMotorwaysBot()