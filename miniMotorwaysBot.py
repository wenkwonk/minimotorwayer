import pyautogui
import cv2
import numpy as np
import math

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

def capture_contours_after_click():
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
    img, contours = capture_contours_after_click()
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
    if topLeft and botRight:
        #unpacking cell positions
        x1, y1, ts1 = botRight
        x2, y2, ts2 = topLeft
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

#cell GEOMETRY

def getcellPixelTL(boardTL, cellSize, row, col):
    #returns the top-left pixel of a cell
    return [boardTL[0] + cellSize * col, boardTL[1] + cellSize * row]

def getcellPixelCenter(cellTL, cellSize, scaleX, scaleY):
    #computes pixel center of a cell for clicking
    return [cellTL[0] + (cellSize / scaleX), cellTL[1] + (cellSize / scaleY)]

#BOARD INITIALIZATION

class cell:
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
    board = [[] for _ in range(rows)]
    return board

#BOARD INDEXING AND SETUP

def colorIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY, threshold):
    #reads screenshot and assigns each cell a color group
    colors = []
    threshold = 60  #threshold variable
    shot = getScreenshotArray()
    for i in range(rows):
        for j in range(cols):
            #computing cell region of interest
            tl = getcellPixelTL(boardTL, cellSize, i, j)
            x = int(tl[0] * scaleX + cellSize * scaleX / 8)
            y = int(tl[1] * scaleY + cellSize * scaleY / 8)
            w = int(cellSize * scaleX * 3 / 4)
            h = int(cellSize * scaleY * 3 / 4)
            roi = shot[y:y+h, x:x+w]
            r, g, b = np.mean(roi, axis=(0, 1))
            matched = False
            #checking similarity with existing colors
            for idx, c in enumerate(colors):
                if (abs(c[0] - r) < threshold and 
                    abs(c[1] - g) < threshold and 
                    abs(c[2] - b) < threshold):
                    board[i].append(cell(i, j, idx, None))
                    matched = True
                    break
            #assign new color id
            if not matched:
                colors.append([float(r), float(g), float(b)])
                board[i].append(cell(i, j, len(colors) - 1, None))
    return board

def typeIndexBoard(board):
    #assigns cell types via pattern recognition
    rows, cols = len(board), len(board[0])
    colorCounts = {}
    #dictionary to count color frequencies
    for cellRow in board:
        for cell in cellRow:
            if cell.color in colorCounts:
                colorCounts[cell.color] += 1
            else:
                colorCounts[cell.color] = 1
    for key in colorCounts:
        #cell types making up over a quarter of cells are likely environment
        if colorCounts[key] > rows*cols/4:
            colorCounts[key] = 'ev'
        else:
            #otherwise likely a special type, use house as placeholder
            colorCounts[key] = 'hs'
    for cellRow in board:
        for cell in cellRow:
            cell.type = colorCounts[cell.color]
    #groups of houses are classified into objectives
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if i + 1 < rows and j + 1 < cols:
                TL = board[i][j]
                TR = board[i][j+1]
                BL = board[i+1][j]
                BR = board[i+1][j+1]
                if (
                    TL.color == TR.color == BL.color == BR.color and TL.type  == TR.type  == BL.type  == BR.type  == 'hs'):
                    TL.type = TR.type = BL.type = BR.type = 'ob'
    return board

def printBoard(board):
    #printing board cell 
    for row in board:
        print([f'{cell.color}{cell.type[0]}' for cell in row])
    return board

#MOUSE ACTIONS

def clearAllRoads(boardTL, rows, cols, cellSize, scaleX, scaleY):
    #clears every cell by right-click dragging over whole board
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
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
    pyautogui.FAILSAFE = False
    for coordList in coordListList:
        first = coordList[0]
        tl = getcellPixelTL(boardTL, cellSize, first[0], first[1])
        center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
        #starting drag at first cell without placing
        pyautogui.moveTo(*center)
        pyautogui.mouseDown()
        #dragging to each cell in list in a straight line
        for r, c in coordList:
            tl = getcellPixelTL(boardTL, cellSize, r, c)
            center = getcellPixelCenter(tl, cellSize, scaleX, scaleY)
            pyautogui.moveTo(*center)
    pyautogui.mouseUp()
    pyautogui.PAUSE = 1

#PATHFINDING

def findOptimalPaths(board):
    moveLists = []
    houseCoordList = []
    rows, cols = len(board), len(board[0])
    for i in range(rows):
        for j in range(cols):
            if board[i][j].type == 'hs':
                houseCoordList.append((i, j, board[i][j].color))
    for houseCoord in houseCoordList:
        path = pathFind(board, [[houseCoord[0], houseCoord[1]]], houseCoord[2], set())
        if path != None:
            moveLists.append(path)
    return moveLists

def pathFind(board, moveList, color, visited):
    row, col = moveList[-1][0], moveList[-1][1]
    #cycle prevention
    if (row, col) in visited:
        return None
    visited.add((row, col))
    if board[row][col].type == 'ob' and board[row][col].color == color:
        return moveList
    else:
        for direction in ((-1, -1), (-1, 0), (-1, +1), (0, -1), (0, +1), (+1, -1), (+1, 0), (+1, +1)):
            newRow, newCol = row + direction[0], col + direction[1]
            if 0 <= newRow < len(board) and 0 <= newCol < len(board[0]):
                cell = board[newRow][newCol]
                if cell.type == 'ob' and cell.color != color:
                    continue
                if cell.type == 'ob' and cell.color == color:
                    moveList.append([newRow, newCol])
                    return moveList
                if cell.type == 'ev':
                    moveList.append([newRow, newCol])
                    chain = pathFind(board, moveList, color, visited)
                    if chain is not None:
                        return chain
                    moveList.pop()
        return None

#MAIN BOT

def miniMotorwaysBot():
    #initializes bot and reads board layout
    (screenW, screenH), (scaleX, scaleY) = getScreenScaling()
    rows, cols, boardTL, cellSize = findBorder()
    board = initializeBoard(rows)
    colorIndexBoard(board, boardTL, rows, cols, cellSize, scaleX, scaleY, 60) #LAST VARIABLE IS COLOR DETECTION THRESHOLD, CHANGE AS NEEDED
    typeIndexBoard(board)
    printBoard(board)
    paths = findOptimalPaths(board)
    placeRoads(boardTL, paths, cellSize, scaleX, scaleY)

#CALL

if __name__ == "__main__":
    miniMotorwaysBot()