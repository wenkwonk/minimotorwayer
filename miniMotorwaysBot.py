import pyautogui
import cv2
import numpy as np
import math

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
    screenW, screenH = pyautogui.size()
    ss = get_screenshot_array()
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
    screenshot = get_screenshot_array()
    pyautogui.mouseUp()

    #converts screenshot to gray then extracts contours
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def findBorder():
    #getting screen scaling
    (screenW, screenH), (scaleX, scaleY) = get_screen_scaling()
    #taking screenshot after click to capture game window
    img, contours = capture_contours_after_click()
    #list to track common rectangle sizes (likely tiles)
    contourStats = [[1, 1, 1, 1, 0]]  # [x,y,w,h,freq]
    #iterating all contours to group tile-like rectangles
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
            #checks if contour area matches existing tile size
            if prevArea * 0.975 <= area <= prevArea * 1.025:
                saved[4] += 1
                matched = True
                break
        #adds new tile size grouping
        if not matched:
            contourStats.append([x, y, w, h, 1])
    #chooses most common rectangle as tile size
    tileContour = max(contourStats, key=lambda x: x[4])
    tileSize = tileContour[3]
    #collecting only true tile-shaped contours
    valid = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        rawArea = cv2.contourArea(c)
        if rawArea / area >= 0.975 and area >= 5000:
            valid.append((x, y, w, h))
    #computing board top-left tile
    topLeft = [min(v[0] for v in valid), min(v[1] for v in valid), tileSize]
    #computing board bottom-right tile
    botRight = [max(v[0] for v in valid),max(v[1] for v in valid),tileSize]
    #visualizing contours and detected board region
    if topLeft and botRight:
        #unpacking tile positions
        x1, y1, ts1 = botRight
        x2, y2, ts2 = topLeft
        #drawing all found contours
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        #drawing rectangles around extreme tiles
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
    boardBotRight = [botRight[0] + tileSize, botRight[1] + tileSize]
    pixelW = boardBotRight[0] - topLeft[0]
    pixelH = boardBotRight[1] - topLeft[1]
    #estimating number of rows/cols
    rows = pixelH / tileSize
    cols = pixelW / tileSize
    #computing corrected tileSize because borders weren't in contour
    cellUnder = (cols * 100000) % 100000 / (math.floor(cols) * 1000)
    trueTileSize = tileSize + tileSize * (cellUnder / 100)
    #printing board summary
    print(f"board detected: {math.floor(rows)} rows x {math.floor(cols)} cols")
    print(f"board top-left pixels: {topLeft[0]/scaleX}, {topLeft[1]/scaleY}")
    print(f"adjusted tile size: {trueTileSize/scaleX}")
    return (math.floor(rows), math.floor(cols), [topLeft[0] / scaleX, topLeft[1] / scaleY], trueTileSize / scaleX)

#TILE GEOMETRY

def getTilePixelTL(boardTL, tileSize, row, col):
    #returns the top-left pixel of a tile
    return [boardTL[0] + tileSize * col, boardTL[1] + tileSize * row]

def getTilePixelCenter(tileTL, tileSize, scaleX, scaleY):
    #computes pixel center of a tile for clicking
    return [tileTL[0] + (tileSize / scaleX), tileTL[1] + (tileSize / scaleY)]

#MOUSE ACTIONS

def clearAllRoads(boardTL, rows, cols, tileSize, scaleX, scaleY):
    #clears every tile by right-click dragging over whole board
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    coords = [(i, j) for i in range(rows) for j in range(cols)]
    first = coords[0]
    #starting at first tile
    tl = getTilePixelTL(boardTL, tileSize, first[0], first[1])
    center = getTilePixelCenter(tl, tileSize, scaleX, scaleY)
    pyautogui.moveTo(*center)
    pyautogui.mouseDown(button='right')
    #dragging across every tile
    for r, c in coords:
        tl = getTilePixelTL(boardTL, tileSize, r, c)
        center = getTilePixelCenter(tl, tileSize, scaleX, scaleY)
        pyautogui.moveTo(*center)
    pyautogui.mouseUp(button='right')
    pyautogui.PAUSE = 1

def placeRoads(boardTL, coordList, tileSize, scaleX, scaleY):
    #places roads by dragging across provided tile list
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    first = coordList[0]
    tl = getTilePixelTL(boardTL, tileSize, first[0], first[1])
    center = getTilePixelCenter(tl, tileSize, scaleX, scaleY)
    #starting drag at first tile without placing
    pyautogui.moveTo(*center)
    pyautogui.mouseDown()
    #dragging to each tile in list in a straight line
    for r, c in coordList:
        tl = getTilePixelTL(boardTL, tileSize, r, c)
        center = getTilePixelCenter(tl, tileSize, scaleX, scaleY)
        pyautogui.moveTo(*center)
    pyautogui.mouseUp()
    pyautogui.PAUSE = 1

#BOARD INDEXING

class Tile:
    #represents a single tile on the board
    def __init__(self, row, col, color, type):
        self.row = row
        self.col = col
        self.color = color
        self.type = type

    def __repr__(self):
        return f"Tile({self.row},{self.col}) color={self.color}"

def indexingBoard(boardTL, rows, cols, tileSize, scaleX, scaleY):
    #reads screenshot and assigns each tile a color group
    shot = get_screenshot_array()
    board = [[] for _ in range(rows)]
    colors = []
    #THRESHOLD VARIABLE
    threshold = 60 
    #ADJUST TO CHANGE COLOR TOLERANCE
    for i in range(rows):
        for j in range(cols):
            #computing tile region of interest
            tl = getTilePixelTL(boardTL, tileSize, i, j)
            x = int(tl[0] * scaleX + tileSize * scaleX / 8)
            y = int(tl[1] * scaleY + tileSize * scaleY / 8)
            w = int(tileSize * scaleX * 3 / 4)
            h = int(tileSize * scaleY * 3 / 4)
            roi = shot[y:y+h, x:x+w]
            r, g, b = np.mean(roi, axis=(0, 1))
            matched = False
            #checking similarity with existing colors
            for idx, c in enumerate(colors):
                if (abs(c[0] - r) < threshold and abs(c[1] - g) < threshold and abs(c[2] - b) < threshold):
                    board[i].append(Tile(i, j, idx, None))
                    matched = True
                    break
            #assign new color id
            if not matched:
                colors.append([float(r), float(g), float(b)])
                board[i].append(Tile(i, j, len(colors) - 1, None))
    #printing board color layout
    print("\nindexed board colors:")
    for row in board:
        print([tile.color for tile in row])
    return board

#MAIN BOT

def miniMotorwaysBot():
    #initializes bot and reads board layout
    (screenW, screenH), (scaleX, scaleY) = get_screen_scaling()
    rows, cols, boardTL, tileSize = findBorder()
    board = indexingBoard(boardTL, rows, cols, tileSize, scaleX, scaleY)

#CALL

if __name__ == "__main__":
    miniMotorwaysBot()
