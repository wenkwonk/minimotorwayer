import pyautogui
import cv2
import numpy as np
import math

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1
screenshot = np.array(pyautogui.screenshot())

#get dimensions
screenWidth, screenHeight = pyautogui.size()
screenshotHeight, screenshotWidth = screenshot.shape[:2]

#calculate scaling factors
scaleX = screenshotWidth / screenWidth
scaleY = screenshotHeight / screenHeight
scaleX, scaleY = int(scaleX), int(scaleY)

print("Screen size:", screenWidth, screenHeight)
print("Screenshot size:", screenshotWidth, screenshotHeight)
print("Scale factors:", scaleX, scaleY)

def findBorder():
    #screen capping game
    pyautogui.moveTo(screenWidth/2, screenHeight/2, duration = 0.5)
    pyautogui.mouseDown()
    screenshot = pyautogui.screenshot()
    pyautogui.mouseUp()
    #converting screen cap to gray
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #converting gray to edge and contours
    edges = cv2.Canny(gray, 50, 50)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourList = [[1,1,1,1,0]]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        rawArea = cv2.contourArea(contour)
        if rawArea / area < 0.975 or area < 5000:
            continue
        else:
            matched = False
            for contour in contourList:
                prevArea = contour[2] * contour[3]
                if prevArea * 0.975 <= area <= prevArea * 1.025:
                    contour[4] += 1
                    matched = True
                    break
            if not matched:
                contourList.append([x, y, w, h, 1])
    #finding the most common rectangular shape (likely tile)
    mostCommonContour = []
    highestFreq = 0
    for contour in contourList:
        currFreq = contour[4]
        if currFreq > highestFreq:
            highestFreq = currFreq
            mostCommonContour = contour
    tileSize = mostCommonContour[3]
    #finding top left
    topLeft = [None, None, tileSize]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        rawArea = cv2.contourArea(contour)
        if rawArea / area < 0.975 or area < 5000:
            continue
        else:
            if topLeft == [None, None, tileSize]:
                topLeft = [x, y, tileSize]
            if x < topLeft[0]:
                topLeft[0] = x
            if y < topLeft[1]:
                topLeft[1] = y
    #finding bot right
    botRight = [0, 0, tileSize]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        rawArea = cv2.contourArea(contour)
        if rawArea / area < 0.975 or area < 5000:
            continue
        else:
            if botRight == [None, None, tileSize]:
                botRight = [x, y, tileSize]
            if x > botRight[0]:
                botRight[0] = x
            if y > botRight[1]:
                botRight[1] = y
    #displaying topLeft and botRight found
    if topLeft and botRight:
        x1, y1, ts1 = botRight 
        x2, y2, ts2 = topLeft 
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
        cv2.rectangle(img, (x1, y1), (x1+ts1, y1+ts1), (0, 0, 255), 2)
        cv2.rectangle(img, (x2, y2), (x2+ts2, y2+ts2), (0, 0, 255), 2)
        cv2.imshow("Detected Board", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pyautogui.sleep(3)
    else:
        print("No suitable board area detected.")
    #calculating the pixel and grid dimensions of the board
    boardBotRight = [botRight[0] + botRight[2], botRight[1] + botRight[2]]
    boardPixelWidth, boardPixelHeight = boardBotRight[0] - topLeft[0], boardBotRight[1] - topLeft[1] 
    boardRows, boardCols = (boardPixelHeight/tileSize), (boardPixelWidth/tileSize)
    #calculating for true cell size as the borders aren't counted as part of the contour shapes
    cellUndersizePercent = (boardCols * 100000) % 100000 / (math.floor(boardCols) * 1000)
    trueTileSize = tileSize + (tileSize * (cellUndersizePercent / 100))
    print(f"board is {math.floor(boardRows)} rows by {math.floor(boardCols)} cols beginning at {[topLeft[0]/scaleX, topLeft[1]/scaleY]} ending at {[botRight[0]/scaleX, botRight[1]/scaleY]} with a true tile size of {trueTileSize/2}")
    return math.floor(boardRows), math.floor(boardCols), [topLeft[0]/scaleX, topLeft[1]/scaleY], trueTileSize/scaleX

def getTileRowCol(boardPixelTopLeft, x, y):
    pass

def getTilePixelTL(boardPixelTopLeft, tileSize, row, col):
    tempCoordX = boardPixelTopLeft[0] 
    tempCoordX += tileSize * col
    tempCoordY = boardPixelTopLeft[1] 
    tempCoordY += tileSize * row
    return [tempCoordX, tempCoordY]

def getTilePixelCenter(tilePixelTL, tileSize):
    tempCoordX = tilePixelTL[0] 
    tempCoordX += (tileSize / scaleX)
    tempCoordY = tilePixelTL[1]
    tempCoordY += (tileSize / scaleY)
    return [tempCoordX, tempCoordY]

def clearAllRoads(boardPixelTopLeft, rows, cols, tileSize):
    clearSequence = []
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    for i in range(rows):
        for j in range(cols):
            clearSequence.append([i, j])
    moveCoords = getTilePixelCenter(getTilePixelTL(boardPixelTopLeft, tileSize, clearSequence[0][0], clearSequence[0][1]), tileSize)
    pyautogui.moveTo(moveCoords[0], moveCoords[1], duration = 0)
    pyautogui.mouseDown(button = 'right')
    for move in clearSequence:
        moveCoords = getTilePixelCenter(getTilePixelTL(boardPixelTopLeft, tileSize, move[0], move[1]), tileSize)
        pyautogui.moveTo(moveCoords[0], moveCoords[1], duration = 0)
    pyautogui.mouseUp(button = 'right')
    pyautogui.PAUSE = 1

def placeRoads(boardPixelTopLeft, coordList, tileSize):
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    moveCoords = getTilePixelCenter(getTilePixelTL(boardPixelTopLeft, tileSize, coordList[0][0], coordList[0][1]), tileSize)
    pyautogui.moveTo(moveCoords[0], moveCoords[1], duration = 0)
    pyautogui.mouseDown()
    for move in coordList:
        moveCoords = getTilePixelCenter(getTilePixelTL(boardPixelTopLeft, tileSize, move[0], move[1]), tileSize)
        pyautogui.moveTo(moveCoords[0], moveCoords[1], duration = 0.1)
    pyautogui.mouseUp()
    pyautogui.PAUSE = 1

def indexingBoard(boardPixelTopLeft, rows, cols, tileSize):
    board = [[] for i in range(rows)]
    colors = [] 
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    screenshot = np.array(pyautogui.screenshot())
    if screenshot.shape[2] == 4:
        screenshot = screenshot[:, :, :3]
    for i in range(rows):
        for j in range(cols):
            tileTL = getTilePixelTL(boardPixelTopLeft, tileSize, i, j)
            topX, topY = tileTL
            w = int(tileSize * scaleX * 3 / 4)
            h = int(tileSize * scaleY * 3 / 4)
            x = int(topX * scaleX) + int(tileSize * scaleX * 1 / 8)
            y = int(topY * scaleY) + int(tileSize * scaleY * 1 / 8)
            roi = screenshot[y:y+h, x:x+w]
            avgRGB = np.mean(roi, axis=(0,1))  #average color of the given cell
            r, g, b = avgRGB
            #check if color already exists
            foundSimilar = False
            for count in range(len(colors)):
                color = colors[count]
                if (abs(color[0]-r) < 50 and abs(color[1]-g) < 50 and abs(color[2]-b) < 50):
                    foundSimilar = True
                    board[i].append([i, j, count])
                    break  #already known color
            if not foundSimilar:
                colors.append([float(r), float(g), float(b), len(colors)])
                board[i].append([i, j, colors[-1][3]])
    for line in board:
        lineL = []
        for count in range(cols):
            lineL.append(line[count][2])
        print(lineL)
    return board


def miniMotorwaysBot():
    boardRows, boardCols, boardPixelTopLeft, tileSize = findBorder()
    #clearAllRoads(boardPixelTopLeft, boardRows, boardCols, tileSize)
    moveList = [[0, 1], [3, 6], [1, 10], [8, 2]]
    #placeRoads(boardPixelTopLeft, moveList, tileSize)
    indexingBoard(boardPixelTopLeft, boardRows, boardCols, tileSize)

miniMotorwaysBot()