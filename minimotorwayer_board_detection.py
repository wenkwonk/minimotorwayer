import cv2
import pyautogui
import math

import minimotorwayer_config
import minimotorwayer_utils

#BOARD DETECTION

def capture_contours_after_click():
    #takes screenshot right after holding click to capture game window state
    sw, sh = pyautogui.size()
    pyautogui.moveTo(sw / 2, sh / 2, duration=0.5)
    pyautogui.mouseDown()
    screenshot = minimotorwayer_utils.get_screenshot_array()
    pyautogui.mouseUp()

    #converts screenshot to gray then extracts contours
    img = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def find_border():
    #getting screen scaling
    (screenW, screenH), (scale_X, scale_Y) = minimotorwayer_utils.get_screen_scaling()
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

print("BOARD DETECTION LOADED")