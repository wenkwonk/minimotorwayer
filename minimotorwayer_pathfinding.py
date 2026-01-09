import heapq

import minimotorwayer_config

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
    
print("PATHFINDING LOADED")