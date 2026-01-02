#CONFIG FILE FOR MINIMOTORWAYER

class Debug:
    mode = False

class Detection:
    #contour detection
    min_contour_area = 5000
    min_contour_squareness = 0.975
    #color detection
    color_similarity_threshold = 55
    #board detection
    ev_tile_ratio = 15

class Sampling:
    #tile color sampling
    color_sampling_ratio = 2
    #carpark
    cp_sampling_ratio = 3

class Pathfinding:
    #movement cost
    diagonal_cost = 1.414
    #octahedral movement
    directions = [(-1, -1), (-1, 0), (-1, +1),
                  (0, -1),            (0, +1),
                  (+1, -1), (+1, 0), (+1, +1)]