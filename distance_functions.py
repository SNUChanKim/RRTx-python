import numpy as np
import math

def euclidian_dist(x, y):
    return np.linalg.norm(x - y)

def r3s_dist(x, y):
    dist = np.linalg.norm(x[:3]-y[:3])**2 + np.minimum(np.abs(x[3] - y[3]), np.minimum(x[3], y[3]) + 2.0*math.pi - max(x[3], y[3]))**2
    return np.sqrt(dist)
    
def dubins_dist_along_path(x, y):
    return np.linalg.norm(x[:2] - y[:2])

def dubins_dist_along_time_path(x, y):
    return np.linalg.norm(x[:3] - y[:3])

def right_turn_dist(point1, point2, circle_center, r):
    theta = math.atan2(point1[1] - circle_center[1], point1[0] - circle_center[0]) - math.atan2(point2[1] - circle_center[1], point2[0] - circle_center[0])
    if theta < 0:
        theta = theta + 2.0*math.pi
    return theta*r 

def left_turn_dist(point1, point2, circle_center, r):
    theta = math.atan2(point2[1] - circle_center[1], point2[0] - circle_center[0]) - math.atan2(point1[1] - circle_center[1], point1[0] - circle_center[0])
    if theta < 0:
        theta = theta + 2.0*math.pi
    return theta*r 