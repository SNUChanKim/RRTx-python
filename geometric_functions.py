import numpy as np
import math
from copy import deepcopy
import distance_functions as df
import collision_checking_functions as ccf

def point_in_polygon(point, polygon): #point is 1X2 and polygon is pX2
    P = len(polygon)
    if P < 2:
        return False
    
    num_crossings = 0
    start_point = polygon[P-1][:2]
    for i in range(P):
        end_point = polygon[i][:2]
        if ((start_point[1] > point[1] and end_point[1] < point[1]) or 
            (start_point[1] < point[1] and end_point[1] > point[1])):
            if start_point[0] > point[0] and end_point[0] > point[0]:
                num_crossings += 1
            elif start_point[0] < point[0] and end_point[0] < point[0]:
                start_point = end_point
                continue
            else:
                T = 2.0*np.maximum(start_point[0], end_point[0])
                
                x = (-((start_point[0]*end_point[1]-start_point[1]*end_point[0])*(point[0]-T))+((start_point[0]-end_point[0])*(point[0]*point[1]-point[1]*T)))/((start_point[1]-end_point[1])*(point[0]-T))
                
                if x > point[0]:
                    num_crossings += 1
        start_point = end_point
    if num_crossings % 2 == 0:
        return False
    return True
        
def distance_sqrd_point_to_segment(point, start_point, end_point):
    vx = point[0] - start_point[0]
    vy = point[1] - start_point[1]
    ux = end_point[0] - start_point[0]
    uy = end_point[1] - start_point[1]
    determinate = vx*ux + vy*uy 
    
    if determinate <= 0:
        return vx*vx + vy*vy
    else:
        len = ux*ux + uy*uy
        if determinate >= len:
            return np.linalg.norm(end_point[:2] - point[:2])**2
        else:
            return (ux*vy - uy*vx)**2 / len
        
def dist_to_polygon_sqrd(point, polygon):
    min_distance_sqrd = math.inf
    
    P = len(polygon)
    start_point = polygon[P-1][:2]
    
    for i in range(P):
        end_point = polygon[i][:2]
        
        this_distance_sqrd = distance_sqrd_point_to_segment(point, start_point, end_point)
        
        if this_distance_sqrd < min_distance_sqrd:
            min_distance_sqrd = this_distance_sqrd
        start_point = end_point
    
    return min_distance_sqrd

def dist_to_polygon_prism_sqrd(point, polygon, this_obstacle):
    min_distance_sqrd = math.inf
    
    P = len(polygon)
    start_point = polygon[P-1][:2]
    
    for i in range(P):
        end_point = polygon[i][:2]
        
        if not ccf.seg_in_prism(this_obstacle, point, point):
            continue
        
        this_distance_sqrd = distance_sqrd_point_to_segment(point, start_point, end_point)
        
        if this_distance_sqrd < min_distance_sqrd:
            min_distance_sqrd = this_distance_sqrd
        
        start_point = end_point
    
    return min_distance_sqrd

def segment_dist_sqrd(PA, PB, QA, QB): # return minimum distance between line segments [PA, PB] and [QA, QB]
    possible_intersect = True
    
    if np.abs(PB[0] - PA[0]) < 0.000001: # check if P is close to vertical
        if (QA[0] >= PA[0] and QB[0] >= PA[0]) or (QA[0] <= PA[0] and QB[0] <= PA[0]):
            possible_intersect = False
    else:
        m = (PB[1] - PA[1])/(PB[0] - PA[0])
        
        diff_A = (m*(QA[0] - PA[0]) + PA[1]) - QA[1] 
        diff_B = (m*(QB[0] - PA[0]) + PA[1]) - QB[1]
        if (diff_A > 0.0 and diff_B > 0.0) or (diff_A < 0.0 and diff_B < 0.0):
            possible_intersect = False
        
    if possible_intersect:
        if np.abs(QB[0] - QA[0]) < 0.000001:
            if (PA[0] >= QA[0] and PB[0] >= QA[0]) or (PA[0] <= QA[0] and PB[0] <= QA[0]):
                possible_intersect = False    
        else:
            m = (QB[1] - QA[1])/(QB[0] - QA[0])
            
            diff_A = (m*(PA[0] - QA[0]) + QA[1]) - PA[1] 
            diff_B = (m*(PB[0] - QA[0]) + QA[1]) - PB[1]
            if (diff_A > 0.0 and diff_B > 0.0) or (diff_A < 0.0 and diff_B < 0.0):
                possible_intersect = False
    
    if possible_intersect:
        return 0.0
    
    return np.min([distance_sqrd_point_to_segment(PA, QA, QB),
                  distance_sqrd_point_to_segment(PB, QA, QB),
                  distance_sqrd_point_to_segment(QA, PA, PB),
                  distance_sqrd_point_to_segment(QB, PA, PB)])
    
def find_index_before_time(path, time_to_find):
    if len(path) < 1:
        return -1
    for i in range(len(path)):
        if path[i][2] > time_to_find:
            return i - 1
    
    return i

def find_transform_obs_to_time_of_point(this_obstacle, point):
    ind_before = find_index_before_time(this_obstacle.path, point[2])
    
    if ind_before < 0:
        dx = this_obstacle.path[0][0]
        dy = this_obstacle.path[0][1]
        return (dx, dy)
    elif ind_before == len(this_obstacle.path) - 1:
        dx = this_obstacle.path[-1][0]
        dy = this_obstacle.path[-1][1]
        return (dx, dy)
    
    ind_after = ind_before + 1
    per_portion_along_edge = (point[2] - this_obstacle.path[ind_before][2])/(this_obstacle.path[ind_after][2] - this_obstacle.path[ind_before][2])
    
    dx = this_obstacle.path[ind_before][0] + per_portion_along_edge*(this_obstacle.path[ind_after][0] - this_obstacle.path[ind_before][0])
    dy = this_obstacle.path[ind_before][1] + per_portion_along_edge*(this_obstacle.path[ind_after][1] - this_obstacle.path[ind_before][1])
    
    return (dx, dy)
