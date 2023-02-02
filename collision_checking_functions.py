import numpy as np
import math
from copy import deepcopy
# import simple_edge_functions as ef
# import dubins_edge_functions as ef
import geometric_functions as gf
# from simple_edge import SimpleEdge as Edge
# from dubins_edge import DubinsEdge as Edge
from edge import Edge
import edge_functions as ef
from data_structure import RRTNode, Obstacle, CSpace

def quick_check_2d(this_obstacle: Obstacle, point):
    point_inside_obstacle = False
    if this_obstacle.obstacle_unused or this_obstacle.life_span <= 0:
        return False
    
    if (1 <= this_obstacle.kind <= 5) and ef.w_dist(this_obstacle.position, point) > this_obstacle.radius:
        return False
    
    if this_obstacle.kind == 1:
        return True
    elif this_obstacle.kind == 2 or this_obstacle.kind == 4:
        raise NotImplementedError('Need to be Implemented for obstacle kind 2 and 4')
    elif this_obstacle.kind == 3:
        if gf.point_in_polygon(point, this_obstacle.polygon):
            return True
    elif this_obstacle.kind == 5:
        if gf.point_in_polygon(point[:2], this_obstacle.polygon):
            if seg_in_prism(this_obstacle, point, point):
                return True
    elif this_obstacle.kind == 6 or this_obstacle.kind == 7:
        (dx, dy) = gf.find_transform_obs_to_time_of_point(this_obstacle, point)
        
        if ef.w_dist(np.array([this_obstacle.position[0] + dx, this_obstacle.position[1] + dy]), point) > this_obstacle.radius:
            return False

        this_obstacle.polygon = [np.array([this_obstacle.original_polygon[i][0] + dx, this_obstacle.original_polygon[i][1] + dy]) for i in range(len(this_obstacle.original_polygon))]
        if gf.point_in_polygon(point, this_obstacle.polygon):
            return True
    
    return False

def quick_check(C: CSpace, point):
    if hasattr(point, 'position'):
        position = point.position
    else:
        position = point
        
    for obstacle in C.obstacles:
        if quick_check_2d(obstacle, position):
            return True
    return False

def explicit_point_check_2d(this_obstacle: Obstacle, point, min_dist, robot_radius):
    this_dist = math.inf
    if this_obstacle.obstacle_unused or this_obstacle.life_span <= 0:
        return (False, min_dist)
    
    if (1 <= this_obstacle.kind <= 5):
        this_dist = ef.w_dist(this_obstacle.position, point) - robot_radius
        if this_dist - this_obstacle.radius > min_dist:
            return (False, min_dist)
    
    if this_obstacle.kind == 1:
        this_dist = this_dist - this_obstacle.radius
        if this_dist < 0.0:
            return (True, 0.0)
        
    elif this_obstacle.kind == 2 or this_obstacle.kind == 4:
        raise NotImplementedError('Need to be Implemented for obstacle kind 2 and 4')

    elif this_obstacle.kind == 3:
        if gf.point_in_polygon(point, this_obstacle.polygon):
            return (True, 0.0)
        this_dist = np.sqrt(gf.dist_to_polygon_sqrd(point, this_obstacle.polygon)) - robot_radius
        
        if this_dist < 0.0:
            return (True, 0.0)
    
    elif this_obstacle.kind == 5:
        if gf.point_in_polygon(point[:2], this_obstacle.polygon):
            if seg_in_prism(this_obstacle, point, point):
                return (True, 0.0)
        this_dist = np.sqrt(gf.dist_to_polygon_prism_sqrd(point, this_obstacle.polygon, this_obstacle)) - robot_radius
        
        if this_dist < 0.0:
            return (True, 0.0)
    
    elif this_obstacle.kind == 6 or this_obstacle.kind == 7:
        (dx, dy) = gf.find_transform_obs_to_time_of_point(this_obstacle, point)
        
        this_dist = ef.w_dist(np.array([this_obstacle.position[0] + dx, this_obstacle.position[1] + dy]), point) - robot_radius
        if this_dist - this_obstacle.radius > min_dist:
            return (False, min_dist)
        
        this_obstacle.polygon = [np.array([this_obstacle.original_polygon[i][0] + dx, this_obstacle.original_polygon[i][1] + dy]) for i in range(len(this_obstacle.original_polygon))]
        
        if gf.point_in_polygon(point, this_obstacle.polygon):
            return (True, 0.0)
        
        this_dist = np.sqrt(gf.dist_to_polygon_sqrd(point, this_obstacle.polygon)) - robot_radius
        if this_dist < 0.0:
            return (True, 0.0)
    
    else:
        raise NotImplementedError('This kind of obstacles are not implemented')
    
    return (False, np.minimum(min_dist, this_dist))

def explicit_point_check(C: CSpace, point):
    if C.in_warmup_time:
        return (False, math.inf)
    
    if quick_check(C, point):
        return (True, 0.0)
    
    R = C.d//2
    ret_cert = math.inf
    
    for obstacle in C.obstacles:
        (this_ret_val, this_cert) = explicit_point_check_2d(obstacle, point, ret_cert, C.robot_radius)
        
        if this_ret_val:
            return (True, 0.0)
        
        if this_cert < ret_cert:
            ret_cert = this_cert
        
    return (False, ret_cert)

def explicit_node_check(C: CSpace, node):
    return explicit_point_check(C, node.position)

def seg_in_prism(this_obstacle: Obstacle, start_point, end_point):
    delta = math.inf
    for d in range(2, len(this_obstacle.prism_span_max)):
        if delta > this_obstacle.prism_span_max[d] - this_obstacle.prism_span_min[d]:
            delta = this_obstacle.prism_span_max[d] - this_obstacle.prism_span_min[d]
        
    delta /= 20.0
    steps = ef.dist(start_point, end_point)/delta
    
    if np.array_equal(start_point, end_point):
        delta = 1
        steps = 1
    
    for s in np.arange(0.0, steps+1.0, 1.0):
        test_pose = start_point + (end_point - start_point)*s/steps
        
        all_in = True
        for d in range(2, len(this_obstacle.prism_span_max)):
            if (test_pose[d] < this_obstacle.prism_span_min[d] or 
                test_pose[d] > this_obstacle.prism_span_max[d]):
                all_in = False
                break
        
        if all_in:
            return True
        
        return False

def explicit_edge_check_2d(this_obstacle: Obstacle, start_point, end_point, robot_radius, margin=0):
    if this_obstacle.obstacle_unused or this_obstacle.life_span <= 0:
        return False
    
    if 1 <= this_obstacle.kind <= 5:
        dist_sqrd = gf.distance_sqrd_point_to_segment(this_obstacle.position, start_point[:2], end_point[:2])
        if dist_sqrd > (robot_radius + this_obstacle.radius)**2:
            return False
    
    if this_obstacle.kind == 1:
        return True
    elif this_obstacle.kind == 2 or this_obstacle.kind == 4:
        raise NotImplementedError('Need to be Implemented for obstacle kind 2 and 4')
    elif this_obstacle.kind == 3 or this_obstacle.kind == 5:
        P = len(this_obstacle.polygon)
        if P < 2:
            return False
        
        A = this_obstacle.polygon[P-1][:2]
        for i in range(P):
            B = this_obstacle.polygon[i][:2]
            
            if gf.segment_dist_sqrd(start_point[:2], end_point[:2], A, B) < robot_radius**2:
                if this_obstacle.kind == 5:
                    if seg_in_prism(this_obstacle, start_point, end_point):
                        return True
                else:
                    return True
            A = B
    elif this_obstacle.kind == 6 or this_obstacle.kind == 7:
        if start_point[2] < end_point[2]:
            early_point = start_point
            late_point = end_point
        else:
            late_point = start_point
            early_point = end_point
        
        first_obs_ind = np.maximum(gf.find_index_before_time(this_obstacle.path, early_point[2]), 0)
        last_obs_ind = np.minimum(1 + gf.find_index_before_time(this_obstacle.path, late_point[2]), len(this_obstacle.path) - 1)
        if last_obs_ind <= first_obs_ind:
            return False

        for i_start in range(first_obs_ind, last_obs_ind):
            i_end = i_start + 1
            
            x_1 = early_point[0]
            y_1 = early_point[1]
            t_1 = early_point[2]
            
            x_2 = this_obstacle.path[i_start][0] + this_obstacle.position[0]
            y_2 = this_obstacle.path[i_start][1] + this_obstacle.position[1]
            t_2 = this_obstacle.path[i_start][2]
            
            m_x1 = (late_point[0] - x_1)/(late_point[2] - t_1 + 1e-10)
            m_y1 = (late_point[1] - y_1)/(late_point[2] - t_1 + 1e-10)
            m_x2 = (this_obstacle.path[i_end][0] + this_obstacle.position[0] - x_2)/(this_obstacle.path[i_end][2] - t_2 + 1e-10)
            m_y2 = (this_obstacle.path[i_end][1] + this_obstacle.position[1] - y_2)/(this_obstacle.path[i_end][2] - t_2 + 1e-10)
            
            t_c = ((m_x1**2 * t_1 + m_x2 * (m_x2 * t_2 + x_1 - x_2) - 
                    m_x1 * (m_x2 * (t_1 + t_2) + x_1 - x_2) + 
                    (m_y1 - m_y2) * (m_y1 * t_1 - m_y2 * t_2 - y_1 + y_2)) / 
                   ((m_x1 - m_x2)**2 + (m_y1 - m_y2)**2))
            
            if t_c < np.maximum(t_1, t_2):
                t_c = np.maximum(t_1, t_2)
            elif t_c > np.minimum(late_point[2], this_obstacle.path[i_end][2]):
                t_c = np.minimum(late_point[2], this_obstacle.path[i_end][2])
                
            r_x = m_x1*(t_c - t_1) + x_1
            r_y = m_y1*(t_c - t_1) + y_1
            o_x = m_x2*(t_c - t_2) + x_2
            o_y = m_y2*(t_c - t_2) + y_2
            # print("radius: ", this_obstacle.radius)
            # print("polygon: ", this_obstacle.polygon)
            # print("position: ", this_obstacle.position)
            if np.sqrt((r_x - o_x)**2 + (r_y - o_y)**2) < this_obstacle.radius + robot_radius + margin:
                return True
    return False
            
def explicit_edge_check(C: CSpace, edge: Edge, obstacle=None, margin=0):
    if obstacle is not None:
        return ef.explicit_edge_check(C, edge, obstacle, margin)
    else:
        if C.in_warmup_time:
            return False
        
        for obstacle in C.obstacles:
            if ef.explicit_edge_check(C, edge, obstacle, margin):
                return True
        return False


    
