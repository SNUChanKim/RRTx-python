import numpy as np
import math
from data_structure import CSpace, Obstacle
from kd_tree_general import KDTree
import collision_checking_functions as ccf
import csv
import warnings
from copy import deepcopy

def prod(list):
    result = 1
    for i in range(len(list)):
        result *= list[i]
    return result

def decrease_life(O: Obstacle):
    O.life_span -= 1.0
    
def change_obstacle_direction(S: CSpace, O: Obstacle, current_time):
    end_time = S.start[2]
    path_time_step = 3.0
    
    while(O.unknown_path[O.next_direction_change_ind][2] > current_time and O.next_direction_change_ind > 0):
        O.next_direction_change_ind -= 1
        
    high_point = np.zeros(3)
    low_point = np.zeros(3)
    
    if (O.unknown_path[O.next_direction_change_ind][2] <= current_time and O.next_direction_change_ind == len(O.unknown_path) - 1):
        ind_we_care_about = len(O.unknown_path) - 1
        
        high_point = O.unknown_path[ind_we_care_about]
        low_point = O.unknown_path[ind_we_care_about - 1]
        
        O.next_direction_change_time = O.unknown_path[ind_we_care_about - 1][2]
    
    elif (O.unknown_path[O.next_direction_change_ind][2] > current_time and O.next_direction_change_ind <= 0):
        high_point = np.array([O.unknown_path[0][0], O.unknown_path[0][1], current_time])
        high_point = np.array([O.unknown_path[0][0], O.unknown_path[0][1], current_time - path_time_step])
        
        O.next_direction_change_time = -math.inf
        
    else:
        high_point = O.unknown_path[O.next_direction_change_ind + 1]
        low_point = O.unknown_path[O.next_direction_change_ind]
        O.next_direction_change_time = O.unknown_path[O.next_direction_change_ind][2]
        
    mx = (high_point[0] - low_point[0]) / (high_point[2] - low_point[2])
    my = (high_point[1] - low_point[1]) / (high_point[2] - low_point[2])
    
    ts = list(np.arange(current_time, end_time, -path_time_step))
    ts.append(end_time)
    L = len(ts)
    
    O.path = []
    for t in range(L):
        point_x = low_point[0] + (ts[L-t-1] - low_point[2])*mx
        point_y = low_point[1] + (ts[L-t-1] - low_point[2])*my
        point_t = low_point[2] + ts[L-t-1] - low_point[2]
        O.path.append(np.array([point_x, point_y, point_t]))
        
def save_obstacle_locations(obstacles, file_name):
    fptr = open(file_name, 'w')
    writer = csv.writer(fptr)
    
    for ob in obstacles:
        if ob.kind == 6 or ob.kind == 7:
            for i in range(len(ob.path)):
                writer.writerow([ob.path[i][0] + ob.position[0], ob.path[i][1] + ob.position[1], ob.path[i][2] + 0.0, ob.radius])
            writer.writerow([math.nan, math.nan, math.nan, math.nan])
            continue
        if ob.kind != 3 and ob.kind != 4 and ob.kind != 5:
            warnings.warn("Warning: Cannot save non-polygon obstacle to file (not implemented)")
            continue
        
        if ob.obstacle_unused:
            continue
        
        for i in range(len(ob.polygon)):
            writer.writerow(ob.polygon[i][:])
        
        writer.writerow(ob.polygon[0][:])
        writer.writerow([math.nan, math.nan])
        
    fptr.close()

def random_sample_obs(S: CSpace, KD: KDTree, ob: Obstacle):
    if not S.space_has_time and not S.space_has_theta:
        ob_hyper_volume_bound = (2.0*ob.radius)**(S.d)
        
        if S.hyper_volume == 0:
            S.hyper_volume = prod(S.width)
    elif not S.space_has_time and S.space_has_theta:
        ob_hyper_volume_bound = (2.0*ob.radius)**2

        if S.hyper_volume == 0.0:
            S.hyper_volume = prod(S.width[:2])
    else:
        raise NotImplementedError("not coded yet")

    num_obs_samples = KD.tree_size * ob_hyper_volume_bound/S.hyper_volume + 1.0

    for smp in np.arange(1, num_obs_samples, 1):
        new_point = np.random.rand(S.d)
        new_point[0] = ob.position[0] + new_point[0] * ob.radius*2.0 - ob.radius
        new_point[1] = ob.position[1] + new_point[1] * ob.radius*2.0 - ob.radius

        if ccf.quick_check_2d(ob, new_point):
            S.sample_stack.append(new_point)

def add_obs_to_cspace(C: CSpace, ob: Obstacle):
    C.obstacles.append(ob)

def read_obstacle_from_file(S: CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
        
        for i in range(obs_mult):
            add_obs_to_cspace(S, Obstacle(3, polygon=polygon))
            
    a.close()
    
def read_dynamic_obstacles_from_file(S: CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
        
        start_time_and_life_span = np.array(list(a.readline()))
        
        for i in range(obs_mult):
            ob = Obstacle(3, polygon=polygon)
            ob.start_time = start_time_and_life_span[0]
            ob.life_span = start_time_and_life_span[1]
            add_obs_to_cspace(S, Obstacle(3, polygon=polygon))
            
    a.close()
    
def read_discoverable_obstacles_from_file(S: CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
            
        obs_behavior_type = int(a.readline())
        
        for i in range(obs_mult):
            ob = Obstacle(3, polygon=polygon)
            
            if obs_behavior_type == 0:
                ob.senseable_obstacle = False
                ob.obstacle_unused_after_sense = False
                ob.obstacle_unused = False
                print("normal")
            
            elif obs_behavior_type == -1:
                ob.senseable_obstacle = True
                ob.obstacle_unused_after_sense = True
                ob.obstacle_unused = False
                print("vanishing")
                
            elif obs_behavior_type == 1:
                ob.senseable_obstacle = True
                ob.obstacle_unused_after_sense = False
                ob.obstacle_unused = True
                print("appearing")
                
            else:
                raise NotImplementedError("unknown behavior type")
            
            add_obs_to_cspace(S, ob)
    a.close()
    
def read_prismatic_obstacles_from_file(S: CSpace, file_name, total_dims):
    a = open(file_name, 'r')
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        prism_span_min = data[2:len(data):2]
        prism_span_max = data[3:len(data):2]
        for n in range(N):
            data = np.array(list(a.readline()))
            polygon.append(data[:2])
        
        add_obs_to_cspace(S, Obstacle(5, polygon=polygon, prism_span_min=prism_span_min, prism_span_max=prism_span_max))
            
    a.close()
    
def read_directional_obstacles_from_file(S: CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    P = int(a.readline())
    
    for p in range(P):
        D = a.readline()
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
        
        start_time_and_life_span = np.array(list(a.readline()))
        
        for i in range(obs_mult):
            if D[0] != 'X':
                add_obs_to_cspace(S, Obstacle(4, polygon=polygon, direction=D[0]))
            else:
                add_obs_to_cspace(S, Obstacle(3, polygon=polygon))
            
    a.close()
    
def read_time_obstacles_from_file(S:CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
        
        obs_speed = float(a.readline())
        M = int(a.readline())
        move_path = []
        
        for m in range(M):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            move_path.append(np.array(list(map(float, b))))
        
        for i in range(obs_mult):
            ob = Obstacle(6, polygon=polygon)
            ob.senseable_obstacle = False
            ob.obstacle_unused_after_sense = False
            ob.obstacle_unused = False
            ob.velocity = obs_speed
            ob.path = move_path
            ob.original_polygon = deepcopy(polygon)
            ob.next_direction_change_time = -math.inf
            
            add_obs_to_cspace(S, ob)
                
    a.close()
    
def read_dynamic_time_obstacles_from_file(S: CSpace, file_name, obs_mult):
    a = open(file_name, 'r')
    
    P = int(a.readline())
    
    for p in range(P):
        N = int(a.readline())
        
        polygon = []
        for n in range(N):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            polygon.append(np.array(list(map(float, b))))
            
        obs_speed = float(a.readline())
        M = int(a.readline())
        move_path = []
        for m in range(M):
            b = a.readline().split(',')
            b[-1] = b[-1].replace('\n','')
            move_path.append(np.array(list(map(float, b))))
        
        for i in range(obs_mult):
            ob = Obstacle(7, polygon=polygon)
            ob.senseable_obstacle = False
            ob.obstacle_unused_after_sense = False
            ob.obstacle_unused = False
            ob.velocity = obs_speed
            ob.unknown_path = move_path
            ob.next_direction_change_ind = len(move_path) - 1
            ob.original_polygon = deepcopy(polygon)
            ob.last_direction_change_time = math.inf
            change_obstacle_direction(S, ob, S.goal[2])
            add_obs_to_cspace(S, ob)
            
    a.close()