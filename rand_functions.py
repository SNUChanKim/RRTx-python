import numpy as np
import math
from copy import deepcopy
from data_structure import CSpace, RRTNode, RobotData
import collision_checking_functions as ccf

def rand_point_default(S: CSpace):
    return S.lower_bounds + np.random.rand(S.d)*S.width

def rand_node_default(S: CSpace):
    return RRTNode(S.lower_bounds + np.random.rand(S.d)*S.width)

def rand_node_near_robot(S: CSpace, R: RobotData):
    position = np.clip(R.robot_pose + S.width/2.0*np.random.randn(S.d), S.lower_bounds, S.upper_bounds)
    return RRTNode(position)

def rand_node_or_goal(S: CSpace):
    if np.random.rand() > S.p_goal:
        return rand_node_default(S)
    
    return S.goal_node

def rand_node_its(S: CSpace):
    if S.its_until_sample == 0:
        S.its_sample_point -= 1
        return RRTNode(S.its_sample_point)
    S.its_until_sample -= 1
    return rand_node_or_goal(S)

def rand_node_time(S: CSpace):
    if not np.isinf(S.wait_time) and S.elapsed_time >= S.wait_time:
        S.wait_time = math.inf
        return RRTNode(S.time_sample_point)
    return rand_node_or_goal(S)

def rand_node_time_with_obstacle_remove(S: CSpace):
    if not np.isinf(S.wait_time) and S.elapsed_time >= S.wait_time:
        S.wait_time = math.inf
        S.obstacles_to_remove.obstacle_unused = True
        S.obstacles_to_remove.start_time = -math.inf
        S.obstacles_to_remove.life_span = 0.0
        S.obstacles_to_remove.senseable_obstacle = False
        S.obstacles_to_remove.obstacle_unused_after_sense = True
        
        print("XXXXXXXXXXXXXXXXXXXXXXXX sample XXXXXXXXXXXXXXXXXXXXXXXX")
        a = S.time_sample_point
        return RRTNode(a)
    return rand_node_or_goal(S)

def rand_node_its_with_obstacle_remove(S: CSpace):
    S.its_until_sample -= 1
    if S.its_until_sample == 0:
        S.wait_time = math.inf
        S.obstacles_to_remove.obstacle_unused = True
        S.obstacles_to_remove.start_time = -math.inf
        S.obstacles_to_remove.life_span = 0.0
        S.obstacles_to_remove.senseable_obstacle = False
        S.obstacles_to_remove.obstacle_unused_after_sense = True
        
        print("XXXXXXXXXXXXXXXXXXXXXXXX sample XXXXXXXXXXXXXXXXXXXXXXXX")
        a = S.time_sample_point
        return RRTNode(a)
    return rand_node_or_goal(S)

def rand_node_or_from_stack(S: CSpace):
    if len(S.sample_stack) > 0:
        return RRTNode(S.sample_stack.pop(-1))
    else:
        return rand_node_or_goal(S)
    
def rand_node_in_time_or_from_stack(S: CSpace):
    if len(S.sample_stack) > 0:
        return RRTNode(S.sample_stack.pop(-1))
    else:
        new_node = rand_node_or_goal(S)
        if new_node == S.goal_node:
            return new_node

        min_time_to_reach_node = S.start[2] + np.linalg.norm(np.array([new_node.position[0] - S.root.position[0], new_node.position[1] - S.root.position[1]]))/S.robot_velocity
        
        if (new_node.position[2] < min_time_to_reach_node or 
            (new_node.position[2] > S.move_goal.position[2] and
            S.move_goal != S.goal_node)):
            
            new_node.position[2] = min_time_to_reach_node + np.random.rand()*(S.move_goal.position[2] - min_time_to_reach_node)
        
        return new_node