from cmath import inf
import os
from re import S
import sys
from this import d

cur_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
parent_dir = os.path.dirname(cur_dir)
sys.path.append(parent_dir)

from copy import deepcopy
import numpy as np
import math

import collision_checking_functions as ccf
import geometric_functions as gf
# import simple_edge_functions as ef
# import dubins_edge_functions as ef
import edge_functions as ef
import distance_functions as df
import rand_functions as rf
import obstacle_functions as of
import save_functions as sf
import kd_tree_general as kd
import heap as hp
# from simple_edge import SimpleEdge as Edge
from edge import Edge
from data_structure import RRTNode, rrtXQueue, RobotData, Obstacle, CSpace
from kd_tree_general import KDTree

    
def key_q(node: RRTNode):
    g_min = np.minimum(node.rrt_tree_cost, node.rrt_lmc)
    return (g_min + 0.0, g_min)

def less_q(a: RRTNode, b: RRTNode):
    a_key_first, a_key_second = key_q(a)
    b_key_first, b_key_second = key_q(b)
    
    if ((a_key_first < b_key_first) or 
        (a_key_first == b_key_first and a_key_second < b_key_second) or
        (a_key_first == b_key_first and a_key_second == b_key_second and a.is_move_goal)):
        return True
    return False

def greater_q(a: RRTNode, b: RRTNode):
    a_key_first, a_key_second = key_q(a)
    b_key_first, b_key_second = key_q(b)
    
    if ((a_key_first > b_key_first) or 
        (a_key_first == b_key_first and a_key_second > b_key_second) or
        (a_key_first == b_key_first and a_key_second == b_key_second and b.is_move_goal)):
        return True
    return False

def mark_q(node: RRTNode):
    node.in_priority_queue = True

def unmark_q(node: RRTNode):
    node.in_priority_queue = False

def marked_q(node: RRTNode):
    return node.in_priority_queue

def set_index_q(node: RRTNode, val):
    node.priority_queue_index = val

def unset_index_q(node: RRTNode):
    node.priority_queue_index = -1
    
def get_index_q(node: RRTNode):
    return node.priority_queue_index

def mark_os(node: RRTNode):
    node.in_os_queue = True

def unmark_os(node: RRTNode):
    node.in_os_queue = False

def marked_os(node: RRTNode):
    return node.in_os_queue
    
def verify_in_queue(Q: rrtXQueue, node: RRTNode):
    if marked_q(node):
        hp.update_heap(Q.Q, node)
    else:
        hp.add_to_heap(Q.Q, node)
        
def verify_in_os_queue(Q: rrtXQueue, node: RRTNode):
    if marked_q(node):
        hp.update_heap(Q.Q, node)
        hp.remove_from_heap(Q.Q, node)
    
    if not marked_os(node):
        mark_os(node)
        Q.OS.append(node)
 
def make_neighbor_of(new_neighbor, node, edge): #node= start, new_neighbor=end
    if not np.array_equal(node.position, edge.start_node.position):
        print("node.position: ", node.position)
        print("start_node.position: ", edge.start_node.position)
        print("end_node.position: ", edge.end_node.position)
        raise ValueError("This is not rrt out neighbor!")
    node.rrt_neighbors_out.append(edge)
    edge.list_item_in_start_node = edge
    
    if not np.array_equal(new_neighbor.position, edge.end_node.position):
        print("node.position: ", new_neighbor.position)
        print("start_node.position: ", edge.start_node.position)
        print("end_node.position: ", edge.end_node.position)
        raise ValueError("This is not rrt in neighbor!")
    new_neighbor.rrt_neighbors_in.append(edge)
    edge.list_item_in_end_node = edge
    return 0

def make_initial_out_neighbor_of(new_neighbor: RRTNode, node: RRTNode, edge: Edge):
    if not np.array_equal(node.position, edge.start_node.position):
        print("node.position: ",node.position)
        print("start_node.position: ", edge.start_node.position)
        print("end_node.position: ", edge.end_node.position)
        raise ValueError("This is not out neighbor!")
    node.initial_neighbor_list_out.append(edge)

def make_initial_in_neighbor_of(new_neighbor: RRTNode, node: RRTNode, edge: Edge):
    if not np.array_equal(node.position, edge.end_node.position):
        print("node.position: ",node.position)
        print("start_node.position: ", edge.start_node.position)
        print("end_node.position: ", edge.end_node.position)
        raise ValueError("This is not in neighbor!")
    node.initial_neighbor_list_in.append(edge)
    
def make_parent_of(new_parent: RRTNode, node: RRTNode, edge: Edge, root=None):
    if node.rrt_parent_used:
        node.rrt_parent_edge.end_node.successor_list.remove(node.successor_list_item_in_parent)
    
    if not np.array_equal(node.position, edge.start_node.position) or not np.array_equal(new_parent.position, edge.end_node.position):
        print("node.position: ",node.position)
        print("new_parent.position: ",new_parent.position)
        print("start_node.position: ", edge.start_node.position)
        print("end_node.position: ", edge.end_node.position)
        raise ValueError("This is not this node's parent!")
    
    node.rrt_parent_edge = edge
    node.rrt_parent_used = True
    
    back_edge = ef.new_edge(new_parent, node)
    back_edge.dist = math.inf
    edge_key_pair = (back_edge, math.inf)
    new_parent.successor_list.append(edge_key_pair)
    
    node.successor_list_item_in_parent = edge_key_pair

def find_best_parent(S: CSpace, new_node: RRTNode, node_list, closest_node: RRTNode, hyper_ball_rad, save_all_edges):
    if len(node_list) == 0:
        if S.goal_node != new_node:
            node_list.append((closest_node, None))
    
    new_node.rrt_lmc = math.inf
    new_node.rrt_tree_cost = math.inf
    new_node.rrt_parent_used = False
    
    new_parent_found = False
    rrt_parent = None
    rrt_parent_edge = None
    
    for near_node, key in node_list:
        this_edge = ef.new_edge(new_node, near_node)
        ef.calculate_trajectory(S, this_edge)
        
        near_node.temp_edge = this_edge
        
        if ccf.explicit_edge_check(S, this_edge) or not ef.valid_move(S, this_edge):
            near_node.temp_edge.dist = math.inf
            continue
        
        if (new_node.rrt_lmc > near_node.rrt_lmc + this_edge.dist):
            new_node.rrt_lmc = near_node.rrt_lmc + this_edge.dist
            rrt_parent_edge = this_edge
            rrt_parent = near_node
            new_parent_found = True
        
    if new_parent_found:
        make_parent_of(rrt_parent, new_node, rrt_parent_edge)

def cull_current_neighbors(this_node: RRTNode, hyper_ball_rad):
    for neighbor_edge in this_node.rrt_neighbors_out:
        if not np.array_equal(neighbor_edge.start_node.position, this_node.position):
            raise ValueError("Edge's start_node is not this node")
        
        if neighbor_edge.dist > hyper_ball_rad:
            neighbor_node: RRTNode = neighbor_edge.end_node
            this_node.rrt_neighbors_out.remove(neighbor_edge.list_item_in_start_node)
            if neighbor_edge.list_item_in_end_node not in neighbor_node.rrt_neighbors_in:
                raise ValueError("this edge is not in neighbor node's 'in neighbor'.")
            neighbor_node.rrt_neighbors_in.remove(neighbor_edge.list_item_in_end_node)
                

def recalculate_lmc_mine_v_two(Q: rrtXQueue, this_node: RRTNode, root: RRTNode, hyper_ball_rad):
    if this_node == root:
        return
    
    new_parent_found = False
    rrt_parent = None
    rrt_parent_edge = None
    
    cull_current_neighbors(this_node, hyper_ball_rad)
    
    out_neighbors = this_node.initial_neighbor_list_out + this_node.rrt_neighbors_out
    for neighbor_edge in out_neighbors:
        neighbor_node: RRTNode = neighbor_edge.end_node
        neighbor_dist = neighbor_edge.dist
        if marked_os(neighbor_node):
            continue
        
        if not np.array_equal(neighbor_edge.start_node.position, this_node.position):
            print('----------------------------------')
            print(len(out_neighbors))
            print("neighbor_edge.start_node.position: ", neighbor_edge.start_node.position)
            print("neighbor_edge.end_node.position: ", neighbor_edge.end_node.position)
            print("this_node.position: ", this_node.position)
            print("this_node.out_neighbor_counter: ", this_node.out_neighbor_counter)
            print("this_node.initial_out_neighbor_counter: ", this_node.initial_out_neighbor_counter)
            raise ValueError
        
        if (this_node.rrt_lmc > neighbor_node.rrt_lmc + neighbor_dist and 
            (not neighbor_node.rrt_parent_used or neighbor_node.rrt_parent_edge.end_node != this_node) and
            ef.valid_move(Q.S, neighbor_edge)):
            
            this_node.rrt_lmc = neighbor_node.rrt_lmc + neighbor_dist
            rrt_parent = neighbor_node
            rrt_parent_edge = neighbor_edge
            new_parent_found = True
    
    if new_parent_found:
        make_parent_of(rrt_parent, this_node, rrt_parent_edge, root)
    
        
def extend(S: CSpace, KD: KDTree, Q: rrtXQueue, new_node: RRTNode, closest_node: RRTNode, delta, hyper_ball_rad, move_goal: RRTNode):
    node_list = kd.kd_find_within_range(KD, hyper_ball_rad, new_node.position)
    find_best_parent(S, new_node, node_list, closest_node, hyper_ball_rad, True)
    
    if not new_node.rrt_parent_used:
        kd.empty_range_list(node_list)
        return

    kd.kd_insert(KD, new_node)
    # kd.kd_print_tree(KD)
    
    for near_node, key in node_list:
        if near_node.temp_edge.dist != math.inf:
            make_initial_out_neighbor_of(near_node, new_node, near_node.temp_edge)
            make_initial_in_neighbor_of(new_node, near_node, near_node.temp_edge)
            make_neighbor_of(near_node, new_node, near_node.temp_edge)
        
        this_edge = ef.new_edge(near_node, new_node)
        ef.calculate_trajectory(S, this_edge)
        
        if ef.valid_move(S, this_edge) and not ccf.explicit_edge_check(S, this_edge):
            make_initial_out_neighbor_of(new_node, near_node, this_edge)
            make_initial_in_neighbor_of(near_node, new_node, this_edge)
            make_neighbor_of(new_node, near_node, this_edge)
        else:
            continue
        
        if (near_node.rrt_lmc > new_node.rrt_lmc + this_edge.dist and 
            new_node.rrt_parent_edge.end_node != near_node and
            new_node.rrt_lmc + this_edge.dist < move_goal.rrt_lmc):
            
            make_parent_of(new_node, near_node, this_edge, KD.root)
            
            old_lmc = near_node.rrt_lmc
            near_node.rrt_lmc = new_node.rrt_lmc + this_edge.dist
            
            if old_lmc - near_node.rrt_lmc > Q.change_thresh and near_node != KD.root:
                verify_in_queue(Q, near_node)
    
    kd.empty_range_list(node_list)
    verify_in_queue(Q, new_node)
    
def rewire(Q: rrtXQueue, this_node: RRTNode, root: RRTNode, hyper_ball_rad, change_thresh):
    delta_cost = this_node.rrt_tree_cost - this_node.rrt_lmc
    if delta_cost <= change_thresh:
        return
    cull_current_neighbors(this_node, hyper_ball_rad)
    
    in_neighbors = this_node.initial_neighbor_list_in + this_node.rrt_neighbors_in
    for neighbor_edge in in_neighbors:
        neighbor_node: RRTNode = neighbor_edge.start_node
        if ((this_node.rrt_parent_used and this_node.rrt_parent_edge.end_node == neighbor_node) or 
            not ef.valid_move(Q.S, neighbor_edge)):
            continue
        
        if not np.array_equal(neighbor_edge.end_node.position, this_node.position):
            print("this_node.position: ", this_node.position, this_node.is_move_goal)
            print("start_node.position: ", neighbor_edge.start_node.position)
            print("end_node.position: ", neighbor_edge.end_node.position)
            raise ValueError("Edge's end_node is not this node")
        
        if neighbor_node.rrt_lmc > this_node.rrt_lmc + neighbor_edge.dist:
            neighbor_node.rrt_lmc = this_node.rrt_lmc + neighbor_edge.dist
            
            if not neighbor_node.rrt_parent_used or neighbor_node.rrt_parent_edge.end_node != this_node:
                make_parent_of(this_node, neighbor_node, neighbor_edge, root)
            
            if neighbor_node.rrt_tree_cost - neighbor_node.rrt_lmc > change_thresh:
                verify_in_queue(Q, neighbor_node)
                

def reduce_inconsistency(Q: rrtXQueue, goal_node: RRTNode, robot_rad, root: RRTNode, hyper_ball_rad):
    while (Q.Q.index_of_last > -1 and 
            (less_q(hp.top_heap(Q.Q), goal_node) or np.isinf(goal_node.rrt_lmc) or np.isinf(goal_node.rrt_tree_cost) or marked_q(goal_node))):
        this_node: RRTNode = hp.pop_heap(Q.Q)
        if this_node.rrt_tree_cost - this_node.rrt_lmc > Q.change_thresh:
            recalculate_lmc_mine_v_two(Q, this_node, root, hyper_ball_rad)
            rewire(Q, this_node, root, hyper_ball_rad, Q.change_thresh)
                
        this_node.rrt_tree_cost = this_node.rrt_lmc
        
        
def propogate_descendants(Q: rrtXQueue, R: RobotData):
    if len(Q.OS) <= 0:
        return
    
    counter = 0
    while counter < len(Q.OS):
        this_node = Q.OS[counter]
        for successor_list_edge, key in this_node.successor_list:
            successor_node: RRTNode = successor_list_edge.end_node
            
            verify_in_os_queue(Q, successor_node)
        
        counter += 1
        
    for this_node in reversed(Q.OS):
        
        out_neighbors = this_node.initial_neighbor_list_out + this_node.rrt_neighbors_out
        for neighbor_edge in out_neighbors:
            if neighbor_edge.start_node != this_node:
                raise ValueError("Edge's start_node is not this node")
            neighbor_node: RRTNode = neighbor_edge.end_node
            
            if marked_os(neighbor_node):
                continue
            
            neighbor_node.rrt_tree_cost = math.inf
            verify_in_queue(Q, neighbor_node)
            
        if this_node.rrt_parent_used and not marked_os(this_node.rrt_parent_edge.end_node):
            this_node.rrt_parent_edge.end_node.rrt_tree_cost = math.inf
            verify_in_queue(Q, this_node.rrt_parent_edge.end_node)
    
    while len(Q.OS) > 0:
        this_node = Q.OS.pop(0)
        unmark_os(this_node)
        
        if this_node == R.next_move_target:
            R.current_move_invalid = True
        
        if this_node.rrt_parent_used:
            this_node.rrt_parent_edge.end_node.successor_list.remove(this_node.successor_list_item_in_parent)
            this_node.rrt_parent_edge = ef.new_edge(this_node, this_node)
            this_node.rrt_parent_edge.dist = math.inf
            this_node.rrt_parent_used = False
            
        this_node.rrt_tree_cost = math.inf
        this_node.rrt_lmc = math.inf
        
def add_other_times_to_root(S: CSpace, KD: KDTree, goal: RRTNode, root: RRTNode):
    
    insert_step = 2.0
    last_time_to_insert = goal.position[2] - ef.w_dist(root, goal)/S.robot_velocity
    first_time_to_insert = S.start[2] + insert_step
    previous_node = root
    safe_to_goal = True
    for time_to_insert in np.arange(first_time_to_insert, last_time_to_insert+insert_step, insert_step):
        new_pose = deepcopy(root.position)
        new_pose[2] = time_to_insert
        
        new_node = RRTNode(new_pose)
        
        this_edge = ef.new_edge(new_node, previous_node)
        ef.calculate_hover_trajectory(S, this_edge)
        
        make_parent_of(previous_node, new_node, this_edge, root)
        make_initial_out_neighbor_of(previous_node, new_node, this_edge)
        make_initial_in_neighbor_of(new_node, previous_node, this_edge)
    
        if ccf.explicit_edge_check(S, this_edge):
            this_edge.dist = math.inf
            safe_to_goal = False
            new_node.rrt_lmc = math.inf
            new_node.rrt_tree_cost = math.inf
        elif safe_to_goal:
            this_edge.dist = 0.0
            new_node.rrt_lmc = 0.0
            new_node.rrt_tree_cost = 0.0
        else:
            this_edge.dist = math.inf
            new_node.rrt_lmc = math.inf
            new_node.rrt_tree_cost = math.inf
        
        kd.kd_insert(KD, new_node)
        
        previous_node = new_node
        
def find_points_in_conflict_with_obstacles(S: CSpace, KD: KDTree, ob: Obstacle, root: RRTNode):
    L = []
    if 1 <= ob.kind <= 5:
        if not S.space_has_time and not S.space_has_theta:
            search_range = S.robot_radius + S.delta + ob.radius
            L = kd.kd_find_within_range(KD, search_range, ob.position)
        elif not S.space_has_time and S.space_has_theta:
            search_range = S.robot_radius + S.delta + ob.radius + math.pi
            obs_center_dubins = np.array([ob.position[0], ob.position[1], 0.0, math.pi])
            L = kd.kd_find_within_range(KD, search_range, obs_center_dubins)
        else:
            raise NotImplementedError("this type of obstacle not coeded for this type of space")
        
    elif 6 <= ob.kind <= 7:
        base_search_range = S.robot_radius + S.delta + ob.radius
        for i in range(len(ob.path)):
            if len(ob.path) == 1:
                j = 0
            else:
                j = i+1
            
            query_pose = deepcopy(ob.position)
            query_pose = np.insert(query_pose, query_pose.size, 0.0)
            query_pose = query_pose + (ob.path[i] + ob.path[j])/2.0
            if S.space_has_theta:
                query_pose = np.insert(query_pose, query_pose.size, math.pi)
                
            search_range = base_search_range + df.euclidian_dist(ob.path[i], ob.path[j])/2.0
            
            if S.space_has_theta:
                search_range += math.pi
            
            if i == 0:
                L = kd.kd_find_within_range(KD, search_range, query_pose)
            else:
                kd.kd_find_more_within_range(KD, search_range, query_pose, L)
            
            if j == len(ob.path) - 1:
                break
    else:
        raise NotImplementedError("this case not coded yet")
    
    return L            