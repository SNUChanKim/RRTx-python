import numpy as np
import math
from copy import deepcopy
from data_structure import RRTNode, CSpace, RobotData
from kd_tree_general import KDTree
# import simple_edge_functions as ef
# import dubins_edge_functions as ef
import edge_functions as ef
import csv

def save_rrt_sub_tree(node: RRTNode, root: RRTNode, writer):
    
    if node.rrt_parent_used:
        writer.writerow(list(node.position) + [node.rrt_tree_cost])
        writer.writerow(list(node.rrt_parent_edge.end_node.position) + [node.rrt_parent_edge.end_node.rrt_tree_cost])
        
    if node.kd_child_l_exist:
        save_rrt_sub_tree(node.kd_child_l, root, writer)
    if node.kd_child_r_exist:
        save_rrt_sub_tree(node.kd_child_r, root, writer)
        
def save_rrt_tree(tree: KDTree, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    if tree.tree_size > 0:
        save_rrt_sub_tree(tree.root, tree.root, writer)
    
    fptr.close()
    
def save_rrt_sub_graph(node: RRTNode, writer):
    for out_neighbor in node.rrt_neighbors_out:
        writer.writerow(list(node.position))
        writer.writerow(list(out_neighbor.position))
    
    if node.kd_child_l_exist:
        save_rrt_sub_graph(node.kd_child_l, writer)
    
    if node.kd_child_r_exist:
        save_rrt_sub_graph(node.kd_child_r, writer)
        
def save_rrt_graph(tree: KDTree, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    if tree.tree_size > 0:
        save_rrt_sub_graph(tree.root, writer)
    fptr.close()
    
def save_rrt_sub_nodes(node: RRTNode, writer):
    writer.writerow(list(node.position) + [node.rrt_tree_cost, node.rrt_lmc])
    
    if node.kd_child_l_exist:
        save_rrt_sub_nodes(node.kd_child_l, writer)
    
    if node.kd_child_r_exist:
        save_rrt_sub_nodes(node.kd_child_r, writer)
        
def save_rrt_nodes(tree: KDTree, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    if tree.tree_size > 0:
        save_rrt_sub_nodes(tree.root, writer)
    else:
        print("tree size = 0")
    fptr.close()
    
def save_rrt_sub_nodes_collision(node: RRTNode, writer):
    writer.write(list(node.position) + [np.minimum(node.rrt_tree_cost, node.rrt_lmc)])
    
    if node.kd_child_l_exist:
        save_rrt_sub_nodes_collision(node.kd_child_l, writer)
    
    if node.kd_child_r_exist:
        save_rrt_sub_nodes_collision(node.kd_child_r, writer)
    
def save_rrt_nodes_collision(tree: KDTree, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    
    if tree.tree_size > 0 :
        save_rrt_sub_nodes_collision(tree.root, writer)
    else:
        print("tree size = 0")
    fptr.close()  
    
def save_rrt_path(S: CSpace, node: RRTNode, root: RRTNode, robot: RobotData, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    this_node = node
    
    if robot.robot_edge_for_plotting_used and robot.robot_edge_for_plotting.start_node != this_node:
        if S.space_has_time:
            ef.save_end_of_trajectory_time(S, writer, robot.robot_edge_for_plotting, robot.time_along_robot_edge_for_plotting)
        else:
            ef.save_end_of_trajectory(S, writer, robot.robot_edge_for_plotting, robot.dist_along_robot_edge_for_plotting)
            
    i = 0
    while this_node != root and this_node.rrt_parent_used and i < 1000:
        ef.save_edge_trajectory(S, writer, this_node.rrt_parent_edge)
        this_node = this_node.rrt_parent_edge.end_node
        i += 1
    
    if S.space_has_time:
        writer.writerow(list(this_node.position[:3]))
    else:
        writer.writerow(list(this_node.position[:2]))
    
    fptr.close()
                
def save_data(data, file_name):
    fptr = open(file_name, "w")
    writer = csv.writer(fptr)
    for i in range(len(data)):
        writer.writerow(list(data[i][:]))
    fptr.close()
        
def extract_path_length(node: RRTNode, root: RRTNode):
    path_length = 0.0
    this_node = node
    while this_node != root:
        if not this_node.rrt_parent_used:
            path_length = math.inf
            break
        
        path_length += this_node.rrt_parent_edge.dist
        this_node = this_node.rrt_parent_edge.end_node
    
    return path_length