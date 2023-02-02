import numpy as np
import math
import heap
from copy import deepcopy

class ghostPointIterator(object):
    def __init__(self,
                 kd_tree,
                 query_point):
        self.kd_tree = kd_tree
        self.query_point = deepcopy(query_point)
        self.wrap_dim_flags = np.zeros([kd_tree.num_wraps])
        self.ghost_tree_depth = kd_tree.num_wraps - 1
        self.current_ghost = query_point
        self.closest_unwrapped_point = query_point
        
def get_next_ghost_point(G: ghostPointIterator, best_dist):
    while True:
        while G.ghost_tree_depth >= 0 and G.wrap_dim_flags[G.ghost_tree_depth] != 0:
            G.ghost_tree_depth -= 1
        
        if G.ghost_tree_depth == -1:
            return None
        
        G.wrap_dim_flags[G.ghost_tree_depth] = 1
        
        dim_val = G.query_point[G.kd_tree.wraps[G.ghost_tree_depth]]
        dim_closest = 0.0
        if G.query_point[G.kd_tree.wraps[G.ghost_tree_depth]] < G.kd_tree.wrap_points[G.ghost_tree_depth]/2.0:
            dim_val += G.kd_tree.wrap_points[G.ghost_tree_depth]
            dim_closest += G.kd_tree.wrap_points[G.ghost_tree_depth]
        else:
            dim_val -= G.kd_tree.wrap_points[G.ghost_tree_depth]
            
        G.current_ghost[G.kd_tree.wraps[G.ghost_tree_depth]] = dim_val
        G.closest_unwrapped_point[G.kd_tree.wraps[G.ghost_tree_depth]] = dim_closest
        
        while G.ghost_tree_depth < G.kd_tree.num_wraps - 1:
            G.ghost_tree_depth += 1
            G.wrap_dim_flags[G.ghost_tree_depth] = 0
            G.current_ghost[G.kd_tree.wraps[G.ghost_tree_depth]] = G.query_point[G.kd_tree.wraps[G.ghost_tree_depth]]
            G.closest_unwrapped_point[G.kd_tree.wraps[G.ghost_tree_depth]] = G.current_ghost[G.kd_tree.wraps[G.ghost_tree_depth]]
        
        if G.kd_tree.distance_function(G.closest_unwrapped_point, G.current_ghost) > best_dist:
            continue
            
        return G.current_ghost