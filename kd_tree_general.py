import numpy as np
import math
from heap import BinaryHeap
from ghost_point import ghostPointIterator
import heap as hp
import ghost_point as gp

class KDTreeNode(object):
    def __init__(self,
                 kd_in_tree=False,
                 kd_parent_exist=False,
                 kd_child_l_exist=False,
                 kd_child_r_exist=False,
                 heap_index=-1,
                 in_heap=False):
        self.kd_in_tree = kd_in_tree
        self.kd_parent_exist = kd_parent_exist
        self.kd_child_l_exist = kd_child_l_exist
        self.kd_child_r_exist = kd_child_r_exist
        self.heap_index = heap_index
        self.in_heap = in_heap
        self.data = None
        self.position = None
        self.kd_split = None
        self.kd_parent: KDTreeNode = None
        self.kd_child_l: KDTreeNode = None
        self.kd_child_r: KDTreeNode = None
    
class KDTree(object):
    def __init__(self,
                 d=0,
                 distance_function=None,
                 tree_size=0,
                 num_wraps=0,
                 wraps=None,
                 wrap_points=None,
                 root=None
                 ):
        self.d = d
        self.distance_function = distance_function
        self.tree_size = tree_size
        if wraps is not None:
            self.num_wraps = len(wraps)
        else:
            self.num_wraps = num_wraps
        self.wraps = wraps
        self.wrap_points = wrap_points
        self.root: KDTreeNode = root
        
def kd_tree_init(K: KDTree, d, f):
    K.d = d
    K.f = f
    K.tree_size = 0

def kd_insert(tree: KDTree, this_node):
    if not hasattr(this_node, 'position'):
        node = type(tree.root)()
        node.position = this_node
    else:
        node = this_node
    
    if node.kd_in_tree:
        return
    node.kd_in_tree = True
    
    if tree.tree_size == 0:
        tree.root = node
        tree.root.kd_split = 0
        tree.tree_size = 1
        return

    parent: KDTreeNode = tree.root
    while True:
        if node.position[parent.kd_split] < parent.position[parent.kd_split]:
            if not parent.kd_child_l_exist:
                parent.kd_child_l = node
                parent.kd_child_l_exist = True
                break
            
            parent = parent.kd_child_l
            continue
        else:
            if not parent.kd_child_r_exist:
                parent.kd_child_r = node
                parent.kd_child_r_exist = True
                break
            
            parent = parent.kd_child_r
            continue
    
    node.kd_parent = parent
    node.kd_parent_exist = True
    if parent.kd_split == tree.d - 1:
        node.kd_split = 0
    else:
        node.kd_split = parent.kd_split + 1
    tree.tree_size += 1
    # tree.node_list.append(this_node)
    
def kd_print_sub_tree(node: KDTreeNode):
    print([node.kd_split], ":", node.position[node.kd_split], " ->  ")
    
    if node.kd_child_l_exist:
        print(node.kd_child_l.position[node.kd_split])
    else:
        print("NULL")
    
    print("    |    ")
    
    if node.kd_child_r_exist:
        print(node.kd_child_r.position[node.kd_split])
    else:
        print("NULL")
        
    if node.kd_child_l_exist:
        kd_print_sub_tree(node.kd_child_l)
    if node.kd_child_r_exist:
        kd_print_sub_tree(node.kd_child_r)

def kd_print_tree(tree: KDTree):
    if tree.tree_size == 0:
        print("tree is empty")
        return

    kd_print_sub_tree(tree.root)
    
def kd_find_nearest_in_subtree(distance_function, root: KDTreeNode, query_point, suggested_closest_node: KDTreeNode, suggested_closest_dist):
    parent: KDTreeNode = root
    current_closest_node: KDTreeNode = suggested_closest_node
    current_closest_dist = suggested_closest_dist
    while True:
        if query_point[parent.kd_split] < parent.position[parent.kd_split]:
            if not parent.kd_child_l_exist:
                break
            parent = parent.kd_child_l
            continue
        else:
            if not parent.kd_child_r_exist:
                break
            parent = parent.kd_child_r
            continue
        
    new_dist = distance_function(query_point, parent.position)
    if new_dist < current_closest_dist:
        current_closest_node = parent
        current_closest_dist = new_dist
    
    while True:
        parent_hyper_plane_dist = (query_point[parent.kd_split] - parent.position[parent.kd_split])
        if parent_hyper_plane_dist > current_closest_dist:
            if parent == root:
                return (current_closest_node, current_closest_dist)
            
            parent = parent.kd_parent
            continue
        
        if current_closest_node != parent:
            new_dist = distance_function(query_point, parent.position)
            if new_dist < current_closest_dist:
                current_closest_node = parent
                current_closest_dist = new_dist
    
        if query_point[parent.kd_split] < parent.position[parent.kd_split] and parent.kd_child_r_exist:
            r_node, r_dist = kd_find_nearest_in_subtree(distance_function, parent.kd_child_r, query_point, current_closest_node, current_closest_dist)
            
            if r_dist < current_closest_dist:
                current_closest_dist = r_dist
                current_closest_node = r_node
                
        elif parent.position[parent.kd_split] <= query_point[parent.kd_split] and parent.kd_child_l_exist:
            l_node, l_dist = kd_find_nearest_in_subtree(distance_function, parent.kd_child_l, query_point, current_closest_node, current_closest_dist)
            
            if l_dist < current_closest_dist:
                current_closest_dist = l_dist
                current_closest_node = l_node
        
        if parent == root:
            return (current_closest_node, current_closest_dist)
        
        parent = parent.kd_parent

def kd_find_nearest(tree: KDTree, query_point):
    dist_to_root = tree.distance_function(query_point, tree.root.position)
    l_node, l_dist = kd_find_nearest_in_subtree(tree.distance_function, tree.root, query_point, tree.root, dist_to_root)
    
    if tree.num_wraps > 0:
        point_iterator = ghostPointIterator(tree, query_point)
        while True:
            this_ghost_point = gp.get_next_ghost_point(point_iterator, l_dist)
            if this_ghost_point is None:
                break
            
            dist_ghost_to_root = tree.distance_function(this_ghost_point, tree.root.position)
            this_l_node, this_l_dist = kd_find_nearest_in_subtree(tree.distance_function, tree.root, this_ghost_point, tree.root, dist_ghost_to_root)
            if this_l_dist < l_dist:
                l_dist = this_l_dist
                l_node = this_l_node
    return (l_node, l_dist)

def add_to_range_list(S, this_node: KDTreeNode, key):
    if this_node.in_heap:
        return
    
    this_node.in_heap = True
    S.append((this_node, key))
    
def pop_from_range_list(S):
    this_node, key = S.pop(-1)
    this_node.in_heap = False
    
    return this_node, key

def empty_range_list(S):
    for node, key in S:
        node.in_heap = False
    S.clear()
    
def empty_and_print_range_list(S):
    while len(S) > 0:
        node = pop_from_range_list(S)
        print(node.data)

def kd_find_within_range_in_subtree(distance_function, root: KDTreeNode, range, query_point, node_list):
    parent: KDTreeNode = root
    while True:
        if query_point[parent.kd_split] < parent.position[parent.kd_split]:
            if not parent.kd_child_l_exist:
                break
            parent = parent.kd_child_l
            continue
        else:
            if not parent.kd_child_r_exist:
                break
            parent = parent.kd_child_r
            continue
        
    new_dist = distance_function(query_point, parent.position)
    if new_dist <= range:
        add_to_range_list(node_list, parent, new_dist)
    
    while True:
        parent_hyper_plane_dist = query_point[parent.kd_split] - parent.position[parent.kd_split]
        if parent_hyper_plane_dist > range:
            if parent == root:
                return
            
            parent = parent.kd_parent
            continue
        
        if not parent.in_heap:
            new_dist = distance_function(query_point, parent.position)
            if new_dist <= range:
                add_to_range_list(node_list, parent, new_dist)
            
        if query_point[parent.kd_split] < parent.position[parent.kd_split] and parent.kd_child_r_exist:
            kd_find_within_range_in_subtree(distance_function, parent.kd_child_r, range, query_point, node_list)
        elif parent.position[parent.kd_split] <= query_point[parent.kd_split] and parent.kd_child_l_exist:
            kd_find_within_range_in_subtree(distance_function, parent.kd_child_l, range, query_point, node_list)
        if parent == root:
            return
        
        parent = parent.kd_parent
            
def kd_find_within_range(tree: KDTree, range, query_point):
    L = []
    dist_to_root = tree.distance_function(query_point, tree.root.position)
    if dist_to_root <= range:
        add_to_range_list(L, tree.root, dist_to_root)
        
    
    kd_find_within_range_in_subtree(tree.distance_function, tree.root, range, query_point, L)
    
    if tree.num_wraps > 0:
        point_iterator = ghostPointIterator(tree, query_point)
        while True:
            this_ghost_point = gp.get_next_ghost_point(point_iterator, range)
            if this_ghost_point is None:
                break
            
            kd_find_within_range_in_subtree(tree.distance_function, tree.root, range, this_ghost_point, L)

    return L

def kd_find_more_within_range(tree: KDTree, range, query_point, L):
    dist_to_root = tree.distance_function(query_point, tree.root.position)
    if dist_to_root <= range:
        add_to_range_list(L, tree.root, dist_to_root)
        
    kd_find_within_range_in_subtree(tree.distance_function, tree.root, range, query_point, L)
    
    if tree.num_wraps > 0:
        point_iterator = ghostPointIterator(tree, query_point)
        while True:
            this_ghost_point = gp.get_next_ghost_point(point_iterator, range)
            if this_ghost_point is None:
                break
            
            kd_find_within_range_in_subtree(tree.distance_function, tree.root, range, this_ghost_point, L)
    return L