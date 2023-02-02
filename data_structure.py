import numpy as np
import math
from heap import BinaryHeap as Heap
# from simple_edge import SimpleEdge as Edge
from edge import Edge
# from dubins_edge import DubinsEdge as Edge

class RRTNode(object):
    def __init__(self, 
                 position,
                 kd_in_tree=False, 
                 kd_parent_exist=False, 
                 kd_child_l_exist=False, 
                 kd_child_r_exist=False,
                 heap_index=-1,
                 in_heap=False,
                 rrt_parent_used=False,
                 priority_queue_index=-1,
                 in_priority_queue=False,
                 in_os_queue=False,
                 is_move_goal=False,
                 ):
        
        self.kd_in_tree = kd_in_tree
        self.kd_parent_exist = kd_parent_exist
        self.kd_in_tree = kd_in_tree
        self.kd_parent_exist = kd_parent_exist 
        self.kd_child_l_exist = kd_child_l_exist
        self.kd_child_r_exist = kd_child_r_exist
        self.heap_index = heap_index
        self.in_heap = in_heap
        self.rrt_parent_used = rrt_parent_used
        self.rrt_neighbors_out = []
        self.rrt_neighbors_in = []
        self.priority_queue_index = priority_queue_index
        self.in_priority_queue = in_priority_queue
        self.successor_list = []
        self.initial_neighbor_list_out = []
        self.initial_neighbor_list_in = []
        self.in_os_queue = in_os_queue
        self.is_move_goal = is_move_goal
        self.position = position
        self.kd_split = None
        self.kd_parent: RRTNode = None
        self.kd_child_l: RRTNode = None
        self.kd_child_r: RRTNode = None
        self.rrt_parent_edge: Edge = None
        self.rrt_tree_cost = None
        self.rrt_lmc = None
        self.rrt_h = None
        self.temp_edge: Edge = None
        self.successor_list_item_in_parent = None
        self.has_certificate = False
        self.certificate_value = None
        self.certifying_node = None  
        self.out_neighbor_counter = 0
        self.initial_out_neighbor_counter = 0
        self.in_neighbor_counter = 0
        self.initial_in_neighbor_counter = 0
        
class Obstacle(object):
    def __init__(self,
                 kind,
                 position=None,
                 radius=None,
                 span=None,
                 polygon = None,
                 direction = None,
                 prism_span_min = None,
                 prism_span_max = None,
                 start_time=0.0,
                 life_span=math.inf,
                 obstacle_unused=False,
                 senseable_obstacle=False,
                 obstacle_unused_after_sense=True,
                 ):
        self.kind = kind
        self.start_time = start_time
        self.life_span = life_span
        self.obstacle_unused = obstacle_unused
        self.senseable_obstacle = senseable_obstacle
        self.obstacle_unused_after_sense = obstacle_unused_after_sense
        self.position = position
        self.radius = radius
        self.span = span
        self.polygon = polygon
        self.direction = direction
        self.prism_span_min = prism_span_min
        self.prism_span_max = prism_span_min
        self.velocity = None
        self.path = None
        self.original_polygon = None
        self.unknown_path = None
        self.next_direction_change_time = None
        self.next_direction_change_ind = None
        self.last_direction_change_time = None
        
        if radius is None and span is not None:
            self.radius = np.linalg.norm(span)
        
        if polygon is not None:
            polygon_array = np.array(polygon)
            max_x = np.max(polygon_array[:,0])
            min_x = np.min(polygon_array[:,0])
            max_y = np.max(polygon_array[:,1])
            min_y = np.min(polygon_array[:,1])
            self.position = np.array([max_x + min_x, max_y + min_y])/2.0
            self.radius = np.sqrt(np.max(np.sum((polygon_array - self.position)**2, 1)))
            self.span = np.array([-1.0, 1.0])

class CSpace(object):
    def __init__(self,
                 d,
                 obs_delta,
                 lower_bounds,
                 upper_bounds,
                 start,
                 goal):
        
        self.d = d
        self.obstacles = []
        self.obs_delta = obs_delta
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.width = upper_bounds - lower_bounds
        self.start = start
        self.goal = goal
        self.space_has_time = False
        self.space_has_theta = False
        self.p_goal = None
        self.rand_node = None
        self.goal_node: RRTNode = None
        self.root: RRTNode = None
        self.move_goal: RRTNode = None
        self.its_until_sample = None
        self.its_sample_point = None
        self.time_sample_point = None
        self.wait_time = None
        self.start_time_ns = None
        self.elapsed_time = None
        self.obstacles_to_remove: Obstacle = None
        self.robot_radius = None
        self.robot_velocity = None
        self.dubins_min_velocity = None
        self.dubins_max_velocity = None
        self.sample_stack = None
        self.hyper_volume = 0.0
        self.delta = None
        self.min_turning_radius = None
        self.file_ctr = None
        self.warmup_time = 0.0
        self.in_warmup_time = False
        
class rrtXQueue(object):
    def __init__(self):
        self.Q: Heap = None
        self.OS = None
        self.S: CSpace = None
        self.change_thresh = None
        
class RRTNodeNeighborIterator(object):
    def __init__(self,
                 this_node,
                 list_flag=0,
                 list_item=None):
        self.this_node = this_node
        self.list_flag = 0
        self.list_itemNode = list_item
        self.counter = 0
        
        
class RobotData(object):
    def __init__(self,
                 robot_pose,
                 next_move_target,
                 max_path_nodes):
        
        self.robot_pose = robot_pose
        self.next_robot_pose = robot_pose
        self.next_move_target = next_move_target
        self.distance_from_next_robot_pose_to_next_move_target = 0.0
        self.moving = False
        self.current_move_invalid = False
        self.robot_move_path = [robot_pose]
        self.num_robot_move_points = 1
        self.robot_local_path = []
        self.num_local_move_points = 0
        self.max_path_nodes = max_path_nodes
        
        self.robot_edge: Edge = None
        self.robot_edge_used = False
        self.dist_along_robot_edge = 0.0
        self.time_along_robot_edge = 0.0
        self.robot_edge_for_plotting_used =False
        self.robot_edge_for_plotting: Edge = None
        self.dist_along_robot_edge_for_plotting = 0.0
        self.time_along_robot_edge_for_plotting = 0.0