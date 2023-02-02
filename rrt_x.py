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
from datetime import datetime
import gc

import collision_checking_functions as ccf
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
# from dubins_edge import DubinsEdge as Edge
from edge import Edge
from data_structure import RRTNode, RRTNodeNeighborIterator, rrtXQueue, RobotData, Obstacle, CSpace
from heap import BinaryHeap
from kd_tree_general import KDTree
import rrt_functions as rrtf

class RRTX:
    def __init__(self, 
                 C,
                 total_time=50000.0,
                 slice_time=0.1,
                 delta=5.0,
                 ball_constant=100.0,
                 change_thres=1.0,
                 move_robot_flag=False,
                 save_video_data = True,
                 save_tree = True,
                 data_file=None,
                 exp_name=None,
                 robot_sensor_range=20.0
                 ):
        
        self.S: CSpace = C
        self.total_planning_time = total_time
        self.slice_time = slice_time
        self.delta = delta
        self.ball_constant = ball_constant
        self.change_thres = change_thres
        self.move_robot_flag = move_robot_flag
        self.save_video_data = save_video_data
        self.save_tree = save_tree
        self.data_file = data_file
        self.exp_name = exp_name
        self.robot_sensor_range = robot_sensor_range
        # self.closed_edges = []
        
        if self.S.space_has_theta:
            self.KD = KDTree(self.S.d, ef.kd_dist, wraps=np.array([3]), wrap_points=np.array([2*math.pi]))
        else:
            self.KD = KDTree(self.S.d, ef.kd_dist)
            
        self.Q = rrtXQueue()
        self.Q.Q = BinaryHeap(key=rrtf.key_q, 
                              less_than=rrtf.less_q, 
                              greater_than=rrtf.greater_q, 
                              mark=rrtf.mark_q, 
                              unmark=rrtf.unmark_q, 
                              marked=rrtf.marked_q,
                              set_index=rrtf.set_index_q, 
                              unset_index=rrtf.unset_index_q, 
                              get_index=rrtf.get_index_q)
        self.Q.OS = []
        self.Q.S = self.S
        self.Q.change_thresh = change_thres
        
        self.S.sample_stack = []
        self.S.delta = delta
        self.robot_rads = self.S.robot_radius
    
    def check_collision(self, S: CSpace, R: RobotData):
        
        unsafe, _ = ccf.explicit_point_check(S, R.robot_pose)
        if unsafe:
            for ob in S.obstacles:
                if ccf.explicit_edge_check(S, R.robot_edge, ob):
                    print("***********************************************************************")
                    print("ob.polygon: ", ob.polygon)
                    print("ob.position: ", ob.position, ", ob.radius: ", ob.radius)
                    print("R.robot_edge: ", R.robot_edge.start_node.position, R.robot_edge.end_node.position)
                    print("R.robot_pose: ", R.robot_pose)
                    print("R.robot_edge_used: ", R.robot_edge_used)
                    print("Collision occurs!!!", ccf.explicit_edge_check(S, R.robot_edge, ob))
                    print("***********************************************************************")
        
    def find_new_target(self, S: CSpace, KD: KDTree, R: RobotData, hyper_ball_rad):
        
        R.robot_edge_used = False
        R.dist_along_robot_edge = 0.0
        R.time_along_robot_edge = 0.0
        R.robot_edge_for_plotting_used = False
        R.dist_along_robot_edge_for_plotting = 0.0
        R.time_along_robot_edge_for_plotting = 0.0
        
        print("move target has become invalid")
        
        search_ball_rad = np.maximum(hyper_ball_rad, ef.dist(S, R.robot_pose, R.next_move_target.position))
        max_search_ball_rad = ef.dist(S, S.lower_bounds, S.upper_bounds)
        search_ball_rad = np.minimum(search_ball_rad, max_search_ball_rad)
        
        L = kd.kd_find_within_range(KD, search_ball_rad, R.robot_pose)
        
        dummy_robot_node = RRTNode(R.robot_pose)
        edge_to_best_neighbor = Edge()
        
        while True:
            
            best_dist_to_neighbor = math.inf
            best_dist_to_goal = math.inf
            best_neighbor = None
            for neighbor_node, key in L:
                
                this_edge = ef.new_edge(dummy_robot_node, neighbor_node)
                ef.calculate_trajectory(S, this_edge)
                if ef.valid_move(S, this_edge) and not ccf.explicit_edge_check(S, this_edge):
                    
                    dist_to_goal = neighbor_node.rrt_lmc + this_edge.dist
                    
                    if dist_to_goal < best_dist_to_goal and ef.valid_move(S, this_edge):
                        best_dist_to_goal = dist_to_goal
                        best_dist_to_neighbor = this_edge.dist
                        best_neighbor = neighbor_node
                        edge_to_best_neighbor = this_edge
            
            if not np.isinf(best_dist_to_goal):
                R.next_move_target = best_neighbor
                R.distance_from_next_robot_pose_to_next_move_target = best_dist_to_neighbor
                R.current_move_invalid = False
                print("Found a valid move target")
                R.robot_edge = edge_to_best_neighbor
                R.robot_edge_for_plotting = edge_to_best_neighbor
                R.robot_edge_used = True
                R.robot_edge_for_plotting_used = True
                
                if S.space_has_time:
                    R.time_along_robot_edge = 0.0
                    R.time_along_robot_edge_for_plotting = 0.0
                else:
                    R.dist_along_robot_edge = 0.0
                    R.dist_along_robot_edge_for_plotting = 0.0
                
                S.move_goal.is_move_goal = False
                S.move_goal = R.next_move_target
                S.move_goal.is_move_goal = True
                
                break
            
            search_ball_rad *= 2
            if search_ball_rad > max_search_ball_rad:
                print("Unable to find a valid move target")
                break
                
            kd.kd_find_more_within_range(KD, search_ball_rad, R.robot_pose, L)
        kd.empty_range_list(L)
        
    def move_robot(self, S: CSpace, Q: rrtXQueue, KD: KDTree, slice_time, root: RRTNode, hyper_ball_rad, R: RobotData):
                
        if R.moving:
            R.robot_pose = R.next_robot_pose
            for i in range(len(R.robot_local_path)):
                R.robot_move_path.append(R.robot_local_path[i])
            R.robot_local_path.clear()
            
            print("Robot Pose: ", np.around(R.robot_pose[:2], 2), " Goal Pose: ", np.around(S.start[:2], 2))
            
            if S.space_has_time:
                R.time_along_robot_edge_for_plotting = R.time_along_robot_edge
            else:
                R.dist_along_robot_edge_for_plotting = R.dist_along_robot_edge
            
            R.robot_edge_for_plotting = R.robot_edge
            R.robot_edge_for_plotting_used = True
        else:
            R.moving = True
            
            if not S.move_goal.rrt_parent_used:
                R.current_move_invalid = True
                print("S.move_goal does not have parent!")
            else:
                R.robot_edge = S.move_goal.rrt_parent_edge
                R.robot_edge_for_plotting = R.robot_edge
                R.robot_edge_used = True
                R.robot_edge_for_plotting_used = True
                
                if S.space_has_time:
                    R.time_along_robot_edge = 0.0
                    R.time_along_robot_edge_for_plotting = 0.0
                else:
                    R.dist_along_robot_edge = 0.0
                    R.dist_along_robot_edge_for_plotting = 0.0
                
        if R.current_move_invalid:
            self.find_new_target(S, KD, R, hyper_ball_rad)
        else:
            S.move_goal.is_move_goal = False
            S.move_goal = R.next_move_target
            S.move_goal.is_move_goal = True
        
        if not S.space_has_time:
            next_node: RRTNode = R.next_move_target
            next_dist = R.robot_edge.dist - R.dist_along_robot_edge
            dist_remaining = S.robot_velocity*slice_time
            R.robot_local_path.append(R.robot_pose)
            
            while (next_dist <= dist_remaining and next_node != root and
                   next_node.rrt_parent_used and next_node != next_node.rrt_parent_edge.end_node):
                
                R.robot_local_path.append(next_node.position)
                
                dist_remaining -= next_dist
                R.dist_along_robot_edge = 0.0
                R.robot_edge = next_node.rrt_parent_edge
                R.robot_edge_used = True
                
                next_dist = R.robot_edge.dist
                next_node = R.robot_edge.end_node
            
            if next_dist > dist_remaining:
                R.dist_along_robot_edge += dist_remaining
                R.next_robot_pose = ef.pose_at_dist_along_edge(S, R.robot_edge, R.dist_along_robot_edge)
            else:
                R.next_robot_pose = next_node.position
                R.dist_along_robot_edge = R.robot_edge.dist
            
            R.next_move_target = R.robot_edge.end_node
            
            R.robot_local_path.append(R.next_robot_pose)
            
        else:
            next_node: RRTNode = R.next_move_target
            
            R.robot_local_path.append(R.robot_pose)
            
            target_time = R.robot_pose[2] - slice_time
            while(target_time < R.robot_edge.end_node.position[2] and
                  next_node != root and next_node.rrt_parent_used and
                  next_node != next_node.rrt_parent_edge.end_node):
                
                R.robot_local_path.append(next_node.position)
                
                R.robot_edge = next_node.rrt_parent_edge
                R.robot_edge_used = True
                
                next_node = next_node.rrt_parent_edge.end_node
            
            if target_time >= next_node.position[2]:
                R.time_along_robot_edge = R.robot_edge.start_node.position[2] - target_time
                R.next_robot_pose = ef.pose_at_time_along_edge(S, R.robot_edge, R.time_along_robot_edge)
            else:
                R.next_robot_pose = next_node.position
                R.time_along_robot_edge = R.robot_edge.start_node.position[2] - R.robot_edge.end_node.position[2] 
            
            R.next_move_target = R.robot_edge.end_node
            
            R.robot_local_path.append(R.next_robot_pose)   
        
        self.check_collision(S, R)
            
    def find_points_in_conflict_with_obstacles(self, S: CSpace, KD: KDTree, ob: Obstacle, root: RRTNode):
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
    
    def add_new_obstacle(self, S: CSpace, KD: KDTree, Q: rrtXQueue, ob: Obstacle, root: RRTNode, file_counter, R: RobotData):
        ob.obstacle_unused = False
        
        L = rrtf.find_points_in_conflict_with_obstacles(S, KD, ob, root)

        while len(L) > 0:
            this_node, key = kd.pop_from_range_list(L)
            
            out_neighbors = this_node.initial_neighbor_list_out + this_node.rrt_neighbors_out
            for neighbor_edge in out_neighbors:
                if neighbor_edge.start_node != this_node:
                    raise ValueError("Edge's start_node is not this node")
                if ccf.explicit_edge_check(S, neighbor_edge, ob):
                    neighbor_edge.dist = math.inf
                
            if this_node.rrt_parent_used and ccf.explicit_edge_check(S, this_node.rrt_parent_edge, ob):
                this_node.rrt_parent_edge.end_node.successor_list.remove(this_node.successor_list_item_in_parent)
                this_node.rrt_parent_edge = ef.new_edge(this_node, this_node)
                this_node.rrt_parent_edge.dist = math.inf
                this_node.rrt_parent_used = False
                
                rrtf.verify_in_os_queue(Q, this_node)
                
        kd.empty_range_list(L)
        
        if R.robot_edge_used and ccf.explicit_edge_check(S, R.robot_edge, ob):
            R.current_move_invalid = True
            
    def remove_obstacle(self, S: CSpace, KD: KDTree, Q: rrtXQueue, ob: Obstacle, root: RRTNode, hyper_ball_rad, time_elapsed, move_goal: RRTNode):
        L = rrtf.find_points_in_conflict_with_obstacles(S, KD, ob, root)
        
        while len(L) > 0:
            this_node, key = kd.pop_from_range_list(L)
            neighbors_were_blocked = False
            
            out_neighbors = this_node.initial_neighbor_list_out + this_node.rrt_neighbors_out
            for neighbor_edge in out_neighbors:
                if neighbor_edge.start_node != this_node:
                    raise ValueError("Edge's start_node is not this node")
                neighbor_node = neighbor_edge.end_node
                
                if np.isinf(neighbor_edge.dist) and ccf.explicit_edge_check(S, neighbor_edge, ob):
                    conflict_with_other_obs = False
                    for ob_other in S.obstacles:
                        if (ob_other != ob and not ob_other.obstacle_unused and 
                            ob_other.start_time <= time_elapsed <= (ob_other.start_time + ob_other.life_span)):
                            if ccf.explicit_edge_check(S, neighbor_edge, ob_other):
                                conflict_with_other_obs = True
                                break
                    
                    if not conflict_with_other_obs:
                        neighbor_edge.dist = neighbor_edge.dist_original
                        neighbors_were_blocked = True
            
            if neighbors_were_blocked:
                rrtf.recalculate_lmc_mine_v_two(Q, this_node, root, hyper_ball_rad)
                if this_node.rrt_tree_cost != this_node.rrt_lmc and rrtf.less_q(this_node, move_goal):
                    rrtf.verify_in_queue(Q, this_node)

        kd.empty_range_list(L)
        ob.obstacle_unused = True
    
    def run(self):
        start_time = datetime.now().timestamp()
        save_elapsed_time = 0.0
        
        root = RRTNode(self.S.start)
        
        explicitly_unsafe, unused = ccf.explicit_node_check(self.S, root)
        if explicitly_unsafe:
            raise ValueError("root is not safe")
        
        root.rrt_tree_cost = 0.0
        root.rrt_lmc = 0.0
        kd.kd_insert(self.KD, root)
        
        goal = RRTNode(self.S.goal)
        goal.rrt_tree_cost = math.inf
        goal.rrt_lmc = math.inf
        
        self.S.goal_node = goal
        self.S.root = root
        self.S.move_goal = goal
        self.S.move_goal.is_move_goal = True
        
        R = RobotData(self.S.goal, goal, 20000)
        
        v_counter = 0
        self.S.file_ctr = v_counter
        slice_counter = 0
        
        if self.S.space_has_time:
            rrtf.add_other_times_to_root(self.S, self.KD, goal, root)
        
        check_ptr = 0
        it_of_check = []
        it_of_check.append(0)

        elapsed_time = []
        elapsed_time.append(0.0)

        nodes_in_graph = []
        nodes_in_graph.append(1)

        cost_of_goal = []
        cost_of_goal.append(math.inf)
    
        robot_slice_time = datetime.now().timestamp()
        self.S.start_time_ns = robot_slice_time
        self.S.elapsed_time = 0.0

        old_rrt_lmc = math.inf

        while(True):
            hyper_ball_rad = self.shrinking_ball_rad
            it_of_check[check_ptr] += 1
            now_time = datetime.now().timestamp()

            slice_end_time = (1+slice_counter)*self.slice_time

            warmup_time_just_ended = False
            if self.S.in_warmup_time and self.S.warmup_time < self.S.elapsed_time:
                warmup_time_just_ended = True
                self.S.in_warmup_time = False
            
            self.S.elapsed_time = (datetime.now().timestamp() - self.S.start_time_ns) - save_elapsed_time
            
            removed_obstacle = False

            for ob in self.S.obstacles:
                if not ob.senseable_obstacle and not ob.obstacle_unused and (ob.start_time + ob.life_span <= self.S.elapsed_time):
                    self.remove_obstacle(self.S, self.KD, ob, root, hyper_ball_rad, self.S.elapsed_time, self.S.move_goal)
                    removed_obstacle = True
                elif ob.senseable_obstacle and ob.obstacle_unused_after_sense and ef.w_dist(R.robot_pose, ob.position) < self.robot_sensor_range + ob.radius:
                    of.random_sample_obs(self.S, self.KD, ob)
                    self.remove_obstacle(self.S, self.KD, self.Q, ob, root, hyper_ball_rad, self.S.elapsed_time, self.S.move_goal)
                    ob.senseable_obstacle = False
                    ob.start_time = math.inf
                    removed_obstacle = True
                elif self.S.space_has_time and ob.next_direction_change_time > R.robot_pose[2] and ob.last_direction_change_time != R.robot_pose[2]:
                    self.remove_obstacle(self.S, self.KD, self.Q, ob, root, hyper_ball_rad, self.S.elapsed_time, self.S.move_goal)
                    ob.obstacle_unused = False
                    removed_obstacle = True
            
            if removed_obstacle:
                rrtf.reduce_inconsistency(self.Q, self.S.move_goal, self.robot_rads, root, hyper_ball_rad)
            
            added_obstacle = False
            for ob in self.S.obstacles:
                if not ob.senseable_obstacle and ob.obstacle_unused and (ob.start_time <= self.S.elapsed_time <= ob.start_time + ob.life_span):
                    self.add_new_obstacle(self.S, self.KD, self.Q, ob, root, v_counter, R)
                    added_obstacle = True
                elif ob.senseable_obstacle and not ob.obstacle_unused_after_sense and ef.w_dist(R.robot_pose, ob.position) < self.robot_sensor_range + ob.radius:
                    self.add_new_obstacle(self.S, self.KD, self.Q, ob, root, v_counter, R)
                    ob.senseable_obstacle = False
                    added_obstacle = True
                elif self.S.space_has_time and ob.next_direction_change_time > R.robot_pose[2] and ob.last_direction_change_time != R.robot_pose[2]:
                    ob.obstacle_unused = False
                    of.change_obstacle_direction(self.S, ob, R.robot_pose[2])
                    self.add_new_obstacle(self.S, self.KD, self.Q, ob, root, v_counter, R)
                    ob.last_direction_change_time = R.robot_pose[2]
                    added_obstacle = True
                elif warmup_time_just_ended and not ob.obstacle_unused:
                    self.add_new_obstacle(self.S, self.KD, self.Q, ob, root, v_counter, R)
                    added_obstacle = True
                
            if added_obstacle:
                rrtf.propogate_descendants(self.Q, R)
                if not rrtf.marked_os(self.S.move_goal):
                    rrtf.verify_in_queue(self.Q, self.S.move_goal)
                rrtf.reduce_inconsistency(self.Q, self.S.move_goal, self.robot_rads, root, hyper_ball_rad)
            
            self.S.elapsed_time = (datetime.now().timestamp() - self.S.start_time_ns) - save_elapsed_time
            if self.S.elapsed_time >= slice_end_time:
                slice_end_time = (1 + slice_counter)*self.slice_time
                
                robot_slice_start = now_time
                
                slice_counter += 1
                trunc_elapsed_time = math.floor(self.S.elapsed_time*1000)/1000
                
                print("Counter: ", slice_counter, " --- ", "Time: ", trunc_elapsed_time, " ------- ", "Cost to Goal: ", np.around(self.S.move_goal.rrt_tree_cost,4), " ", "LMC: ", np.around(self.S.move_goal.rrt_lmc, 4), " ----")
                elapsed_time.append(self.S.elapsed_time)
                                 
                if elapsed_time[check_ptr] > self.total_planning_time + self.slice_time:
                    if self.move_robot_flag:
                        self.move_robot(self.S, self.Q, self.KD, self.slice_time, root, hyper_ball_rad, R)
                    else:
                        print("done (not moving robot)")
                        break
                
                rrtf.reduce_inconsistency(self.Q, self.S.move_goal, self.robot_rads, root, hyper_ball_rad)
                if (self.S.move_goal.rrt_lmc != old_rrt_lmc):
                    old_rrt_lmc = self.S.move_goal.rrt_lmc
                
                if self.save_video_data:
                    before_save_time = datetime.now().timestamp()
                    if not os.path.isdir("temp/"):
                        os.mkdir("temp/")
                    if not os.path.isdir("temp/{}".format(self.exp_name)):
                        os.mkdir("temp/{}".format(self.exp_name))
                    sf.save_rrt_tree(self.KD, "temp/{}/edges_{}.txt".format(self.exp_name, v_counter))
                    sf.save_rrt_nodes(self.KD, "temp/{}/nodes_{}.txt".format(self.exp_name, v_counter))
                    sf.save_rrt_path(self.S, self.S.move_goal, root, R, "temp/{}/path_{}.txt".format(self.exp_name, v_counter))
                    of.save_obstacle_locations(self.S.obstacles, "temp/{}/obstacles_{}.txt".format(self.exp_name, v_counter))
                    sf.save_data(R.robot_move_path, "temp/{}/robot_move_path_{}.txt".format(self.exp_name, v_counter))

                    v_counter += 1
                    self.S.file_ctr = v_counter
                    
                    save_elapsed_time += (datetime.now().timestamp() - before_save_time)
                
                if np.array_equal(R.robot_pose[:2], root.position[:2]):
                    print("Goal Reached!!")
                    break
                
                if check_ptr < len(cost_of_goal):
                    check_ptr += 1
                    it_of_check.append(it_of_check[-1] + 1)
                    nodes_in_graph.append(self.KD.tree_size)
                    cost_of_goal.append(np.minimum(goal.rrt_tree_cost, goal.rrt_lmc))
                else:
                    print("Warning: out of space to save stats")

            new_node: RRTNode = self.S.rand_node(self.S)
            
            if new_node.kd_in_tree:
                continue
            
            closest_node, closest_dist = kd.kd_find_nearest(self.KD, new_node.position)
            
            if closest_dist > self.delta and new_node != self.S.goal_node:
                new_node.position = ef.saturate(self.S, new_node.position, closest_node.position, self.delta)
                
            explicitly_unsafe, ret_cert = ccf.explicit_node_check(self.S, new_node)
            
            if explicitly_unsafe:
                continue
            
            gc.disable()
            
            rrtf.extend(self.S, self.KD, self.Q, new_node, closest_node, self.delta, hyper_ball_rad, self.S.move_goal)
            rrtf.reduce_inconsistency(self.Q, self.S.move_goal, self.robot_rads, root, hyper_ball_rad)
            
            if (self.S.move_goal.rrt_lmc != old_rrt_lmc):
                old_rrt_lmc = self.S.move_goal.rrt_lmc
            
            gc.enable()    
        
        elapsed_time.append((datetime.now().timestamp() - start_time))
            
        stats = [elapsed_time, it_of_check, nodes_in_graph, cost_of_goal]
        sf.save_data(stats, self.data_file)
        move_length = 0
        
        for i in range(len(R.robot_move_path)-1):
            move_length += ef.w_dist(R.robot_move_path[i], R.robot_move_path[i+1])
        
        print("distance traveled by robot: ", move_length)
        hp.clean_heap(self.Q.Q)
        return True
            
    @property
    def shrinking_ball_rad(self):
        return np.minimum(self.S.delta, self.ball_constant*((np.log(1+self.KD.tree_size)/(self.KD.tree_size))**(1/self.S.d)))