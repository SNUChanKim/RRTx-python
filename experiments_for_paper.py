import numpy as np
import math
import os

from rrt_x import RRTX
from data_structure import CSpace
import obstacle_functions as of
import rand_functions as rf
from config import get_args

def static_2d_debug():
    change_thresh = 1.0
    exp_name = "static_2d_debug"
    
    total_time = 50000.0
    slice_time = 1.0/10.0
    env_rad = 50.0
    robot_rad = 0.5
    start = np.array([0.0, -40.0])
    goal = np.array([-40.0, 40.0])
    
    obstacle_file = "environments/rand_Static.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = False
    save_video_data = True
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 2.0
    
    of.read_discoverable_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = False
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 5.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def dynamic_2d_debug():
    change_thresh = 1.0
    exp_name = "dynamic_2d_debug"
    
    total_time = 20.0
    slice_time = 1.0/10.0
    obstacle_complexity = 1
    env_rad = 50.0
    robot_rad = 0.5
    start = np.array([0.0, -40.0])
    goal = np.array([-40.0, 40.0])
    
    obstacle_file = "environments/rand_Disc2.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 20.0
    C.space_has_time = False
    C.space_has_theta = False
    
    of.read_discoverable_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_or_from_stack
    C.p_goal = 0.01
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 5.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def static_2d_time_debug():
    change_thresh = 1.0
    exp_name = "static_2d_time_debug"
    
    total_time = 3.0
    slice_time = 1.0/10.0
    
    env_rad = 50.0
    max_time = 35.0
    robot_rad = 5.0
    start = np.array([0.0, -40.0, 0.0])
    goal = np.array([0.0, 40.0, max_time])
    
    obstacle_file = "environments/rand_StaticTime.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 3
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d-1)
    lower_bounds = np.append(lower_bounds, 0.0)
    upper_bounds = env_rad*np.ones(d-1)
    upper_bounds = np.append(upper_bounds, max_time)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 20.0
    
    of.read_time_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_in_time_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = True
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def dynamic_2d_time_debug():
    change_thresh = 1.0
    exp_name = "dynamic_2d_time_debug"
    
    total_time = 5.0
    slice_time = 1.0/10.0
    
    env_rad = 50.0
    max_time = 35.0
    min_time = 0.0
    
    robot_rad = 2.0
    start = np.array([0.0, -40.0, min_time])
    goal = np.array([0.0, 40.0, max_time])
    
    obstacle_file = "environments/rand_DynamicTime.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 3
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d-1)
    lower_bounds = np.append(lower_bounds, min_time)
    upper_bounds = env_rad*np.ones(d-1)
    upper_bounds = np.append(upper_bounds, max_time)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 80.0
    
    of.read_dynamic_time_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_in_time_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = True
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()

def bottleneck_repeats():
    first_trial = 1
    total_trials = 1
    
    change_thresh = 0.5
    exp_name = "bottleneck_repeats"
    
    total_time = 60.0
    slice_time = 1.0/10.0
    
    env_rad = 50.0
    robot_rad = 0.5
    
    start = np.array([0.0, -40.0])
    goal = np.array([0.0, 40.0])
    
    obstacle_file = "environments/BNeck.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = False
    save_video_data = True
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    for trial in range(first_trial, total_trials+1):
     
        C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
        
        C.robot_radius = robot_rad
        C.robot_velocity = 20.0
        
        C.space_has_time = False
        C.space_has_theta = False
        
        of.read_obstacle_from_file(C, obstacle_file, 1)
        
        C.rand_node = rf.rand_node_time_with_obstacle_remove
        C.p_goal = 0.01
        C.wait_time = 10.0
        C.obstacles_to_remove = C.obstacles[0]
        C.time_sample_point = np.array([0.0, 0.0])
        
        C.space_has_time = False
        C.space_has_theta = False
        
        data_file = "experiments/{}/rrtx_{}.txt".format(exp_name, trial)
        
        print("#### running trial {} of rrtx".format(trial))
        
        rrt_x = RRTX(C, total_time, slice_time, 5.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
        rrt_x.run()
    
def static_2d_repeats():
    first_trial = 1
    total_trials = 50
    
    change_thresh = 0.5
    exp_name = "static_2d_repeats"
    
    total_time = 30.0
    slice_time = 1.0/10.0
    obstacle_complexity = 1
    
    env_rad = 50.0
    robot_rad = 0.5
    
    start = np.array([35.0, -40.0])
    goal = np.array([-40.0, 40.0])
    
    obstacle_file = "environments/rand_Static.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = False
    save_video_data = False
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    for trial in range(first_trial, total_trials+1):
     
        C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
        
        C.robot_radius = robot_rad
        C.robot_velocity = 20.0
        
        C.space_has_time = False
        C.space_has_theta = False
        
        of.read_discoverable_obstacles_from_file(C, obstacle_file, 1)
        
        C.rand_node = rf.rand_node_or_from_stack
        C.p_goal = 0.01
        
        C.space_has_time = False
        C.space_has_theta = False
        
        data_file = "experiments/{}/rrtx_{}.txt".format(exp_name, trial)
        
        print("#### running trial {} of rrtx".format(trial))
        
        rrt_x = RRTX(C, total_time, slice_time, 5.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
        rrt_x.run()
        
def static_time_repeats():
    first_trial = 1
    total_trials = 1
    
    change_thresh = 0.0
    exp_name = "static_time_repeats"
    
    total_time = 600.0
    slice_time = 1.0
    obstacle_complexity = 1
    
    env_rad = 50.0
    max_time = 35.0
    min_time = -30.0
    robot_rad = 2.0
    
    start = np.array([40.0, 40.0, min_time, math.pi/3])
    goal = np.array([-40.0, -40.0, max_time, math.pi/3])
    
    obstacle_file = "environments/rand_StaticTime_8.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = False
    save_video_data = False
    
    d = 4
    time_out = math.inf
    save_tree = True
    
    lower_bounds = np.array([-env_rad, -env_rad, min_time, 0.0])
    upper_bounds = np.array([env_rad, env_rad, max_time, 2*math.pi])
    
    for trial in range(first_trial, total_trials+1):
     
        C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
        
        C.robot_radius = robot_rad
        C.robot_velocity = 20.0
        C.dubins_min_velocity = 5.0
        C.dubins_max_velocity = 20.0
        
        C.min_turning_radius = 2.0
        
        C.space_has_time = False
        C.space_has_theta = False
        
        of.read_time_obstacles_from_file(C, obstacle_file, 1)
        
        C.rand_node = rf.rand_node_or_from_stack
        C.p_goal = 0.01
        
        C.space_has_time = True
        C.space_has_theta = True
        
        data_file = "experiments/{}/rrtx_{}.txt".format(exp_name, trial)
        
        print("#### running trial {} of rrtx".format(trial))
        
        rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
        rrt_x.run()


def dynamic_2d_forest():
    change_thresh = 1.0
    exp_name = "dynamic_2d_forest"
    
    total_time = 20.0
    slice_time = 1.0/10.0
    obstacle_complexity = 1
    
    env_rad = 50.0
    robot_rad = 1.2
    
    start = np.array([40.0, 40.0])
    goal = np.array([-40.0, -40.0])
    
    obstacle_file = "environments/rand_DiscForest_.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 20.0
    
    C.space_has_time = False
    C.space_has_theta = False
    
    of.read_discoverable_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = False
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 5.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def dynamic_2d_fort():
    change_thresh = 1.0
    exp_name = "dynamic_2d_fort"
    
    total_time = 20.0
    slice_time = 1.0/10.0
    obstacle_complexity = 1
    
    env_rad = 50.0
    robot_rad = 1.5
    
    start = np.array([0.0, -45.0])
    goal = np.array([0.0, 0.0])
    
    obstacle_file = "environments/rand_Disc_3.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 2
    time_out = math.inf
    save_tree = True
    
    lower_bounds = -env_rad*np.ones(d)
    upper_bounds = env_rad*np.ones(d)
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 20.0
    
    C.space_has_time = False
    C.space_has_theta = False
    
    of.read_discoverable_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = False
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def static_2d_time_grid():
    change_thresh = 1.0
    exp_name = "static_2d_time_grid"
    
    total_time = 5.0
    slice_time = 1.0/10.0
    obstacle_complexity = 1
    
    env_rad = 50.0
    max_time = 35.0
    min_time = 20
    robot_rad = 2.0
    
    start = np.array([40.0, 40.0, min_time])
    goal = np.array([-40.0, -40.0, max_time])
    
    obstacle_file = "environments/rand_StaticTime_6.txt"
    
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    
    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
    
    move_robot = True
    save_video_data = True
    
    d = 3
    time_out = math.inf
    save_tree = True
    
    lower_bounds = np.array([-env_rad, -env_rad, 0.0])
    upper_bounds = np.array([env_rad, env_rad, max_time])
    
    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)
    
    C.robot_radius = robot_rad
    C.robot_velocity = 80.0
    
    of.read_time_obstacles_from_file(C, obstacle_file, 1)
    
    C.rand_node = rf.rand_node_in_time_or_from_stack
    C.p_goal = 0.01
    
    C.space_has_time = True
    C.space_has_theta = False
    
    data_file = "experiments/{}/debug_data.txt".format(exp_name)
    
    rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()
    
def dynamic_2d_time_busy():

    change_thresh = 1.0    

    exp_name = "dynamic_2d_time_busy"     

    total_time = 5.0      
    robot_velocity = 80.0  
    slice_time = 1.0/10.0 

    # random obstacle test:
    env_rad = 50.0         
    max_time = 35.0       
    min_time = 0.0 #-15.0     

    robot_rad = 2.0

    
    start = np.array([-40.0, -40.0, min_time])  
    goal = np.array([40.0, 40.0, max_time])   

    obstacle_file = "environments/rand_DynamicTime_3.txt"
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")

    if not os.path.isdir("experiments/{}".format(exp_name)):
        os.mkdir("experiments/{}".format(exp_name))
  
    move_robot = True
    save_video_data = True

    d = 3                  
    time_out = math.inf         
    save_tree = True        

    lower_bounds = np.array([-env_rad, -env_rad, min_time])
    upper_bounds = np.array([env_rad, env_rad, max_time])

    C = CSpace(d, -1.0, lower_bounds, upper_bounds, start, goal)

    C.robot_radius =  robot_rad

    C.robot_velocity = robot_velocity

    of.read_dynamic_time_obstacles_from_file(C, obstacle_file, 1)

    C.rand_node = rf.rand_node_in_time_or_from_stack # use this function to return random nodes
    C.p_goal = .01

    C.space_has_time = True
    C.space_has_theta = False

    data_file = "experiments/{}/debug_data.txt".format(exp_name)

    rrt_x = RRTX(C, total_time, slice_time, 10.0, 100.0, change_thresh, move_robot, save_video_data, save_tree, data_file, exp_name)
    rrt_x.run()

if __name__ == "__main__":
    args = get_args()
    
    np.random.seed(args.seed)
    
    if args.exp_name == 'static_2d_debug':
        static_2d_debug()
    elif args.exp_name == 'dynamic_2d_debug':
        dynamic_2d_debug()
    elif args.exp_name == 'static_2d_time_debug':
        static_2d_time_debug()
    elif args.exp_name == 'dynamic_2d_time_debug':
        dynamic_2d_time_debug()
    elif args.exp_name == 'bottleneck_repeats':
        bottleneck_repeats()
    elif args.exp_name == 'static_2d_repeats':
        static_2d_repeats()
    elif args.exp_name == 'static_time_repeats':
        static_time_repeats()
    elif args.exp_name == 'dynamic_2d_forest':
        dynamic_2d_forest()
    elif args.exp_name == 'dynamic_2d_fort':
        dynamic_2d_fort()
    elif args.exp_name == 'static_2d_time_grid':
        static_2d_time_grid()
    elif args.exp_name == 'dynamic_2d_time_busy':
        dynamic_2d_time_busy()
    else:
        raise NotImplementedError("This experiment is not implemented.")