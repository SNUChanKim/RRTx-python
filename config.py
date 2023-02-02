import argparse

def get_args():
    parser = argparse.ArgumentParser(description='rrt_x')
    parser.add_argument(
        '--exp-name',
        default='dynamic_2d_time_busy',
        help='name of experiment  (default: dynamic_2d_time_busy)')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0,
        help='random seed (default: 0)')
    
    parser.add_argument(
        '--dubins', 
        action='store_true',
        help='whether to use dubins path')
    
    parser.add_argument(
        '--file-ctr', 
        type=int, 
        default=0,
        help='filt_ctr in make_video.py (default: 0)')
    
    parser.add_argument(
        '--max-file-ctr', 
        type=int,
        help='filt_ctr in make_video.py')
    
    parser.add_argument(
        '--start-move-at-ctr', 
        type=int,
        help='start_move_at_ctr in make_video.py')
    
    parser.add_argument(
        '--fps', 
        type=int, 
        default=10,
        help='fps of generated video (default: 10)')
    
    parser.add_argument(
        '--height', 
        type=int, 
        default=480,
        help='height of the video (default: 480)')
    
    parser.add_argument(
        '--width', 
        type=int, 
        default=480,
        help='width of the video (default: 480)')
    
    args = parser.parse_args()
    return args
