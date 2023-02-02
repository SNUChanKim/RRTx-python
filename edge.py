import numpy as np
import math
        
class Edge(object):
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.dist = None
        self.dist_original = None
        self.w_dist = None
        self.list_item_in_start_node = None
        self.list_item_in_end_node = None
        self.dubins_type = None
        self.trajectory = None
        self.velocity = None