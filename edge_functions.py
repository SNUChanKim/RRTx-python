import numpy as np
import math
from copy import deepcopy
import distance_functions as df
import collision_checking_functions as ccf
from data_structure import CSpace, Obstacle
from edge import Edge

def dist(S: CSpace, x, y):
    if S.space_has_theta:
        if hasattr(x, 'position'):
            return df.r3s_dist(x.position, y.position)
        return df.r3s_dist(x, y)
    else: 
        if hasattr(x, 'position'):
            return df.euclidian_dist(x.position, y.position)
        return df.euclidian_dist(x, y)

def kd_dist(x, y):
    if hasattr(x, 'position'):
        return df.euclidian_dist(x.position, y.position)
    return df.euclidian_dist(x, y)

def w_dist(x, y):
    if hasattr(x, 'position'):
        return df.euclidian_dist(x.position[:2], y.position[:2])
    return df.euclidian_dist(x[:2], y[:2])

def saturate(S: CSpace, new_point, closest_point, delta):
    if S.space_has_theta:
        this_dist = dist(S, new_point, closest_point)
        if this_dist > delta:
            new_point[:3] = closest_point[:3] + (new_point[:3] - closest_point[:3])*delta/this_dist
            
            if np.abs(new_point[3] - closest_point[3]) < math.pi:
                new_point[3] = closest_point[3] + (new_point[3] - closest_point[3])*delta/this_dist
            else:
                if new_point[3] < math.pi:
                    new_point[3] = new_point[3] + 2*math.pi
                else:
                    new_point[3] = new_point[3] - 2*math.pi
                    
                new_point[3] = closest_point[3] + (new_point[3] - closest_point[3])*delta/this_dist
                
                new_point[3] = np.maximum(np.minimum(new_point[3], 2*math.pi), 0)
        return new_point
    else:
        this_dist = dist(S, new_point, closest_point)
        if this_dist > delta:
            new_point = closest_point + (new_point - closest_point)*delta/this_dist
        return new_point

def new_edge(start_node, end_node):
    edge = Edge()
    edge.start_node = start_node
    edge.end_node = end_node
    return edge

def valid_move(S: CSpace, edge: Edge):
    if S.space_has_theta:
        if S.space_has_time:
            return ((edge.start_node.position[2] > edge.end_node.position[2]) and (S.dubins_min_velocity <= edge.velocity <= S.dubins_max_velocity))
        return True
    else:
        if S.space_has_time:
            return edge.w_dist <= (edge.start_node.position[2] - edge.end_node.position[2])*S.robot_velocity
        return True

def pose_at_dist_along_edge(S: CSpace, edge: Edge, dist_along_edge):
    if S.space_has_theta:
        dist_remaining = dist_along_edge
    
        if len(edge.trajectory) < 2 or edge.dist <= dist_along_edge:
            return edge.end_node.position
        
        i = 1
        this_dist = math.inf
        time_in_path = (edge.trajectory[0].size >= 3)
        while i < len(edge.trajectory):
            if time_in_path:
                this_dist = df.dubins_dist_along_time_path(edge.trajectory[i-1], edge.trajectory[i])
            else:
                this_dist = df.dubins_dist_along_path(edge.trajectory[i-1], edge.trajectory[i])

            if dist_remaining - this_dist <= 0:
                break
        
            dist_remaining -= this_dist
            i += 1
            
        if dist_remaining > this_dist:
            dist_remaining = this_dist
        
        ratio = dist_remaining/this_dist
        if i == len(edge.trajectory):
            i = i - 1
        ret = edge.trajectory[i-1] + ratio*(edge.trajectory[i] - edge.trajectory[i-1])
        ret_time_ratio = dist_along_edge/edge.dist
        ret_time = edge.start_node.position[2] + ret_time_ratio*(edge.end_node.position[2] - edge.start_node.position[2])
        ret_theta = math.atan2(edge.trajectory[i][1] - edge.trajectory[i-1][1], edge.trajectory[i][0] - edge.trajectory[i-1][0])
        
        return [ret[0], ret[1], ret_time, ret_theta]
    else:
        if edge.dist == 0.0:
            return edge.end_node.position
        ratio_along_edge = dist_along_edge/edge.dist
        ret = edge.start_node.position + ratio_along_edge * (edge.end_node.position - edge.start_node.position)
        return ret

def pose_at_time_along_edge(S: CSpace, edge: Edge, time_along_edge):
    if S.space_has_theta:
        if len(edge.trajectory) < 2 or (edge.start_node.position[2] - edge.end_node.position[2]) <= time_along_edge:
            return edge.end_node.position
        
        i = 1
        while edge.trajectory[i][2] > edge.start_node.position[2] - time_along_edge:
            i+=1
            
        ratio = (edge.trajectory[i-1][2] - (edge.start_node.position[2] - time_along_edge))/(edge.trajectory[i-1][2] - edge.trajectory[i][2])
        
        ret = edge.trajectory[i-1] + ratio*(edge.trajectory[i] - edge.trajectory[i-1])
        ret_time = edge.start_node.position[2] - time_along_edge
        ret_theta = math.atan2(edge.trajectory[i][1] - edge.trajectory[i-1][1] , edge.trajectory[i][0] - edge.trajectory[i-1][0])
        
        return [ret[0], ret[1], ret_time, ret_theta]
    else:
        ratio_along_edge = time_along_edge/(edge.start_node.position[2] - edge.end_node.position[2] + 1e-10)
        ret = edge.start_node.position + ratio_along_edge*(edge.end_node.position - edge.start_node.position)
        return ret

def save_end_of_trajectory(S: CSpace, writer, edge: Edge, dist_from_front):
    if S.space_has_theta:
        dist_remaining = dist_from_front
        if len(edge.trajectory) < 2 or edge.dist <= dist_from_front:
            writer.writerow(edge.end_node.position[:2])
            return
        
        i = 1
        this_dist = math.inf
        time_in_path = (edge.trajectory[0].size >= 3)
        while i<= len(edge.trajectory):
            if time_in_path:
                this_dist = df.dubins_dist_along_time_path(edge.trajectory[i-1], edge.trajectory[i])
            else:
                this_dist = df.dubins_dist_along_path(edge.trajectory[i-1], edge.trajectory[i])
            
            if dist_remaining - this_dist <= 0:
                break
            
            dist_remaining -= this_dist
            i += 1
        
        if dist_remaining > this_dist:
            dist_remaining = this_dist
        
        ratio = dist_remaining/this_dist
        ret = edge.trajectory[i-1] + ratio*(edge.trajectory[i] - edge.trajectory[i-1])
        
        writer.writerow(ret)
        
        for j in range(i, len(edge.trajectory)):
            position_at_j = edge.trajectory[j]
            
            writer.writerow(position_at_j)
    else:
        ratio_along_edge = dist_from_front/edge.dist
        ret = edge.start_node.position + ratio_along_edge*(edge.end_node.position - edge.start_node.position)
        writer.writerow(list(ret))
        writer.writerow(list(edge.end_node.position))
    
def save_end_of_trajectory_time(S: CSpace, writer, edge: Edge, time_from_front):
    if S.space_has_theta:
        if len(edge.trajectory) < 2 or (edge.start_node.position[2] - edge.end_node.position[2]) <= time_from_front:
            writer.writerow(edge.end_node.position[0][:3])
            return 
        
        i = 1
        while edge.trajectory[i][2] > edge.start_node.position[2] - time_from_front:
            i+=1
            
        ratio = (edge.trajectory[i-1][2] - (edge.start_node.position[2] - time_from_front))/(edge.trajectory[i-1][2] - edge.trajectory[i][2])
        
        ret = edge.trajectory[i-1] + ratio*(edge.trajectory[i] - edge.trajectory[i-1])
        ret_time = edge.start_node.position[2] - time_from_front
        
        writer.writerow([ret[0], ret[1], ret_time])
        
        for j in range(i, len(edge.trajectory)):
            writer.writerow(edge.trajectory[j][:3])
    else:
        ratio_along_edge = time_from_front/(edge.start_node.position[2] - edge.end_node.position[2] + 1e-10)
        ret = edge.start_node.position + ratio_along_edge*(edge.end_node.position - edge.start_node.position)
        writer.writerow(list(ret))
        writer.writerow(list(edge.end_node.position))
    
def calculate_trajectory(S: CSpace, edge: Edge):
    if S.space_has_theta:
        r_min = S.min_turning_radius
    
        initial_location = edge.start_node.position[:2]
        initial_theta = edge.start_node.position[3]
        goal_location = edge.end_node.position[:2]
        goal_theta = edge.end_node.position[3]
        
        irc_center = initial_location + r_min*np.array([math.cos(initial_theta - math.pi/2.0), math.sin(initial_theta - math.pi/2.0)])
        ilc_center = initial_location + r_min*np.array([math.cos(initial_theta + math.pi/2.0), math.sin(initial_theta + math.pi/2.0)])
        grc_center = goal_location + r_min*np.array([math.cos(goal_theta - math.pi/2.0), math.sin(goal_theta - math.pi/2.0)])
        glc_center = goal_location + r_min*np.array([math.cos(goal_theta + math.pi/2.0), math.sin(goal_theta + math.pi/2.0)])
        
        best_dist = math.inf
        best_traj_type = "xxx"
        
        D = np.linalg.norm(glc_center - irc_center)
        v = (glc_center - irc_center)/D
        R = -2.0*r_min/D
        
        if abs(R) > 1.0:
            rsl_length = math.inf
        else:
            sq = np.sqrt(1-R**2)
            a = r_min*(R*v[0] + v[1]*sq)
            b = r_min*(R*v[1] - v[0]*sq)
            rsl_tangent_x = [irc_center[0] - a, glc_center[0] + a]
            rsl_tangent_y = [irc_center[1] - b, glc_center[1] + b]
            
            first_dist = df.right_turn_dist(initial_location, np.array([rsl_tangent_x[0], rsl_tangent_y[0]]), irc_center, r_min)
            second_dist = np.linalg.norm(np.array([rsl_tangent_x[1] - rsl_tangent_x[0], rsl_tangent_y[1] - rsl_tangent_y[0]]))
            third_dist = df.left_turn_dist(np.array([rsl_tangent_x[1], rsl_tangent_y[1]]), goal_location, glc_center, r_min)
            
            rsl_length = first_dist + second_dist + third_dist
            
            if best_dist > rsl_length:
                best_dist = rsl_length
                best_traj_type = "rsl"
        
        D = np.linalg.norm(grc_center - irc_center)
        v = (grc_center - irc_center)/D
        rsr_tangent_x = -r_min*v[1] + np.array([irc_center[0], grc_center[0]])
        rsr_tangent_y = r_min*v[0] + np.array([irc_center[1], grc_center[1]])
        
        first_dist = df.right_turn_dist(initial_location, np.array([rsr_tangent_x[0], rsr_tangent_y[0]]), irc_center, r_min)
        second_dist = np.linalg.norm(np.array([rsr_tangent_x[1] - rsr_tangent_x[0], rsr_tangent_y[1] - rsr_tangent_y[0]]))
        third_dist = df.left_turn_dist(np.array([rsr_tangent_x[1], rsr_tangent_y[1]]), goal_location, glc_center, r_min)
        rsr_length = first_dist + second_dist + third_dist
        
        if best_dist > rsr_length:
            best_dist = rsr_length
            best_traj_type = "rsr"
        
        rlr_rl_tangent = np.array([math.nan, math.nan])
        rlr_lr_tangent = np.array([math.nan, math.nan])
        
        if D < 4.0*r_min:
            theta = -math.acos(D/(4*r_min)) + math.atan2(v[1], v[0])
            rlr_l_circle_center = irc_center + 2*r_min*np.array([np.cos(theta), np.sin(theta)])
            rlr_rl_tangent = (rlr_l_circle_center + irc_center)/2.0
            rlr_lr_tangent = (rlr_l_circle_center + grc_center)/2.0
            
            first_dist = df.right_turn_dist(initial_location, rlr_rl_tangent, irc_center, r_min)
            second_dist = df.left_turn_dist(rlr_rl_tangent, rlr_lr_tangent, rlr_l_circle_center, r_min)
            third_dist = df.right_turn_dist(rlr_lr_tangent, goal_location, grc_center, r_min)
            rlr_length = first_dist + second_dist + third_dist
            
            if best_dist > rlr_length:
                best_dist = rlr_length
                best_traj_type = "rlr"
        else:
            rlr_length = math.inf
            
        D = np.linalg.norm(grc_center - ilc_center)
        v = (grc_center - ilc_center)/D
        R = -2.0*r_min/D
        if abs(R) > 1:
            lsr_length = math.inf
        else:
            sq = np.sqrt(1-R**2)
            a = r_min*(R*v[0] + v[1]*sq)
            b = r_min*(R*v[1] - v[0]*sq)
            lsr_tangent_x = np.array([ilc_center[0] + a, grc_center[0] - a])
            lsr_tangent_y = np.array([ilc_center[1] + b, grc_center[1] - b])
            
            first_dist = df.left_turn_dist(initial_location, np.array([lsr_tangent_x[0], lsr_tangent_y[0]]), ilc_center, r_min)
            second_dist = np.linalg.norm(np.array([lsr_tangent_x[1] - lsr_tangent_x[0], lsr_tangent_y[1] - lsr_tangent_y[0]]))
            third_dist = df.left_turn_dist(np.array([lsr_tangent_x[1], lsr_tangent_y[1]]), goal_location, glc_center, r_min)
            
            lsr_length = first_dist + second_dist + third_dist
            
            if best_dist > lsr_length:
                best_dist = lsr_length
                best_traj_type = "lsr"
        
        
        D = np.linalg.norm(glc_center - ilc_center)
        v = (glc_center - ilc_center)/D
        lsl_tangent_x = r_min*v[1] + np.array([ilc_center[0], glc_center[0]])
        lsl_tangent_y = -r_min*v[0] + np.array([ilc_center[1], glc_center[1]])
        
        first_dist = df.right_turn_dist(initial_location, np.array([lsl_tangent_x[0], lsl_tangent_y[0]]), ilc_center, r_min)
        second_dist = np.linalg.norm(np.array([lsl_tangent_x[1] - lsl_tangent_x[0], lsl_tangent_y[1] - lsl_tangent_y[0]]))
        third_dist = df.left_turn_dist(np.array([lsl_tangent_x[1], lsl_tangent_y[1]]), goal_location, glc_center, r_min)
        lsl_length = first_dist + second_dist + third_dist
        
        if best_dist > lsl_length:
            best_dist = lsl_length
            best_traj_type = "lsl"
        
        lrl_rl_tangent = np.array([math.nan, math.nan])
        lrl_lr_tangent = np.array([math.nan, math.nan])
        if D < 4.0*r_min:
            theta = math.acos(D/(4*r_min)) + math.atan2(v[1], v[0])
            lrl_r_circle_center = ilc_center + 2.0*r_min*np.array([math.cos(theta), math.sin(theta)])
            lrl_lr_tangent = (lrl_r_circle_center + ilc_center)/2.0
            lrl_rl_tangent = (lrl_r_circle_center + glc_center)/2.0
            
            first_dist = df.left_turn_dist(initial_location, lrl_lr_tangent, ilc_center, r_min)
            second_dist = df.right_turn_dist(lrl_lr_tangent, lrl_rl_tangent, lrl_r_circle_center, r_min)
            third_dist = df.left_turn_dist(lrl_rl_tangent, goal_location, glc_center, r_min)
            lrl_length = first_dist + second_dist + third_dist
            
            if best_dist > lrl_length:
                best_dist = lrl_length
                best_traj_type = "lrl"
                
        else:
            lrl_length = math.inf
            
        delta_phi = 0.1
        
        if best_traj_type[0] == 'r':
            if best_traj_type == "rsl":
                p = np.array([rsl_tangent_x[0], rsl_tangent_y[0]])
            elif best_traj_type == "rsr":
                p = np.array([rsr_tangent_x[0], rsr_tangent_y[0]])
            else:
                p = rlr_rl_tangent
            phi_start = math.atan2(initial_location[1] - irc_center[1], initial_location[0] - irc_center[0])
            phi_end = math.atan2(p[1] - irc_center[1], p[0] - irc_center[0])
            
            if phi_end > phi_start:
                phi_end = phi_end - 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, -delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)
            
            first_path_x = irc_center[0] + r_min*np.cos(phis)
            first_path_y = irc_center[1] + r_min*np.sin(phis)
            
        elif best_traj_type[0] == 'l':
            if best_traj_type == "lsl":
                p = np.array([lsl_tangent_x[0], lsl_tangent_y[0]])
            elif best_traj_type == "lsr":
                p = np.array([lsr_tangent_x[0], lsr_tangent_y[0]])
            else:
                p = lrl_lr_tangent
            
            phi_start = math.atan2(initial_location[1] - ilc_center[1], initial_location[0] - ilc_center[0])
            phi_end = math.atan2(p[1] - ilc_center[1], p[0] - ilc_center[0])
            
            if phi_end < phi_start:
                phi_end = phi_end + 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)

            first_path_x = ilc_center[0] + r_min*np.cos(phis)
            first_path_y = ilc_center[1] + r_min*np.sin(phis)
        
        if best_traj_type[1] == 's':
            
            if best_traj_type == "lsr":
                p1 = np.array([lsr_tangent_x[0], lsr_tangent_y[0]])
                p2 = np.array([lsr_tangent_x[1], lsr_tangent_y[1]])
            elif best_traj_type == "lsl":
                p1 = np.array([lsl_tangent_x[0], lsl_tangent_y[0]])
                p2 = np.array([lsl_tangent_x[1], lsl_tangent_y[1]])
            elif best_traj_type == "rsr":
                p1 = np.array([rsr_tangent_x[0], rsr_tangent_y[0]])
                p2 = np.array([rsr_tangent_x[1], rsr_tangent_y[1]])
            else:
                p1 = np.array([rsl_tangent_x[0], rsl_tangent_y[0]])
                p2 = np.array([rsl_tangent_x[1], rsl_tangent_y[1]])
            
            second_path_x = np.array([p1[0], p2[0]])
            second_path_y = np.array([p1[1], p2[1]])
        
        elif best_traj_type[1] == 'r':
            
            phi_start = math.atan2(lrl_lr_tangent[1] - lrl_r_circle_center[1], lrl_lr_tangent[0] - lrl_r_circle_center[0])
            phi_end = math.atan2(lrl_rl_tangent[1] - lrl_r_circle_center[1], lrl_rl_tangent[0] - lrl_r_circle_center[0])
            
            if phi_end > phi_start:
                phi_end = phi_end - 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, -delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)
            
            second_path_x = lrl_r_circle_center[0] + r_min*np.cos(phis)
            second_path_y = lrl_r_circle_center[1] + r_min*np.sin(phis)
        
        elif best_traj_type[1] == "l":
            
            phi_start = math.atan2(rlr_rl_tangent[1] - rlr_l_circle_center[1], rlr_rl_tangent[0] - rlr_l_circle_center[0])
            phi_end = math.atan2(rlr_lr_tangent[1] - rlr_l_circle_center[1], rlr_lr_tangent[0] - rlr_l_circle_center[0])
            
            if phi_end < phi_start:
                phi_end = phi_end + 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)
                
            second_path_x = rlr_l_circle_center[0] + r_min*np.cos(phis)
            second_path_y = rlr_l_circle_center[1] + r_min*np.sin(phis)
            
        if best_traj_type[2] == 'r':
            
            if best_traj_type == "rsr":
                p = np.array([rsr_tangent_x[1], rsr_tangent_y[1]])
            elif best_traj_type == "lsr":
                p = np.array([lsr_tangent_x[1], lsr_tangent_y[1]])
            else:
                p = rlr_lr_tangent
            
            phi_start = math.atan2(p[1] - grc_center[1], p[0] - grc_center[0])
            phi_end = math.atan2(goal_location[1] - grc_center[1], goal_location[0] - grc_center[0])
            
            if phi_end > phi_start:
                phi_end = phi_end - 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, -delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)
            
            third_path_x = grc_center[0] + r_min*np.cos(phis)
            third_path_y = grc_center[1] + r_min*np.sin(phis)
            
        elif best_traj_type[2] == 'l':
            if best_traj_type == "lsl":
                p = np.array([lsl_tangent_x[1], lsl_tangent_y[1]])
            elif best_traj_type == "rsl":
                p = np.array([rsl_tangent_x[1], rsl_tangent_y[1]])
            else:
                p = lrl_rl_tangent
            
            phi_start = math.atan2(p[1] - glc_center[1], p[0] - glc_center[0])
            phi_end = math.atan2(goal_location[1] - glc_center[1], goal_location[0] - glc_center[0])
            
            if phi_end < phi_start:
                phi_end = phi_end + 2.0*math.pi
            
            if phi_end == phi_start:
                phis = phi_start
            else:
                phis = list(np.arange(phi_start, phi_end, delta_phi))
                phis.append(phi_end)
                phis = np.array(phis)
            
            third_path_x = glc_center[0] + r_min*np.cos(phis)
            third_path_y = glc_center[1] + r_min*np.sin(phis)
        
        edge.dubins_type = best_traj_type
        edge.w_dist = best_dist
        
        if np.isinf(edge.w_dist):
            edge.dist = math.inf
        elif S.space_has_time:
            edge.dist = np.sqrt(best_dist**2 + (edge.start_node.position[2] - edge.end_node.position[2])**2)
            
            edge.velocity = edge.w_dist/(edge.start_node.position[2] - edge.end_node.position[2])
            
            trajectory_x = np.concatenate((first_path_x, second_path_x, third_path_x), axis=0)
            trajectory_y = np.concatenate((first_path_y, second_path_y, third_path_y), axis=0)
            trajectory_t = np.zeros_like(trajectory_x)
            edge.trajectory = list(np.transpose(np.array([trajectory_x, trajectory_y, trajectory_t])))
            edge.trajectory[0][2] = edge.start_node.position[2]
            cumulative_dist = 0.0
            
            for i in range(1, len(edge.trajectory)-1):
                cumulative_dist += w_dist(edge.trajectory[i-1][:2], edge.trajectory[i][:2])
                edge.trajectory[i][2] = edge.start_node.position[2] - cumulative_dist/edge.velocity
            
            edge.trajectory[-1][:3] = edge.end_node.position[:3]
        else:
            edge.dist = best_dist
            trajectory_x = np.concatenate((first_path_x, second_path_x, third_path_x), axis=0)
            trajectory_y = np.concatenate((first_path_y, second_path_y, third_path_y), axis=0)
            edge.trajectory = list(np.transpose(np.array([trajectory_x, trajectory_y])))
        
        edge.dist_original = edge.dist
    else:
        edge.dist = dist(S, edge.start_node.position, edge.end_node.position)
        edge.dist_original = edge.dist
        edge.w_dist = w_dist(edge.start_node.position, edge.end_node.position)
    
def calculate_hover_trajectory(S: CSpace, edge: Edge):
    if S.space_has_theta:
        edge.dubins_type = "xxx"
        edge.w_dist = 0.0
        edge.dist = 0.0
        
        if S.space_has_time:
            edge.velocity = S.dubins_min_velocity
            edge.trajectory = [edge.start_node.position[:3], edge.end_node.position[:3]]
        else:
            edge.trajectory = [edge.start_node.position[:2], edge.end_node.position[:2]]
    else:
        calculate_trajectory(S, edge)
    
def save_edge_trajectory(S: CSpace, writer, edge: Edge):
    if S.space_has_theta:
        for i in range(len(edge.trajectory)):
            writer.writerow(list(edge.trajectory[i]))
    else:
        writer.writerow(list(edge.start_node.position))
        writer.writerow(list(edge.end_node.position))
    
def explicit_edge_check(S: CSpace, edge: Edge, obstacle: Obstacle, margin=0):
    if S.space_has_theta:
        if not ccf.explicit_edge_check_2d(obstacle, edge.start_node.position, edge.end_node.position, S.robot_radius + 2*S.min_turning_radius):
            return False
        
        for i in range(1, len(edge.trajectory)):
            if ccf.explicit_edge_check_2d(obstacle, edge.trajectory[i-1], edge.trajectory[i], S.robot_radius):
                return True
            
        return False
    else:
        return ccf.explicit_edge_check_2d(obstacle, edge.start_node.position, edge.end_node.position, S.robot_radius, margin)
    
    