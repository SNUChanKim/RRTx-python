import numpy as np
import math
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
from video_utils import colorPlot, plotPackman, plotSpikes, plotVehicle, plotVehicleTheta
from copy import deepcopy
from config import get_args

def make_video(args):
    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.file_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr

    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50
    tickInterval = 10

    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)
    contourGranularity = 2.5 # x, y granularity for cost contour plot

    contourLevels = np.arange(0, 202, 2)
    mycolormap = np.zeros((contourLevels.size, 3))
    mycolormap = cm.jet(contourLevels)
    mycolormap = mycolormap[:,:3]

    sensor_radius = 20/contourGranularity

    videoFill = True

    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeMovie.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)

        EdgeFile = open('{}edges_{}.txt'.format(dir, file_ctr))
        RawEdgeData = EdgeFile.readlines()
        EdgeData = []
        for edge_data in RawEdgeData:
            edge_data = edge_data.split(',')
            edge_data[-1] = edge_data[-1].replace('\n','')
            EdgeData.append(np.array(list(map(float, edge_data))))
        EdgeData = np.array(EdgeData)
        
        raw_x = EdgeData[:, 0]
        raw_y = EdgeData[:, 1]
        
        NodeFile = open('{}nodes_{}.txt'.format(dir, file_ctr))
        RawNodeData = NodeFile.readlines()
        NodeData = []
        for node_data in RawNodeData:
            node_data = node_data.split(',')
            node_data[-1] = node_data[-1].replace('\n','')
            NodeData.append(np.array(list(map(float, node_data))))
        NodeData = np.array(NodeData)
        
        node_x = EdgeData[:, 0]
        node_y = EdgeData[:, 1]
        node_z = EdgeData[:, -1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
        else:
            obs_x = np.array([])
            obs_y = np.array([])
        
        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        move_x = MoveData[:, 0]
        move_y = MoveData[:, 1]
        move_theta = 0
        
        
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = PathData[:, 0]
            path_y = PathData[:, 1]
        else:
            path_x = np.array([])
            path_y = np.array([])
            
        if os.path.exists('{}obsNodes.txt'.format(dir)):
            ObsNodeFile = open('{}obsNodes.txt'.format(dir))
            RawObsNodeData =  ObsNodeFile.readlines()
            ObsNodeData = []
            for obs_node_data in RawObsNodeData:
                obs_node_data = obs_node_data.split(',')
                obs_node_data[-1] = obs_node_data[-1].replace('\n','')
                ObsNodeData.append(np.array(list(map(float, obs_node_data))))
            ObsNodeData = np.array(ObsNodeData)
            OBSNodes_x = ObsNodeData[:, 0]
            OBSNodes_y = ObsNodeData[:, 1]
        else:
            OBSNodes_x = np.array([])
            OBSNodes_y = np.array([])
            
        if os.path.exists('{}obsNodesNeighbors.txt'.format(dir)):
            ObsNodeNFile = open('{}obsNodesNeighbors.txt'.format(dir))
            RawObsNodeNData =  ObsNodeNFile.readlines()
            ObsNodeNData = []
            for obs_node_n_data in RawObsNodeNData:
                obs_node_n_data = obs_node_n_data.split(',')
                obs_node_n_data[-1] = obs_node_n_data[-1].replace('\n','')
                ObsNodeNData.append(np.array(list(map(float, obs_node_n_data))))
            ObsNodeNData = np.array(ObsNodeNData)
            OBSNodesN_x = ObsNodeNData[:, 0]
            OBSNodesN_y = ObsNodeNData[:, 1]
        else:
            OBSNodesN_x = np.array([])
            OBSNodesN_y = np.array([])
        
        
        Xs = np.arange(minXval, maxXval, contourGranularity)
        Ys = np.arange(minYval, maxYval, contourGranularity)
        
        Z = np.zeros((Ys.size, Xs.size))
        Zmin = np.ones((Ys.size, Xs.size))*np.inf
        Counts = np.zeros((Ys.size, Xs.size))
        
        for v in range(len(node_z)):
            j = np.maximum(np.minimum(math.floor((node_x[v] - minXval)/contourGranularity), Z.shape[1] - 1), 0)
            i = np.maximum(np.minimum(math.floor((node_y[v] - minYval)/contourGranularity), Z.shape[0] - 1), 0)
            
            if np.isinf(node_z[v]):
                continue
            
            Z[i,j] = Z[i,j] + node_z[v]
            Counts[i,j] = Counts[i,j] + 1
            
            Zmin[i,j] = np.minimum(Zmin[i,j], node_z[v])
        
        Z[np.where(Z!=0)] = Z[np.where(Z!=0)]/Counts[np.where(Z!=0)]
        
        jjj = 1

        while np.sum(Z.reshape(-1)==0) > 0:
            jjj = jjj + 1
            
            if np.sum(Z.reshape(-1)!=0) < 5 or jjj > 3:
                Z[np.where(Z==0)] = contourLevels[-1]
                Zmin[np.where(Z==0)] = contourLevels[-1]
                break
            
            yZeroInds, xZeroInds = np.where(Z == 0)
            dZ = np.zeros_like(Z)
            dZmin = np.zeros_like(Zmin)
            for k in range(xZeroInds.size):
                xZind = xZeroInds[k]
                yZind = yZeroInds[k]
                
                minxZind = np.maximum(xZind - 1, 0)
                maxxZind = np.minimum(xZind + 1, Z.shape[1] - 1)
                minyZind = np.maximum(yZind - 1, 0)
                maxyZind = np.minimum(yZind + 1, Z.shape[0] - 1)
                
                if videoFill and (Ys.size-1)/2 - 2 < yZind and yZind < (Ys.size-1)/2 + 2:
                    minyZind = yZind
                    
                subZ = Z[minyZind:maxyZind+1, minxZind:maxxZind+1]
                if np.sum(subZ.reshape(-1) != 0) > 1:
                    dZ[yZind, xZind] = np.sum(subZ.reshape(-1))/np.sum(subZ.reshape(-1) != 0)
                    dZmin[yZind, xZind] = np.min(subZ[np.where(subZ != 0)])
            
            Z = Z + dZ
            Zmin[np.where(np.logical_and(np.isinf(Zmin), dZmin != 0))] = dZmin[np.where(np.logical_and(np.isinf(Zmin), dZmin != 0))]
            
        c_node_x = (node_x - minXval)/contourGranularity + 0.5
        c_node_y = (node_y - minYval)/contourGranularity + 0.5
        
        c_path_x = (path_x - minXval)/contourGranularity + 0.5
        c_path_y = (path_y - minYval)/contourGranularity + 0.5
        
        c_move_x = (move_x - minXval)/contourGranularity + 0.5
        c_move_y = (move_y - minYval)/contourGranularity + 0.5
        c_move_theta = move_theta - math.pi/2.0
        
        c_x = (raw_x - minXval)/contourGranularity + 0.5
        c_y = (raw_y - minYval)/contourGranularity + 0.5
        
        c_obs_x = (obs_x - minXval)/contourGranularity + 0.5
        c_obs_y = (obs_y - minYval)/contourGranularity + 0.5
        
        theXticks = np.arange(minXval, maxXval+tickInterval, tickInterval)
        theYticks = np.arange(minYval, maxYval+tickInterval, tickInterval)
        
        c_theXticks = (theXticks - minXval)/contourGranularity + 0.5
        c_theYticks = (theYticks - minYval)/contourGranularity + 0.5
        
        c_obsNodes_x = (OBSNodes_x - minXval)/contourGranularity + 0.5
        c_obsNodes_y = (OBSNodes_y - minYval)/contourGranularity + 0.5
        
        c_obsNodesN_x = (OBSNodesN_x - minXval)/contourGranularity + 0.5
        c_obsNodesN_y = (OBSNodesN_y - minYval)/contourGranularity + 0.5
            
        
        if obs_x.size != 0:
            naninds = np.where(np.isnan(obs_x))[0]
            polyStartInds = naninds[0:-1] + 1
            polyStartInds = np.insert(polyStartInds, 0, 0)
            polyEndInds = naninds - 1
        else:
            polyStartInds = np.array([])
            polyEndInds = np.array([])

        robot_Zind_x = math.floor(c_move_x[-1])+1
        robot_Zind_y = math.floor(c_move_y[-1])+1
        
        robot_minxZind = np.maximum(robot_Zind_x - 1, 0)
        robot_maxxZind = np.minimum(robot_Zind_x + 1, Z.shape[1] - 1)
        robot_minyZind = np.maximum(robot_Zind_y - 1, 0)
        robot_maxyZind = np.minimum(robot_Zind_y + 1, Z.shape[0] - 1)
        robotSamplePatch = Zmin[robot_minyZind:robot_maxyZind+1, robot_minxZind:robot_maxxZind+1]

        robotSampleVal = robotSamplePatch[np.where(robotSamplePatch != math.inf)]
        
        if robotSampleVal.size == 0:
            robotSampleVal = 0
        else:
            robotSampleVal = np.mean(robotSampleVal)
        plotContourValIndRobotVal = np.where(contourLevels >= robotSampleVal)[0]
        if plotContourValIndRobotVal.size == 0:
            plotContourValIndRobotVal = contourLevels.size - 1
        elif plotContourValIndRobotVal[0] > contourLevels.size - 1:
            plotContourValIndRobotVal = contourLevels.size - 1
        else:
            plotContourValIndRobotVal = plotContourValIndRobotVal[0]
        plotContourValIndRobotVal = np.minimum(plotContourValIndRobotVal + 1, contourLevels.size - 1)
        dividingCost = contourLevels[plotContourValIndRobotVal]
        Zmin[np.where(Zmin > dividingCost)] = math.inf
        Zmin[robot_minyZind:robot_maxyZind+1, robot_minxZind:robot_maxxZind+1] = robotSamplePatch
        
        tempcolormap = deepcopy(mycolormap)
        tempcolormap[plotContourValIndRobotVal:,:] = 0.3
        
        tempcolormap = list(tempcolormap)
        tempcolormap = [tuple(array) for array in tempcolormap]
        XS, YS = np.meshgrid(Xs, Ys)
        ax.contourf(XS, YS, Zmin, contourLevels, colors=tempcolormap)

        for i in np.arange(0, len(c_x), 2):
            ax.plot([c_x[i], c_x[i+1]], [c_y[i], c_y[i+1]], color='r', linewidth=0.1)
        
        ax.plot(c_node_x, c_node_y, marker='o', color='indianred', markersize=1, linestyle='none')
        ax.plot(c_move_x, c_move_y, color='k', linewidth=3)
        ax.plot(c_move_x, c_move_y, color='r', linewidth=1)
        
        if c_path_x.size != 0:
            ax.plot(c_path_x, c_path_y, color='k', linewidth=3)
            ax.plot(c_path_x, c_path_y, color='w', linewidth=1)
        
        for p in range(len(polyStartInds)):
            poly_inds = np.arange(polyStartInds[p],polyEndInds[p] + 1)
            polygon = [(c_obs_x[i], c_obs_y[i]) for i in poly_inds]
            ax.add_patch(patches.Polygon(polygon, color='k'))
            
        ax.plot(c_obs_x, c_obs_y, color='w', linewidth=1)
        
        if start_move_at_ctr > file_ctr:
            ax.plot(c_node_x[0], c_node_y[0], color='b', linewidth=1, markeredgecolor='k', markerfacecolor='w',markersize=8)
        elif path_x.size != 0:
            ax.plot(c_path_x[-1], c_path_y[-1], color='b', linewidth=1, markeredgecolor='k',markerfacecolor='w',markersize=8)
        
        ax.plot(c_obsNodes_x, c_obsNodes_y, marker='.', color='k')
        ax.plot(c_obsNodesN_x, c_obsNodesN_y, marker='o', color='k')
        
        if c_move_x.size < 2:
            c_move_x = np.concatenate((c_move_x, c_move_x), axis=0)
            c_move_y = np.concatenate((c_move_y, c_move_y), axis=0)
        if c_move_x.size != 0:
            plotVehicle(fig, np.array([c_move_x[-2], c_move_y[-2]]), np.array([c_move_x[-1], c_move_y[-1]]), .5, 'k', sensor_radius, '--w', ax)
        
        ax.set_title('RRT^X')
        ax.set_xlim(0, Z.shape[0])
        ax.set_ylim(0, Z.shape[1])
        ax.set_xticks(ticks=c_theXticks)
        ax.set_xticklabels(theXticks)
        ax.set_yticks(ticks=c_theYticks)
        ax.set_yticklabels(theYticks)
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()
    
def make_video_dubins(args):
    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.file_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr

    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50
    tickInterval = 10

    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)
    contourGranularity = 2.5 # x, y granularity for cost contour plot

    contourLevels = np.arange(0, 320, 3)
    mycolormap = np.zeros((contourLevels.size, 3))
    mycolormap = cm.jet(contourLevels)
    mycolormap = mycolormap[:,:3]

    sensor_radius = 10/contourGranularity

    videoFill = True

    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeMovieDubins.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)

        EdgeFile = open('{}edges_{}.txt'.format(dir, file_ctr))
        RawEdgeData = EdgeFile.readlines()
        EdgeData = []
        for edge_data in RawEdgeData:
            edge_data = edge_data.split(',')
            edge_data[-1] = edge_data[-1].replace('\n','')
            EdgeData.append(np.array(list(map(float, edge_data))))
        EdgeData = np.array(EdgeData)
        
        raw_x = EdgeData[:, 0]
        raw_y = EdgeData[:, 1]
        
        NodeFile = open('{}nodes_{}.txt'.format(dir, file_ctr))
        RawNodeData = NodeFile.readlines()
        NodeData = []
        for node_data in RawNodeData:
            node_data = node_data.split(',')
            node_data[-1] = node_data[-1].replace('\n','')
            NodeData.append(np.array(list(map(float, node_data))))
        NodeData = np.array(NodeData)
        
        node_x = EdgeData[:, 0]
        node_y = EdgeData[:, 1]
        node_z = EdgeData[:, -1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
        else:
            obs_x = np.array([])
            obs_y = np.array([])
        
        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        move_x = MoveData[:, 0]
        move_y = MoveData[:, 1]
        move_theta = MoveData[:, 3]
        
        
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = PathData[:, 0]
            path_y = PathData[:, 1]
        else:
            path_x = np.array([])
            path_y = np.array([])
            
        if os.path.exists('{}obsNodes.txt'.format(dir)):
            ObsNodeFile = open('{}obsNodes.txt'.format(dir))
            RawObsNodeData =  ObsNodeFile.readlines()
            ObsNodeData = []
            for obs_node_data in RawObsNodeData:
                obs_node_data = obs_node_data.split(',')
                obs_node_data[-1] = obs_node_data[-1].replace('\n','')
                ObsNodeData.append(np.array(list(map(float, obs_node_data))))
            ObsNodeData = np.array(ObsNodeData)
            OBSNodes_x = ObsNodeData[:, 0]
            OBSNodes_y = ObsNodeData[:, 1]
        else:
            OBSNodes_x = np.array([])
            OBSNodes_y = np.array([])
            
        if os.path.exists('{}obsNodesNeighbors.txt'.format(dir)):
            ObsNodeNFile = open('{}obsNodesNeighbors.txt'.format(dir))
            RawObsNodeNData =  ObsNodeNFile.readlines()
            ObsNodeNData = []
            for obs_node_n_data in RawObsNodeNData:
                obs_node_n_data = obs_node_n_data.split(',')
                obs_node_n_data[-1] = obs_node_n_data[-1].replace('\n','')
                ObsNodeNData.append(np.array(list(map(float, obs_node_n_data))))
            ObsNodeNData = np.array(ObsNodeNData)
            OBSNodesN_x = ObsNodeNData[:, 0]
            OBSNodesN_y = ObsNodeNData[:, 1]
        else:
            OBSNodesN_x = np.array([])
            OBSNodesN_y = np.array([])
        
        
        Xs = np.arange(minXval, maxXval, contourGranularity)
        Ys = np.arange(minYval, maxYval, contourGranularity)
        
        Z = np.zeros((Ys.size, Xs.size))
        Zmin = np.ones((Ys.size, Xs.size))*np.inf
        Counts = np.zeros((Ys.size, Xs.size))
        
        for v in range(len(node_z)):
            j = np.maximum(np.minimum(math.floor((node_x[v] - minXval)/contourGranularity), Z.shape[1] - 1), 0)
            i = np.maximum(np.minimum(math.floor((node_y[v] - minYval)/contourGranularity), Z.shape[0] - 1), 0)
            
            if np.isinf(node_z[v]):
                continue
            
            Z[i,j] = Z[i,j] + node_z[v]
            Counts[i,j] = Counts[i,j] + 1
            
            Zmin[i,j] = np.minimum(Zmin[i,j], node_z[v])
        
        Z[np.where(Z!=0)] = Z[np.where(Z!=0)]/Counts[np.where(Z!=0)]
        
        jjj = 1

        while np.sum(Z.reshape(-1)==0) > 0:
            jjj = jjj + 1
            
            if np.sum(Z.reshape(-1)!=0) < 5 or jjj > 3:
                Z[np.where(Z==0)] = contourLevels[-1]
                Zmin[np.where(Z==0)] = contourLevels[-1]
                break
            
            yZeroInds, xZeroInds = np.where(Z == 0)
            dZ = np.zeros_like(Z)
            dZmin = np.zeros_like(Zmin)
            for k in range(xZeroInds.size):
                xZind = xZeroInds[k]
                yZind = yZeroInds[k]
                
                minxZind = np.maximum(xZind - 1, 0)
                maxxZind = np.minimum(xZind + 1, Z.shape[1] - 1)
                minyZind = np.maximum(yZind - 1, 0)
                maxyZind = np.minimum(yZind + 1, Z.shape[0] - 1)
                
                if videoFill and (Ys.size-1)/2 - 2 < yZind and yZind < (Ys.size-1)/2 + 2:
                    minyZind = yZind
                    
                subZ = Z[minyZind:maxyZind+1, minxZind:maxxZind+1]
                if np.sum(subZ.reshape(-1) != 0) > 1:
                    dZ[yZind, xZind] = np.sum(subZ.reshape(-1))/np.sum(subZ.reshape(-1) != 0)
                    dZmin[yZind, xZind] = np.min(subZ[np.where(subZ != 0)])
            
            Z = Z + dZ
            Zmin[np.where(np.logical_and(np.isinf(Zmin), dZmin != 0))] = dZmin[np.where(np.logical_and(np.isinf(Zmin), dZmin != 0))]
            
        c_node_x = (node_x - minXval)/contourGranularity + 0.5
        c_node_y = (node_y - minYval)/contourGranularity + 0.5
        
        c_path_x = (path_x - minXval)/contourGranularity + 0.5
        c_path_y = (path_y - minYval)/contourGranularity + 0.5
        
        c_move_x = (move_x - minXval)/contourGranularity + 0.5
        c_move_y = (move_y - minYval)/contourGranularity + 0.5
        c_move_theta = move_theta - math.pi/2.0
        
        c_x = (raw_x - minXval)/contourGranularity + 0.5
        c_y = (raw_y - minYval)/contourGranularity + 0.5
        
        c_obs_x = (obs_x - minXval)/contourGranularity + 0.5
        c_obs_y = (obs_y - minYval)/contourGranularity + 0.5
        
        theXticks = np.arange(minXval, maxXval+tickInterval, tickInterval)
        theYticks = np.arange(minYval, maxYval+tickInterval, tickInterval)
        
        c_theXticks = (theXticks - minXval)/contourGranularity + 0.5
        c_theYticks = (theYticks - minYval)/contourGranularity + 0.5
        
        c_obsNodes_x = (OBSNodes_x - minXval)/contourGranularity + 0.5
        c_obsNodes_y = (OBSNodes_y - minYval)/contourGranularity + 0.5
        
        c_obsNodesN_x = (OBSNodesN_x - minXval)/contourGranularity + 0.5
        c_obsNodesN_y = (OBSNodesN_y - minYval)/contourGranularity + 0.5
            
        
        if obs_x.size != 0:
            naninds = np.where(np.isnan(obs_x))[0]
            polyStartInds = naninds[0:-1] + 1
            polyStartInds = np.insert(polyStartInds, 0, 0)
            polyEndInds = naninds - 1
        else:
            polyStartInds = np.array([])
            polyEndInds = np.array([])

        robot_Zind_x = math.floor(c_move_x[-1])+1
        robot_Zind_y = math.floor(c_move_y[-1])+1
        
        robot_minxZind = np.maximum(robot_Zind_x - 1, 0)
        robot_maxxZind = np.minimum(robot_Zind_x + 1, Z.shape[1] - 1)
        robot_minyZind = np.maximum(robot_Zind_y - 1, 0)
        robot_maxyZind = np.minimum(robot_Zind_y + 1, Z.shape[0] - 1)
        robotSamplePatch = Zmin[robot_minyZind:robot_maxyZind+1, robot_minxZind:robot_maxxZind+1]

        robotSampleVal = robotSamplePatch[np.where(robotSamplePatch != math.inf)]
        
        if robotSampleVal.size == 0:
            robotSampleVal = 0
        else:
            robotSampleVal = np.mean(robotSampleVal)
        plotContourValIndRobotVal = np.where(contourLevels >= robotSampleVal)[0]
        if plotContourValIndRobotVal.size == 0:
            plotContourValIndRobotVal = contourLevels.size - 1
        elif plotContourValIndRobotVal[0] > contourLevels.size - 1:
            plotContourValIndRobotVal = contourLevels.size - 1
        else:
            plotContourValIndRobotVal = plotContourValIndRobotVal[0]
        plotContourValIndRobotVal = np.minimum(plotContourValIndRobotVal + 1, contourLevels.size - 1)
        dividingCost = contourLevels[plotContourValIndRobotVal]
        Zmin[np.where(Zmin > dividingCost)] = math.inf
        Zmin[robot_minyZind:robot_maxyZind+1, robot_minxZind:robot_maxxZind+1] = robotSamplePatch
        
        tempcolormap = deepcopy(mycolormap)
        tempcolormap[plotContourValIndRobotVal:,:] = 0.3
        
        tempcolormap = list(tempcolormap)
        tempcolormap = [tuple(array) for array in tempcolormap]
        XS, YS = np.meshgrid(Xs, Ys)
        ax.contourf(XS, YS, Zmin, contourLevels, colors=tempcolormap)

        for i in np.arange(0, len(c_x), 2):
            ax.plot([c_x[i], c_x[i+1]], [c_y[i], c_y[i+1]], color='r', linewidth=0.1)
        
        ax.plot(c_node_x, c_node_y, marker='o', color='indianred', markersize=1, linestyle='none')
        ax.plot(c_move_x, c_move_y, color='k', linewidth=3)
        ax.plot(c_move_x, c_move_y, color='r', linewidth=1)
        
        if c_path_x.size != 0:
            ax.plot(c_path_x, c_path_y, color='k', linewidth=3)
            ax.plot(c_path_x, c_path_y, color='w', linewidth=1)
        
        for p in range(len(polyStartInds)):
            poly_inds = np.arange(polyStartInds[p],polyEndInds[p] + 1)
            polygon = [(c_obs_x[i], c_obs_y[i]) for i in poly_inds]
            ax.add_patch(patches.Polygon(polygon, color='k'))
            
        ax.plot(c_obs_x, c_obs_y, color='w', linewidth=1)
        
        if start_move_at_ctr > file_ctr:
            ax.plot(c_node_x[0], c_node_y[0], color='b', linewidth=1, markeredgecolor='k', markerfacecolor='w',markersize=8)
        elif path_x.size != 0:
            ax.plot(c_path_x[-1], c_path_y[-1], color='b', linewidth=1, markeredgecolor='k',markerfacecolor='w',markersize=8)
        
        ax.plot(c_obsNodes_x, c_obsNodes_y, marker='.', color='k')
        ax.plot(c_obsNodesN_x, c_obsNodesN_y, marker='o', color='k')
        
        if c_move_x.size != 0:
            plotVehicleTheta(fig, np.array([c_move_x[-1], c_move_y[-1]]), c_move_theta[-1], .5, 'k', sensor_radius, '--w', ax)
        
        ax.set_title('RRT^X')
        ax.set_xlim(0, Z.shape[0])
        ax.set_ylim(0, Z.shape[1])
        ax.set_xticks(ticks=c_theXticks)
        ax.set_xticklabels(theXticks)
        ax.set_yticks(ticks=c_theYticks)
        ax.set_yticklabels(theYticks)
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()

def make_time_video_static(args):

    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.start_move_at_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr
    obsShrinkProp = .8 
    carrot_dist = 100 
    
    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50
    maxObs = 11
    obs_used = np.zeros((maxObs)) # uses so each obstacle keeps thier randomly
    # generated paramiters
    obs_type = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]) # 1 = spike, 2 = pacman

    obs_clrs = [(.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35), (.4, .4, .4), (.45, .45, .45), (.5, .5, .5), (.55, .55, .55), (.6, .6, .6)]
    obs_theta = {}
    obs_d_theta = {}
    obs_spikes = {}
    obs_mouth = {}
    d_obs_mouth = {}

    max_time = 0
    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)

    # open the video file
    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeStaticMovie.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)

        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        if MoveData.size > 0:
            move_x = MoveData[:, 0]
            move_y = MoveData[:, 1]
            move_time = MoveData[:, 2]
            move_theta = 0
        else:
            move_x = np.array([math.nan])
            move_y = np.array([math.nan])
            move_time = np.array([math.nan])
            
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = np.insert(PathData[:, 0], 0, move_x[-1])
            path_y = np.insert(PathData[:, 1], 0, move_y[-1])
            path_time = np.insert(PathData[:, 2], 0, move_time[-1])
        else:
            path_x = np.array([move_x[-1]])
            path_y = np.array([move_y[-1]])
            path_time = np.array([move_time[-1]])
            
        ind = np.where(np.logical_and(path_x == path_x[-1], path_y == path_y[-1]))[0][0]
        path_x = path_x[:ind+1]
        path_y = path_y[:ind+1]
        path_time = path_time[:ind+1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
            obs_time = ObstacleData[:, 2]
            obs_rads = ObstacleData[:, 3] * obsShrinkProp
            
            nan_inds = np.where(np.isnan(ObstacleData[:, 0]))[0]
            start_inds = np.insert(nan_inds[:-1] + 1, 0, 0)
            end_inds = nan_inds-1
            
            obs_x_est = []
            obs_y_est = []
            obs_times_est = []
            obs_centers_x = {}
            obs_centers_y = {}
            obs_centers_time = {}
            obs_rad = {}
            obs_tradj = {}
            obs_speed = {}
            obs_centers_next_x = {}
            obs_centers_next_y = {}
            dist_to_carrot = {}
            obs_centers_next_time = {}
            for i in range(end_inds.size):
                si = start_inds[i]
                ei = end_inds[i]
                
                this_obs_times = obs_time[si:ei+1]
                this_obs_x = obs_x[si:ei+1]
                this_obs_y = obs_y[si:ei+1]
                obstacle_radius = obs_rads[si]
                
                before_now_ind = np.where(this_obs_times <= path_time[0])[0]
                
                if before_now_ind.size > 0:
                    before_now_ind = before_now_ind[-1]
                else:
                    before_now_ind = 0
            
                after_now_ind = np.where(this_obs_times >= path_time[0])[0]
                if after_now_ind.size > 0:
                    after_now_ind = after_now_ind[0]
                else:
                    after_now_ind = this_obs_times.size - 1
                
                edge_fraction = (path_time[0] - this_obs_times[before_now_ind])/(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                
                if np.isnan(edge_fraction):
                    edge_fraction = 0
                
                obs_centers_x[i] = this_obs_x[before_now_ind] + edge_fraction*(this_obs_x[after_now_ind] - this_obs_x[before_now_ind])
                obs_centers_y[i] = this_obs_y[before_now_ind] + edge_fraction*(this_obs_y[after_now_ind] - this_obs_y[before_now_ind])
                obs_centers_time[i] = this_obs_times[before_now_ind] + edge_fraction*(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                obs_rad[i] = obstacle_radius
                
                if before_now_ind == after_now_ind:
                    if after_now_ind == 0:
                        obs_tradj[i] = 0
                        obs_speed[i] = 0

                    else:
                        obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[after_now_ind-1],this_obs_x[after_now_ind] - this_obs_x[after_now_ind-1]) + math.pi
                        obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind-1])**2 + (this_obs_y[after_now_ind-1] - this_obs_y[after_now_ind])**2)/(this_obs_times[after_now_ind-1] - this_obs_times[after_now_ind])
                    
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/(obs_speed[i]+1e-10)
                
                else:
                    obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[before_now_ind],this_obs_x[after_now_ind] - this_obs_x[before_now_ind]) + math.pi
                    obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind])**2 + (this_obs_y[before_now_ind] - this_obs_y[after_now_ind])**2)/(this_obs_times[before_now_ind] - this_obs_times[after_now_ind])
                    # these are the carrot (carrot on a stick) used to show estimated movement
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
            
                obs_x_est.append(np.array([obs_centers_x[i], obs_centers_next_x[i], math.nan]))
                obs_y_est.append(np.array([obs_centers_y[i], obs_centers_next_y[i], math.nan]))
                obs_times_est.append(np.array([obs_centers_time[i], obs_centers_next_time[i], math.nan]))
                
                if not (i in obs_tradj):
                    obs_tradj[i] = 0
                if obs_used[i] == 0:
                    obs_theta[i] = np.random.rand()*math.pi
                    obs_d_theta[i] =  .35*math.pi/10 * 3*math.sqrt(obs_rad[i])/10
                    if np.random.rand() > 0.5:
                        obs_d_theta[i] = -obs_d_theta[i]
                    
                    obs_spikes[i] = 5 + math.floor(np.random.rand()*5)
                    
                    obs_mouth[i] = math.pi/6
                    d_obs_mouth[i] = -(math.pi/3)/6
                    
                    obs_used[i] = 1
                
                if not (i in obs_centers_x): # obstacle has finished moving, assume that it stays at the same place
                    obs_centers_x[i] = this_obs_x[0]
                    obs_centers_y[i] = this_obs_y[0]
                    
            obs_x_est = np.array(obs_x_est).reshape(-1)
            obs_y_est = np.array(obs_y_est).reshape(-1)
            obs_times_est = np.array(obs_times_est).reshape(-1)
        else:
            obs_x = np.array([])
            obs_y = np.array([])
            obs_time = np.array([])
            obs_time = np.array([])
            obs_rads = np.array([])
            obs_centers_x = {}
            obs_centers_y = {}
        
        ax.plot(move_x, move_y, color='k', linewidth=3, linestyle="dotted")
        ax.plot(move_x, move_y, color='k', linewidth=1)
        ax.plot(path_x[-1], path_y[-1], marker='o', linewidth=1, markeredgecolor='k', markerfacecolor='w', markersize=8)
        
        time_adjust = path_time[-1]
        path_time_prime = path_time - time_adjust
        obs_time_prime = obs_time - time_adjust
        obs_time_prime_est = obs_times_est - time_adjust
        if path_time_prime[0] > max_time:
            max_time = path_time_prime[0]
        
        obs_x = obs_x[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        obs_y = obs_y[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        obs_time_prime = obs_time_prime[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        if file_ctr > start_move_at_ctr:
            colorPlot(obs_x, obs_y, obs_time_prime, 1, ax)
        for i in range(len(obs_centers_x)):
            if obs_type[i] == 1:
                plotSpikes(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_theta[i], obs_d_theta[i], obs_spikes[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_theta[i] = obs_theta[i] + obs_d_theta[i]
            elif obs_type[i] == 2:
                plotPackman(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_tradj[i], obs_mouth[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_mouth[i] = obs_mouth[i] + d_obs_mouth[i]
                    
                    if obs_mouth[i] >= math.pi/3 - .01:
                        obs_mouth[i] = math.pi/3
                        d_obs_mouth[i] = -d_obs_mouth[i]
                    elif obs_mouth[i] <= .01:
                        obs_mouth[i] = 0
                        d_obs_mouth[i] = -d_obs_mouth[i]
        if file_ctr > start_move_at_ctr:
            colorPlot(path_x, path_y, path_time_prime, 1.5, ax)

        if move_x.size < 2:
            move_x = np.concatenate((move_x, move_x), axis=0)
            move_y = np.concatenate((move_y, move_y), axis=0)
        if move_x.size != 0:
            plotVehicle(fig, np.array([move_x[-2], move_y[-2]]), np.array([move_x[-1], move_y[-1]]), 2, 'k', math.nan, None, ax)
        
        ax.set_title('robot and obstacle pose (x,y) vs. robot time-to-goal (color)')
        ax.set_xlim(minXval, maxXval)
        ax.set_ylim(minYval, maxYval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()
    
def make_time_video_static_dubins(args):

    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.start_move_at_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr
    obsShrinkProp = .8 
    carrot_dist = 100 
    

    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50
    maxObs = 10
    obs_used = np.zeros((maxObs)) # uses so each obstacle keeps thier randomly
    # generated paramiters
    obs_type = np.array([1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1]) # 1 = spike, 2 = pacman

    obs_clrs = [(.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35), (.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35)]
    obs_theta = {}
    obs_d_theta = {}
    obs_spikes = {}
    obs_mouth = {}
    d_obs_mouth = {}

    max_time = 0
    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)

    # open the video file
    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeStaticMovieDubins.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)

        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        if MoveData.size > 0:
            move_x = MoveData[:, 0]
            move_y = MoveData[:, 1]
            move_time = MoveData[:, 2]
            move_theta = MoveData[:, 3] - math.pi/2
        else:
            move_x = np.array([math.nan])
            move_y = np.array([math.nan])
            move_time = np.array([math.nan])
            
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = np.insert(PathData[:, 0], 0, move_x[-1])
            path_y = np.insert(PathData[:, 1], 0, move_y[-1])
            path_time = np.insert(PathData[:, 2], 0, move_time[-1])
        else:
            path_x = np.array([move_x[-1]])
            path_y = np.array([move_y[-1]])
            path_time = np.array([move_time[-1]])
            
        ind = np.where(np.logical_and(path_x == path_x[-1], path_y == path_y[-1]))[0][0]
        path_x = path_x[:ind+1]
        path_y = path_y[:ind+1]
        path_time = path_time[:ind+1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
            obs_time = ObstacleData[:, 2]
            obs_rads = ObstacleData[:, 3] * obsShrinkProp
            
            nan_inds = np.where(np.isnan(ObstacleData[:, 0]))[0]
            start_inds = np.insert(nan_inds[:-1] + 1, 0, 0)
            end_inds = nan_inds-1
            
            obs_x_est = []
            obs_y_est = []
            obs_times_est = []
            obs_centers_x = {}
            obs_centers_y = {}
            obs_centers_time = {}
            obs_rad = {}
            obs_tradj = {}
            obs_speed = {}
            obs_centers_next_x = {}
            obs_centers_next_y = {}
            dist_to_carrot = {}
            obs_centers_next_time = {}
            for i in range(end_inds.size):
                si = start_inds[i]
                ei = end_inds[i]
                
                this_obs_times = obs_time[si:ei+1]
                this_obs_x = obs_x[si:ei+1]
                this_obs_y = obs_y[si:ei+1]
                obstacle_radius = obs_rads[si]
                
                before_now_ind = np.where(this_obs_times <= path_time[0])[0]
                
                if before_now_ind.size > 0:
                    before_now_ind = before_now_ind[-1]
                else:
                    before_now_ind = 0
            
                after_now_ind = np.where(this_obs_times >= path_time[0])[0]
                if after_now_ind.size > 0:
                    after_now_ind = after_now_ind[0]
                else:
                    after_now_ind = this_obs_times.size - 1
                
                edge_fraction = (path_time[0] - this_obs_times[before_now_ind])/(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                
                if np.isnan(edge_fraction):
                    edge_fraction = 0
                
                obs_centers_x[i] = this_obs_x[before_now_ind] + edge_fraction*(this_obs_x[after_now_ind] - this_obs_x[before_now_ind])
                obs_centers_y[i] = this_obs_y[before_now_ind] + edge_fraction*(this_obs_y[after_now_ind] - this_obs_y[before_now_ind])
                obs_centers_time[i] = this_obs_times[before_now_ind] + edge_fraction*(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                obs_rad[i] = obstacle_radius
                
                if before_now_ind == after_now_ind:
                    if after_now_ind == 0:
                        obs_tradj[i] = 0
                        obs_speed[i] = 0

                    else:
                        obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[after_now_ind-1],this_obs_x[after_now_ind] - this_obs_x[after_now_ind-1]) + math.pi
                        obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind-1])**2 + (this_obs_y[after_now_ind-1] - this_obs_y[after_now_ind])**2)/(this_obs_times[after_now_ind-1] - this_obs_times[after_now_ind])
                    
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/(obs_speed[i]+1e-10)
                
                else:
                    obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[before_now_ind],this_obs_x[after_now_ind] - this_obs_x[before_now_ind]) + math.pi
                    obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind])**2 + (this_obs_y[before_now_ind] - this_obs_y[after_now_ind])**2)/(this_obs_times[before_now_ind] - this_obs_times[after_now_ind])
                    # these are the carrot (carrot on a stick) used to show estimated movement
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
            
                obs_x_est.append(np.array([obs_centers_x[i], obs_centers_next_x[i], math.nan]))
                obs_y_est.append(np.array([obs_centers_y[i], obs_centers_next_y[i], math.nan]))
                obs_times_est.append(np.array([obs_centers_time[i], obs_centers_next_time[i], math.nan]))
                
                if not (i in obs_tradj):
                    obs_tradj[i] = 0
                if obs_used[i] == 0:
                    obs_theta[i] = np.random.rand()*math.pi
                    obs_d_theta[i] =  .35*math.pi/10 * 3*math.sqrt(obs_rad[i])/10
                    if np.random.rand() > 0.5:
                        obs_d_theta[i] = -obs_d_theta[i]
                    
                    obs_spikes[i] = 5 + math.floor(np.random.rand()*5)
                    
                    obs_mouth[i] = math.pi/6
                    d_obs_mouth[i] = -(math.pi/3)/6
                    
                    obs_used[i] = 1
                
                if not (i in obs_centers_x): # obstacle has finished moving, assume that it stays at the same place
                    obs_centers_x[i] = this_obs_x[0]
                    obs_centers_y[i] = this_obs_y[0]
                    
            obs_x_est = np.array(obs_x_est).reshape(-1)
            obs_y_est = np.array(obs_y_est).reshape(-1)
            obs_times_est = np.array(obs_times_est).reshape(-1)
        else:
            obs_x = np.array([])
            obs_y = np.array([])
            obs_time = np.array([])
            obs_time = np.array([])
            obs_rads = np.array([])
            obs_centers_x = {}
            obs_centers_y = {}
        
        ax.plot(move_x, move_y, color='k', linewidth=3, linestyle="dotted")
        ax.plot(move_x, move_y, color='k', linewidth=1)
        ax.plot(path_x[-1], path_y[-1], marker='o', linewidth=1, markeredgecolor='k', markerfacecolor='w', markersize=8)
        
        time_adjust = path_time[-1]
        path_time_prime = path_time - time_adjust
        obs_time_prime = obs_time - time_adjust
        obs_time_prime_est = obs_times_est - time_adjust
        if path_time_prime[0] > max_time:
            max_time = path_time_prime[0]
        
        obs_x = obs_x[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        obs_y = obs_y[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        obs_time_prime = obs_time_prime[np.where(np.logical_or(obs_time_prime > 0, np.isnan(obs_time_prime)))]
        if file_ctr > start_move_at_ctr:
            colorPlot(obs_x, obs_y, obs_time_prime, 1, ax)
        for i in range(len(obs_centers_x)):
            if obs_type[i] == 1:
                plotSpikes(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_theta[i], obs_d_theta[i], obs_spikes[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_theta[i] = obs_theta[i] + obs_d_theta[i]
            elif obs_type[i] == 2:
                plotPackman(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_tradj[i], obs_mouth[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_mouth[i] = obs_mouth[i] + d_obs_mouth[i]
                    
                    if obs_mouth[i] >= math.pi/3 - .01:
                        obs_mouth[i] = math.pi/3
                        d_obs_mouth[i] = -d_obs_mouth[i]
                    elif obs_mouth[i] <= .01:
                        obs_mouth[i] = 0
                        d_obs_mouth[i] = -d_obs_mouth[i]
        if file_ctr > start_move_at_ctr:
            colorPlot(path_x, path_y, path_time_prime, 1.5, ax)

        if move_x.size != 0:
            plotVehicleTheta(fig, np.array([move_x[-1], move_y[-1]]), move_theta[-1], 2, 'k', math.nan, None, ax)
        
        ax.set_title('robot and obstacle pose (x,y) vs. robot time-to-goal (color)')
        ax.set_xlim(minXval, maxXval)
        ax.set_ylim(minYval, maxYval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()

def make_time_video_dynamic(args):

    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.start_move_at_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr
    obsShrinkProp = .8 
    carrot_dist = 100 
    
    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50

    maxObs = 11
    obs_used = np.zeros((maxObs)) # uses so each obstacle keeps thier randomly
    # generated paramiters
    obs_type = np.array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]) # 1 = spike, 2 = pacman

    obs_clrs = [(.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35), (.4, .4, .4), (.45, .45, .45), (.5, .5, .5), (.55, .55, .55), (.6, .6, .6)]
    obs_theta = {}
    obs_d_theta = {}
    obs_spikes = {}
    obs_mouth = {}
    d_obs_mouth = {}

    max_time = 0
    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)

    # open the video file
    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeDynamicMovie.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)
        
        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        if MoveData.size > 0:
            move_x = MoveData[:, 0]
            move_y = MoveData[:, 1]
            move_time = MoveData[:, 2]
            move_theta = 0
            
        else:
            move_x = np.array([math.nan])
            move_y = np.array([math.nan])
            move_time = np.array([math.nan])
            
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = np.insert(PathData[:, 0], 0, move_x[-1])
            path_y = np.insert(PathData[:, 1], 0, move_y[-1])
            path_time = np.insert(PathData[:, 2], 0, move_time[-1])
        else:
            path_x = np.array([move_x[-1]])
            path_y = np.array([move_y[-1]])
            path_time = np.array([move_time[-1]])
            
        ind = np.where(np.logical_and(path_x == path_x[-1], path_y == path_y[-1]))[0][0]
        path_x = path_x[:ind+1]
        path_y = path_y[:ind+1]
        path_time = path_time[:ind+1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
            obs_time = ObstacleData[:, 2]
            obs_rads = ObstacleData[:, 3] * obsShrinkProp
            
            nan_inds = np.where(np.isnan(ObstacleData[:, 0]))[0]
            start_inds = np.insert(nan_inds[:-1] + 1, 0, 0)
            end_inds = nan_inds-1
            
            obs_x_est = []
            obs_y_est = []
            obs_times_est = []
            obs_centers_x = {}
            obs_centers_y = {}
            obs_centers_time = {}
            obs_rad = {}
            obs_tradj = {}
            obs_speed = {}
            obs_centers_next_x = {}
            obs_centers_next_y = {}
            dist_to_carrot = {}
            obs_centers_next_time = {}
            for i in range(end_inds.size):
                si = start_inds[i]
                ei = end_inds[i]
                
                this_obs_times = obs_time[si:ei+1]
                this_obs_x = obs_x[si:ei+1]
                this_obs_y = obs_y[si:ei+1]
                obstacle_radius = obs_rads[si]
                
                before_now_ind = np.where(this_obs_times <= path_time[0])[0]
                
                if before_now_ind.size > 0:
                    before_now_ind = before_now_ind[-1]
                else:
                    before_now_ind = 0
            
                after_now_ind = np.where(this_obs_times >= path_time[0])[0]
                if after_now_ind.size > 0:
                    after_now_ind = after_now_ind[0]
                else:
                    after_now_ind = this_obs_times.size - 1
                
                edge_fraction = (path_time[0] - this_obs_times[before_now_ind])/(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                
                if np.isnan(edge_fraction):
                    edge_fraction = 0
                
                obs_centers_x[i] = this_obs_x[before_now_ind] + edge_fraction*(this_obs_x[after_now_ind] - this_obs_x[before_now_ind])
                obs_centers_y[i] = this_obs_y[before_now_ind] + edge_fraction*(this_obs_y[after_now_ind] - this_obs_y[before_now_ind])
                obs_centers_time[i] = this_obs_times[before_now_ind] + edge_fraction*(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                obs_rad[i] = obstacle_radius
                
                if before_now_ind == after_now_ind:
                    if after_now_ind == 0:
                        obs_tradj[i] = 0
                        obs_speed[i] = 0

                    else:
                        obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[after_now_ind-1],this_obs_x[after_now_ind] - this_obs_x[after_now_ind-1]) + math.pi
                        obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind-1])**2 + (this_obs_y[after_now_ind-1] - this_obs_y[after_now_ind])**2)/(this_obs_times[after_now_ind-1] - this_obs_times[after_now_ind])
                    
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
                else:
                    obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[before_now_ind],this_obs_x[after_now_ind] - this_obs_x[before_now_ind]) + math.pi
                    obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind])**2 + (this_obs_y[before_now_ind] - this_obs_y[after_now_ind])**2)/(this_obs_times[before_now_ind] - this_obs_times[after_now_ind])
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
            
                obs_x_est.append(np.array([obs_centers_x[i], obs_centers_next_x[i], math.nan]))
                obs_y_est.append(np.array([obs_centers_y[i], obs_centers_next_y[i], math.nan]))
                obs_times_est.append(np.array([obs_centers_time[i], obs_centers_next_time[i], math.nan]))
                
                if not (i in obs_tradj):
                    obs_tradj[i] = 0
                if obs_used[i] == 0:
                    obs_theta[i] = np.random.rand()*math.pi
                    obs_d_theta[i] =  .35*math.pi/10 * 3*math.sqrt(obs_rad[i])/10
                    if np.random.rand() > 0.5:
                        obs_d_theta[i] = -obs_d_theta[i]
                    
                    obs_spikes[i] = 5 + math.floor(np.random.rand()*5)
                    
                    obs_mouth[i] = math.pi/6
                    d_obs_mouth[i] = -(math.pi/3)/6
                    
                    obs_used[i] = 1
                
                if not (i in obs_centers_x):
                    obs_centers_x[i] = this_obs_x[0]
                    obs_centers_y[i] = this_obs_y[0]
                    
            obs_x_est = np.array(obs_x_est).reshape(-1)
            obs_y_est = np.array(obs_y_est).reshape(-1)
            obs_times_est = np.array(obs_times_est).reshape(-1)
        else:
            obs_x = np.array([])
            obs_y = np.array([])
            obs_time = np.array([])
            obs_time = np.array([])
            obs_rads = np.array([])
            obs_centers_x = {}
            obs_centers_y = {}
        
        ax.plot(move_x, move_y, color='k', linewidth=3, linestyle="dotted")
        ax.plot(move_x, move_y, color='k', linewidth=1)
        ax.plot(path_x[-1], path_y[-1], marker='o', linewidth=1, markeredgecolor='k', markerfacecolor='w', markersize=8)
        
        time_adjust = path_time[-1]
        path_time_prime = path_time - time_adjust
        obs_time_prime = obs_time - time_adjust
        obs_time_prime_est = obs_times_est - time_adjust
        if path_time_prime[0] > max_time:
            max_time = path_time_prime[0]
            
        if file_ctr > start_move_at_ctr:
            colorPlot(obs_x_est, obs_y_est, obs_time_prime_est, 1, ax)
        for i in range(len(obs_centers_x)):
            if obs_type[i] == 1:
                plotSpikes(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_theta[i], obs_d_theta[i], obs_spikes[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_theta[i] = obs_theta[i] + obs_d_theta[i]
            elif obs_type[i] == 2:
                plotPackman(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_tradj[i], obs_mouth[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_mouth[i] = obs_mouth[i] + d_obs_mouth[i]
                    
                    if obs_mouth[i] >= math.pi/3 - .01:
                        obs_mouth[i] = math.pi/3
                        d_obs_mouth[i] = -d_obs_mouth[i]
                    elif obs_mouth[i] <= .01:
                        obs_mouth[i] = 0
                        d_obs_mouth[i] = -d_obs_mouth[i]
        if file_ctr > start_move_at_ctr:
            colorPlot(path_x, path_y, path_time_prime, 1.5, ax)

        if move_x.size < 2:
            move_x = np.concatenate((move_x, move_x), axis=0)
            move_y = np.concatenate((move_y, move_y), axis=0)
        if move_x.size != 0:
            plotVehicle(fig, np.array([move_x[-2], move_y[-2]]), np.array([move_x[-1], move_y[-1]]), 2, 'k', math.nan, None, ax)
        
        ax.set_title('robot and obstacle pose (x,y) vs. robot time-to-goal (color)')
        ax.set_xlim(minXval, maxXval)
        ax.set_ylim(minYval, maxYval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()

def make_time_video_dynamic_dubins(args):
    dir = 'temp/{}/'.format(args.exp_name)
    file_ctr = args.start_move_at_ctr
    max_file_ctr = args.max_file_ctr

    start_move_at_ctr = args.start_move_at_ctr
    obsShrinkProp = .8 
    carrot_dist = 100 
    
    minXval = -50
    minYval = -50
    maxXval = 50
    maxYval = 50

    maxObs = 10
    obs_used = np.zeros((maxObs)) # uses so each obstacle keeps thier randomly
    # generated paramiters
    obs_type = np.array([1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1]) # 1 = spike, 2 = pacman

    obs_clrs = [(.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35), (.1, .1, .1), (.15, .15, .15), (.2, .2, .2), (.25, .25, .25), (.3, .3, .3), (.35, .35, .35)]
    obs_theta = {}
    obs_d_theta = {}
    obs_spikes = {}
    obs_mouth = {}
    d_obs_mouth = {}

    plt.rcParams["figure.figsize"] = (9.6, 9.6)
    fig, ax = plt.subplots(1,1)

    max_time = 0
    fps = args.fps
    size = (args.height, args.width)
    # open the video file
    writerObj = cv2.VideoWriter('{}TimeDynamicMovieDubins.mp4'.format(dir), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while os.path.exists('{}robot_move_path_{}.txt'.format(dir, file_ctr)) and file_ctr < max_file_ctr:
        print(file_ctr)
        
        MoveFile = open('{}robot_move_path_{}.txt'.format(dir, file_ctr))
        RawMoveData = MoveFile.readlines()
        MoveData = []
        for move_data in RawMoveData:
            move_data = move_data.split(',')
            move_data[-1] = move_data[-1].replace('\n','')
            MoveData.append(np.array(list(map(float, move_data))))
        MoveData = np.array(MoveData)
        
        if MoveData.size > 0:
            move_x = MoveData[:, 0]
            move_y = MoveData[:, 1]
            move_time = MoveData[:, 2]
            move_theta = MoveData[:, 3] - math.pi/2

            if start_move_at_ctr < file_ctr: # calculate from path segments instead
                mi = move_x.size
                move_theta = math.atan2(move_y(mi) - move_y(mi-1), move_x(mi) - move_x(mi-1))
        else:
            move_x = np.array([math.nan])
            move_y = np.array([math.nan])
            move_time = np.array([math.nan])
            
        PathFile = open('{}path_{}.txt'.format(dir, file_ctr))
        RawPathData = PathFile.readlines()
        PathData = []
        for path_data in RawPathData:
            path_data = path_data.split(',')
            path_data[-1] = path_data[-1].replace('\n','')
            PathData.append(np.array(list(map(float, path_data))))
        PathData = np.array(PathData)
        
        if PathData.size > 0:
            path_x = np.insert(PathData[:, 0], 0, move_x[-1])
            path_y = np.insert(PathData[:, 1], 0, move_y[-1])
            path_time = np.insert(PathData[:, 2], 0, move_time[-1])
        else:
            path_x = np.array([move_x[-1]])
            path_y = np.array([move_y[-1]])
            path_time = np.array([move_time[-1]])
            
        ind = np.where(np.logical_and(path_x == path_x[-1], path_y == path_y[-1]))[0][0]
        path_x = path_x[:ind+1]
        path_y = path_y[:ind+1]
        path_time = path_time[:ind+1]
        
        ObstacleFile = open('{}obstacles_{}.txt'.format(dir, file_ctr))
        RawObstacleData = ObstacleFile.readlines()
        ObstacleData = []
        for obstacle_data in RawObstacleData:
            obstacle_data = obstacle_data.split(',')
            obstacle_data[-1] = obstacle_data[-1].replace('\n','')
            ObstacleData.append(np.array(list(map(float, obstacle_data))))
        ObstacleData = np.array(ObstacleData)
        
        if ObstacleData.size > 0:
            obs_x = ObstacleData[:, 0]
            obs_y = ObstacleData[:, 1]
            obs_time = ObstacleData[:, 2]
            obs_rads = ObstacleData[:, 3] * obsShrinkProp
            
            nan_inds = np.where(np.isnan(ObstacleData[:, 0]))[0]
            start_inds = np.insert(nan_inds[:-1] + 1, 0, 0)
            end_inds = nan_inds-1
            
            obs_x_est = []
            obs_y_est = []
            obs_times_est = []
            obs_centers_x = {}
            obs_centers_y = {}
            obs_centers_time = {}
            obs_rad = {}
            obs_tradj = {}
            obs_speed = {}
            obs_centers_next_x = {}
            obs_centers_next_y = {}
            dist_to_carrot = {}
            obs_centers_next_time = {}
            for i in range(end_inds.size):
                si = start_inds[i]
                ei = end_inds[i]
                
                this_obs_times = obs_time[si:ei+1]
                this_obs_x = obs_x[si:ei+1]
                this_obs_y = obs_y[si:ei+1]
                obstacle_radius = obs_rads[si]
                
                before_now_ind = np.where(this_obs_times <= path_time[0])[0]
                
                if before_now_ind.size > 0:
                    before_now_ind = before_now_ind[-1]
                else:
                    before_now_ind = 0
            
                after_now_ind = np.where(this_obs_times >= path_time[0])[0]
                if after_now_ind.size > 0:
                    after_now_ind = after_now_ind[0]
                else:
                    after_now_ind = this_obs_times.size - 1
                
                edge_fraction = (path_time[0] - this_obs_times[before_now_ind])/(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                
                if np.isnan(edge_fraction):
                    edge_fraction = 0
                
                obs_centers_x[i] = this_obs_x[before_now_ind] + edge_fraction*(this_obs_x[after_now_ind] - this_obs_x[before_now_ind])
                obs_centers_y[i] = this_obs_y[before_now_ind] + edge_fraction*(this_obs_y[after_now_ind] - this_obs_y[before_now_ind])
                obs_centers_time[i] = this_obs_times[before_now_ind] + edge_fraction*(this_obs_times[after_now_ind] - this_obs_times[before_now_ind])
                obs_rad[i] = obstacle_radius
                
                if before_now_ind == after_now_ind:
                    if after_now_ind == 0:
                        obs_tradj[i] = 0
                        obs_speed[i] = 0

                    else:
                        obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[after_now_ind-1],this_obs_x[after_now_ind] - this_obs_x[after_now_ind-1]) + math.pi
                        obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind-1])**2 + (this_obs_y[after_now_ind-1] - this_obs_y[after_now_ind])**2)/(this_obs_times[after_now_ind-1] - this_obs_times[after_now_ind])
                    
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
                else:
                    obs_tradj[i] = math.atan2(this_obs_y[after_now_ind] - this_obs_y[before_now_ind],this_obs_x[after_now_ind] - this_obs_x[before_now_ind]) + math.pi
                    obs_speed[i] = -math.sqrt((this_obs_x[before_now_ind] - this_obs_x[after_now_ind])**2 + (this_obs_y[before_now_ind] - this_obs_y[after_now_ind])**2)/(this_obs_times[before_now_ind] - this_obs_times[after_now_ind])
                    obs_centers_next_x[i] = obs_centers_x[i] + carrot_dist*math.cos(obs_tradj[i])
                    obs_centers_next_y[i] = obs_centers_y[i] + carrot_dist*math.sin(obs_tradj[i])
                    
                    dist_to_carrot[i] = math.sqrt((obs_centers_x[i]-obs_centers_next_x[i])**2 + (obs_centers_y[i]-obs_centers_next_y[i])**2)
                    obs_centers_next_time[i] = obs_centers_time[i] - dist_to_carrot[i]/obs_speed[i]
            
                obs_x_est.append(np.array([obs_centers_x[i], obs_centers_next_x[i], math.nan]))
                obs_y_est.append(np.array([obs_centers_y[i], obs_centers_next_y[i], math.nan]))
                obs_times_est.append(np.array([obs_centers_time[i], obs_centers_next_time[i], math.nan]))
                
                if not (i in obs_tradj):
                    obs_tradj[i] = 0
                if obs_used[i] == 0:
                    obs_theta[i] = np.random.rand()*math.pi
                    obs_d_theta[i] =  .35*math.pi/10 * 3*math.sqrt(obs_rad[i])/10
                    if np.random.rand() > 0.5:
                        obs_d_theta[i] = -obs_d_theta[i]
                    
                    obs_spikes[i] = 5 + math.floor(np.random.rand()*5)
                    
                    obs_mouth[i] = math.pi/6
                    d_obs_mouth[i] = -(math.pi/3)/6
                    
                    obs_used[i] = 1
                
                if not (i in obs_centers_x):
                    obs_centers_x[i] = this_obs_x[0]
                    obs_centers_y[i] = this_obs_y[0]
                    
        else:
            obs_x = np.array([])
            obs_y = np.array([])
            obs_time = np.array([])
            obs_rads = np.array([])
            obs_centers_x = {}
            obs_centers_y = {}
        
        ax.plot(move_x, move_y, color='k', linewidth=3, linestyle="dotted")
        ax.plot(move_x, move_y, color='k', linewidth=1)
        ax.plot(path_x[-1], path_y[-1], marker='o', linewidth=1, markeredgecolor='k', markerfacecolor='w', markersize=8)
        
        time_adjust = path_time[-1]
        path_time_prime = path_time - time_adjust
        obs_time_prime = obs_time - time_adjust
        obs_time_prime_est = obs_times_est - time_adjust
        if path_time_prime[0] > max_time:
            max_time = path_time_prime[0]
            
        if file_ctr > start_move_at_ctr:
            colorPlot(obs_x, obs_y, obs_time_prime, 1, ax)

        for i in range(len(obs_centers_x)):
            if obs_type[i] == 1:
                plotSpikes(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_theta[i], obs_d_theta[i], obs_spikes[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_theta[i] = obs_theta[i] + obs_d_theta[i]
            elif obs_type[i] == 2:
                plotPackman(np.array([obs_centers_x[i], obs_centers_y[i]]), obs_rad[i], obs_tradj[i], obs_mouth[i], obs_clrs[i], ax)
                if file_ctr > start_move_at_ctr:
                    obs_mouth[i] = obs_mouth[i] + d_obs_mouth[i]
                    
                    if obs_mouth[i] >= math.pi/3 - .01:
                        obs_mouth[i] = math.pi/3
                        d_obs_mouth[i] = -d_obs_mouth[i]
                    elif obs_mouth[i] <= .01:
                        obs_mouth[i] = 0
                        d_obs_mouth[i] = -d_obs_mouth[i]
        if file_ctr > start_move_at_ctr:
            colorPlot(path_x, path_y, path_time_prime, 1.5, ax)
        if move_x.size != 0:
            plotVehicleTheta(fig, np.array([move_x[-1], move_y[-1]]), move_theta[-1], 2, 'k', np.nan, '--w')

        ax.set_title('robot and obstacle pose (x,y) vs. robot time-to-goal (color)')
        ax.set_xlim(minXval, maxXval)
        ax.set_ylim(minYval, maxYval)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        fig.canvas.draw()
        
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, size)
        writerObj.write(image)
        file_ctr = file_ctr + 1 
        plt.cla()
    writerObj.release()
    
if __name__ == "__main__":
    args = get_args()
    print("Start rendering video for experiment: '{}'".format(args.exp_name))
    np.random.seed(args.seed)
    
    if args.exp_name in ['static_2d_debug', 'static_2d_repeats', 'static_time_repeats', 'bottleneck_repeats', 'dynamic_2d_debug', 'dynamic_2d_fort', 'dynamic_2d_forest']:
        if args.dubins:
            make_video_dubins(args)
        else:
            make_video(args)
    elif args.exp_name in ['static_2d_time_debug', 'static_time_repeats', 'static_2d_time_grid']:
        if args.dubins:
            make_time_video_static_dubins(args)
        else:
            make_time_video_static(args)
    elif args.exp_name in ['dynamic_2d_time_debug', 'dynamic_2d_time_busy']:
        if args.dubins:
            make_time_video_dynamic_dubins(args)
        else:
            make_time_video_dynamic(args)
    else:
        raise NotImplementedError("This experiment is not implemented.")