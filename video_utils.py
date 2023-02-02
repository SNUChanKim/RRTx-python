import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import patches

def colorPlot(x, y, c, th, ax):
    if c.size <= 1:
        return
    
    xx = np.array([x,x])
    xx = xx.transpose().reshape(-1)[1:-1]
    yy = np.array([y,y])
    yy = yy.transpose().reshape(-1)[1:-1]
    
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    len = np.sqrt(dx**2 + dy**2)
    
    cc = np.array([c,c])
    cc = cc.transpose().reshape(-1)[1:-1]
    nan_inds = np.where(np.isnan(xx))[0]
    if nan_inds.size == 0:
        start_inds = np.array([0])
        end_inds = np.array([xx.size-1])
    else:
        start_inds = np.insert(nan_inds[:-1] + 1, 0, 0)
        end_inds = nan_inds-1
    
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=np.min(cc[np.where(~np.isnan(cc))]), vmax=np.max(cc[np.where(~np.isnan(cc))]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    for j in range(end_inds.size):
        si = start_inds[j]
        ei = end_inds[j]
        for i in np.arange(si, ei):
            if xx[i] == xx[i+1] and yy[i] == yy[i+1]:
                continue
            interpolate_x = (xx[i+1] - xx[i])/10
            interpolate_y = (yy[i+1] - yy[i])/10
            interpolate_time = (cc[i+1] - cc[i])/10
            for k in range(10):
                colorval = scalarMap.to_rgba((cc[i]+k*interpolate_time))
                ax.plot([xx[i] + k*interpolate_x, xx[i] + (k+1)*interpolate_x], [yy[i] + k*interpolate_y, yy[i] + (k+1)*interpolate_y], c=colorval, linewidth=4)
        # print("*******************8")
            
            
def plotSpikes(pose, radius, theta, d_theta, num, clr, ax):
    th = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    r = np.array([.5, .6, .7, .8, .9, 1, .9, .8, .7, .6, .5])
    
    max_spike = .9
    spike_step = max_spike*2/th.size
    spikes_forward = np.arange(0, 2*max_spike + spike_step, spike_step)
    
    th_r_offset = np.zeros(th.size)
    half_th_size = int(np.ceil(th.size/2))
    th_r_offset[:half_th_size+1] = spikes_forward[:half_th_size+1]
    
    th_r_offset = np.maximum(th_r_offset,  np.flip(th_r_offset))
    if d_theta > 0:
        th = th + th_r_offset
    else:
        th = th - th_r_offset
        
    th = np.tile(th, [num,1]) + np.tile(np.arange(0, num), [th.size, 1]).transpose()
    th = (2*math.pi/num)*th.reshape(-1) + theta
    r = radius*np.tile(r, [num, 1]).reshape(-1)

    x = r*np.cos(th) + pose[0]
    y = r*np.sin(th) + pose[1]
    ax.fill(x, y, color=clr)
    
def plotPackman(pose, radius, theta, phi, clr, ax):

    num_segs = 40
    step = 2*math.pi/num_segs
    th = np.arange(0, 2*math.pi + step, step)
    
    x = radius*np.cos(th)
    y = radius*np.sin(th)
    
    x[np.where(np.logical_or(th < phi, th > 2*math.pi - phi))] = -0.2*radius
    y[np.where(np.logical_or(th < phi, th > 2*math.pi - phi))] = 0
    
    xy = np.array([x,y])
    
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    xy_hat = np.matmul(R, xy)
    
    ax.fill(xy_hat[0,:] + pose[0], xy_hat[1,:] + pose[1] , color=clr)
    
def plotVehicle(f, a, b, c, clr, sr, clrB, ax):

    x = np.array([0, .2, .2, 1, 1, .2, .2,  .5, .5])
    y = np.array([1, .8, .4, .4, -.1, -.1, -.7, -.7, -1])

    x = c*np.concatenate((x, -np.flip(x)), axis=0)
    y = c*np.concatenate((y, np.flip(y)), axis=0)
    
    xy = np.array([x,y])
    
    if not np.array_equal(a, b):
        theta = -math.atan2(b[0]-a[0], b[1]-a[1])
    else:
        theta = 0
    
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    xy_hat = np.matmul(R, xy)
    
    ax.fill(xy_hat[0,:] + a[0], xy_hat[1,:] + a[1] , clr)
    
    if not np.isnan(sr):
        steps = 20
        phis = np.arange(0, 2*math.pi + 2*math.pi/steps, 2*math.pi/steps)
    
        xs = sr*np.cos(phis)
        ys = sr*np.sin(phis)
    
        ax.plot(xs + a[0], ys + a[1] , 'k')  
        ax.plot(xs + a[0], ys + a[1] , clrB)

def plotVehicleTheta(f, a, theta, c, clr, sr, clrB, ax):
    x = np.array([0, .2, .2,  1,  1, .2, .2, .5, .5])
    y = np.array([1, .8, .4, .4, -.1, -.1, -.7, -.7, -1])
    
    x = c*np.concatenate((x, -np.flip(x)), axis=0)
    y = c*np.concatenate((y, np.flip(y)), axis=0)
    
    xy = np.array([x, y])
    
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    xy_hat = np.matmul(R, xy)
    
    ax.fill(xy_hat[0,:] + a[0], xy_hat[1,:] + a[1] , clr)

    if not np.isnan(sr):
        steps = 20
        phis = np.arange(0, 2*math.pi + 2*math.pi/steps, 2*math.pi/steps)
    
        xs = sr*np.cos(phis)
        ys = sr*np.sin(phis)
    
        ax.plot(xs + a[0], ys + a[1] , 'k')  
        ax.plot(xs + a[0], ys + a[1] , clrB)