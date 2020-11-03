import numpy as np
import array as arr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, datasets


def scan_through(x, y):
    precision = 100
    amplitude = (max(x)-min(x))/4
    current_consensus = 0
    max_consensus = 0
    max_index = 0
    for i in range(precision):
        current_consensus = 0
        dy = i* ((max(y)-min(y)-(4*amplitude))/precision) + (min(y)+2*amplitude)
        for j in range(len(x)):
            if ((y[j] < dy+amplitude) & (y[j] > dy-amplitude)):
                current_consensus = current_consensus + 1
        if(current_consensus > max_consensus):
            max_index = i
            max_consensus = current_consensus
            res = dy
    print("best consensus : " + str(max_consensus) + " at idx " + str(max_index) + " value is (" + str(res)+")")
    #res = max_index * ((max(y)-min(y))/precision) + min(y)
    return int(res)


def get_transformation(x_in, x_out, y_in, y_out, x_ransac_info, y_ransac_info):

    dx = dy = da = 0
    dx = scan_through(x_in, x_out)
    dy = scan_through(y_in, y_out)

    #print("coord ("+str(dx)+","+str(dy)+")")
    
    # looking for a y = a*x + b with a = 0 (horizontal)
    # go through y coordinate for both x and y of transformation
    
    
    
    
    """x_mask = x_ransac_info.inlier_mask_
    y_mask = y_ransac_info.inlier_mask_
    x1 = x2 = np.zeros(x_in.size)
    y1 = y2 = np.zeros(y_in.size)
    
    for i in range(x_in.size):
        x1[i] = x_in[i]
        x2[i] = x_out[i]
    for i in range(y_in.size):
        y1[i] = y_in[i]
        y2[i] = y_out[i]

    x1 = x1[x_mask]
    x2 = x2[x_mask]
    y1 = y1[y_mask]
    y2 = y2[y_mask]

    dx = int(sum(x2)/len(x2))
    dy = int(sum(y2)/len(y2))
    print("coord ("+str(dx)+","+str(dy)+")")
    #slope, intercept = np.polyfit(x1, x2, 1)
    #print("slope : " + str(slope))
    #print("inter : " + str(intercept))
    dx = dy = 0
    """    
    return dx, dy, da