from math import asin, acos
import numpy as np
import array as arr
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, multioutput
import cv2 as cv

"""
def homography(x1, x2, y1, y2):
    points1 = np.array([x1,y1]).T
    points2 = np.array([x2,y2]).T
    h, mask = cv.findHomography(points1,points2,cv.RANSAC)
    teta1 = acos(h[0,0])
    teta2 = asin(h[0,1])
    teta3 = asin(-h[1,0])
    teta4 = acos(h[1,1])
    print(teta1)
    print(teta2)
    print(teta3)
    print(teta4)
    print(h)
    return h
"""
def get_line_number(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_data_from_file(file_name):
    # Init np array
    valeurs = np.zeros((get_line_number(file_name), 6))
    idx_ligne = 0
    f = open(file_name, "r")
    Lines = f.readlines()
    # Store lines
    for line in Lines:
        val_ligne = line.split(" ")
        for i in range(6):
            valeurs[idx_ligne][i] = float(val_ligne[i])
        idx_ligne += 1
    return valeurs

def get_frag_name(txt_name):
    tmp = (txt_name.split('/'))[1]
    tmp = (tmp.split('.'))[0]
    tmp = ((4 - len(tmp))*'0' + tmp) 
    tmp = "images/frag/frag_eroded_"+tmp+".ppm"
    #print(tmp)
    return tmp

def execute_ransac(x1, x2, y1, y2):
    # Extract
    frag_points  = np.empty((len(x1), 2))
    fresq_points = np.empty_like(frag_points)
    for i in range(len(x1)):
        frag_points[i, 0] = x1[i]
        frag_points[i, 1] = y1[i]
        fresq_points[i, 0] = x2[i]
        fresq_points[i, 1] = y2[i]
    # Ransac
    ransac = linear_model.RANSACRegressor()#None, 2, None, None, None, 100, np.inf, np.inf, np.inf, 0.99, 'absolute_loss')
    ransac.fit(frag_points, fresq_points)
    # Print it
    #plt.plot(fresq_points[:, 0], fresq_points[:, 1], '+', linewidth=0) 
    #plt.show()
    #plt.plot(fresq_points[0], fresq_points[1])
    return ransac.estimator_
"""
def do_ransac_on_data(x1, x2):
    # Apply ransac algorithm
    ransac = linear_model.RANSACRegressor(None, 2, None, None, None, 100, np.inf, np.inf, np.inf, 0.99, 'absolute_loss')
    ransac.fit(x1, x2)
    # Predict data of estimated models
    x1_ransac = np.arange(x1.min(), x1.max())[:, np.newaxis]
    x2_ransac = ransac.predict(x1_ransac)
    return x1_ransac, x2_ransac, ransac
"""
"""
def print_ransac(x1, x2, x1_ransac, x2_ransac, ransac):
    # Create Masks 
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    # Plot
    plt.scatter(x1[inlier_mask], x2[inlier_mask], color='green', marker='+', label='Inliers')
    plt.scatter(x1[outlier_mask], x2[outlier_mask], color='red', marker='x', label='Outliers')
    plt.plot(x1_ransac, x2_ransac, color='blue', linewidth=1, label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
"""

def copy_image_into_image(frag, fresque, dx, dy, da, H):
    ##### INIT
    """
    img_frag = mpimg.imread(frag)
    img_frag = img_frag[:,:,:3].copy() # pour enlever l'alpha channel
    img_fresque = mpimg.imread(fresque)
    img_fresque2 = img_fresque[:,:,:3].copy()
    """

    img_frag = np.asarray(Image.open(frag))
    img_fresque = np.asarray(Image.open(fresque))
    img_fresque2 = img_fresque.copy()
    #print(img_fresque2.flags)
    
    """
    np_fresque = np.zeros_like(img_fresque2)
    hf,wf = dimensions(img_fresque2)
    for i in range(wf):
        for j in range(hf):
            np_fresque[j,i] = img_fresque2[j,i]
    """
    ##### MODIFY
    hf, wf = dimensions(img_fresque2)
    h,w = dimensions(img_frag)
    for i in range(w):
        progress_bar(i/w, 'Copying image')
        for j in range(h):
            point = np.empty((1,2))
            point[0,0] = i
            point[0,1] = j
            point2 = np.empty_like(point)
            point2 = H.predict(point)
            newx = int(point2[0,0]) #int((H2[0,0]*i + H2[0,1]*j)) #+H2[0,2]
            newy = int(point2[0,1]) #int((H2[1,0]*i + H2[1,1]*j)) #+H2[1,2]
            
            r = (img_frag[j, i, 0])
            g = (img_frag[j, i, 1])
            b = (img_frag[j, i, 2])
            if (not((r == 0) & (g == 0) & (b == 0))):
                #if (i%20 == 0):
                    #print("replacing pixel :")
                    #print(img_fresque2[newy+dy, newx+dx])
                    #print("by -> ", [r,g,b])
                if((newy+dy < hf) & (newx+dx < wf)):
                    img_fresque2[newy+dy, newx+dx] = [r,g,b]
                #np_fresque[newy+dy, newx+dx] = [r,g,b]
                #print("replacing pixel at ", newy+dy, ", ", newx+dx)
                #print([r,g,b])
    
    ##### SAVE
    #print(img_fresque2)
    #img = Image.fromarray(np.uint8(img_fresque2))

    plt.imshow(img_fresque2)
    plt.savefig("images/fresque_new.png")
    print("Saving image : " + fresque)
    img = Image.fromarray(np.uint8(img_fresque2))
    img.save(fresque)
    #print("done save.")
    return

def convert_image(source, destination):
    # Get Image
    im = mpimg.imread(source)
    im = im[:,:,:3].copy()

    #w,h = dimensions(im)
    #for i in range(w):
    #    for j in range(h):
    #        im[j, i] = [0,0,0]

    # Save as normal image
    img = Image.fromarray(im, 'RGB')
    img.save(destination)
    return

"""
def filtre_sur_fresque(image, r, g, b):
    h,w = dimensions(image)
    for i in range(w):
        for j in range(h):
            ri = image[j, i, 0]
            gi = image[j, i, 1]
            bi = image[j, i, 2]
            image[j, i] = [ri+r, gi+g, bi+b]
"""
def dimensions(image):
    w = int(image[0].size / 3)
    h = int(image.size / (3 * w))
    return h,w

    
def progress_bar(percentage, text='', size=40):
    a = int(size*percentage)
    b = size - a
    print(' ' + text + ' ->  [' + a*('+') + b*('-') + ']' + 10*' ', end='\r')
    return

"""    
def pixel_set(image, x, y, r=1, g=1, b=1):
    h,w = dimensions(image)
    if ((x < w) & (y < h)):
        image[y,x] = [r, g, b]

"""