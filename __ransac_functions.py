from math import asin, acos, cos, sin, pi, atan2
import numpy as np
import array as arr
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, metrics, multioutput
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


def pair_filter(val_ligne):
    result = True
    thresh_score = 1.5
    if (float(val_ligne[4]) < float(thresh_score)):
        result = False
    if (int(val_ligne[5]) == 0):
        result = False
    #print(val_ligne)
    #print(result)
    return result


def get_data_from_file(file_name):
    # Init
    #nb_total = get_line_number(file_name)
    nb_valid = 0
    f = open(file_name, "r")
    Lines = f.readlines()
    idx_ligne = 0
    # Get number of valid pairs
    for line in Lines:
        val_ligne = line.split(" ")
        if (pair_filter(val_ligne)):
            nb_valid += 1
    # Store valid pairs
    valeurs = np.zeros((nb_valid, 6))
    for line in Lines:
        val_ligne = line.split(" ")
        if (pair_filter(val_ligne)):
            for i in range(6):
                valeurs[idx_ligne][i] = float(val_ligne[i])
            idx_ligne += 1  
        
    return valeurs, nb_valid

def filter_matches(valeurs, filter_local_max_0 = 1, ):
    index = []
    for i in range(len(valeurs)):
        if (valeurs[i][5] == 0):
            index.append(i)
    newValeurs = np.delete(valeurs, index)
    return newValeurs

def get_frag_name(txt_name):
    tmp = (txt_name.split('/'))[1]
    tmp = (tmp.split('.'))[0]
    tmp = ((4 - len(tmp))*'0' + tmp) 
    tmp = "images/frag/frag_eroded_"+tmp+".ppm"
    #print(tmp)
    return tmp

def execute_openCV_ransac(x1, x2, y1, y2, printIt=False):
    # Extract
    frag_points  = np.empty((len(x1), 2))
    fresq_points = np.empty_like(frag_points)
    for i in range(len(x1)):
        frag_points[i, 0] = x1[i]
        frag_points[i, 1] = y1[i]
        fresq_points[i, 0] = x2[i]
        fresq_points[i, 1] = y2[i]

    h, mask = cv.findHomography(frag_points,fresq_points,cv.RANSAC)
    #print(h)
    # Print it
    if(printIt):
        yesPoints = fresq_points[np.transpose(mask),:]
        outPoints = fresq_points[np.logical_not(np.transpose(mask)),:]
        plt.scatter(yesPoints[:,0], yesPoints[:,1], marker='+',c='g', linewidths=0)
        plt.scatter(outPoints[:,0], outPoints[:,1], marker='.',c='r',linewidths=0)
        plt.show()
    return h

def execute_ransac(x1, x2, y1, y2, printIt=False):
    # Extract
    frag_points  = np.empty((len(x1), 2))
    fresq_points = np.empty_like(frag_points)
    for i in range(len(x1)):
        frag_points[i, 0] = x1[i]
        frag_points[i, 1] = y1[i]
        fresq_points[i, 0] = x2[i]
        fresq_points[i, 1] = y2[i]
    # Ransac
    sample = np.ones((len(x1)))
    base_estim = linear_model.LinearRegression()
    ransac = linear_model.RANSACRegressor()#None, 5, None, None, None, 250, np.inf, np.inf, np.inf, 1, 'absolute_loss')
    #ransac = multioutput.MultiOutputRegressor(linear_model.Ridge(random_state=123))
    ransac.fit(frag_points, fresq_points, sample)

    # Print it
    if(printIt):
        yesPoints = fresq_points[ransac.inlier_mask_,:]
        outPoints = fresq_points[np.logical_not(ransac.inlier_mask_),:]
        plt.scatter(yesPoints[:,0], yesPoints[:,1], marker='+',c='g', linewidths=0)
        plt.scatter(outPoints[:,0], outPoints[:,1], marker='.',c='r',linewidths=0)
        plt.show()
    return ransac.estimator_

def getDaDxDyFromH(H, thresh=np.inf, verbose=False):
    # Translation
    dx = H[0,2]
    dy = H[1,2]
    # Rotation
    # Matrice de la forme 
    # [[a,-b], 
    #  [b, a]]
    da1 = acos(max(min(H[0,0], 1), -1))
    da2 = acos(max(min(H[1,1], 1), -1))
    da3 = asin(max(min(H[0,1], 1), -1))
    da4 = asin(max(min(-H[1,0], 1), -1))
    tang = atan2(H[0,0], H[1,0])
    avg_cos = (da1+da2)/2
    avg_sin = (da3+da4)/2


    offset = -pi
    if (tang > 0):
        offset = -(pi/2)
    angles_base = [da1, da2, da3, da4]
    angles = [da1+offset, da2+offset, da3, da4]
    da = sum(angles)/4

    if (tang > 0):
        da = max(avg_cos, avg_sin)
    else :
        da = min(avg_cos, avg_sin)

    # Goodmatch
    goodmatch = True
    if ((abs(da1-da2) > thresh) or (abs(da3-da4) > thresh)):
        goodmatch = False
    
    if (verbose):
        print("Verbose Mode :\nMatrix 3x3 H :\n", H)
        print("Rotation angles   (no offset):\n", angles_base)
        print("Rotation angles (with offset):\n", angles)    
        print("Tan angle : ", tang)
        print("Avg acos : ", avg_cos)
        print("Avg asin : ", avg_sin)
        print("dx / dy : " + str(dx) + " / " +str(dy))
        print("result da : " + str(da))
    return dx, dy, da, goodmatch

def copy_image_into_image_Transform(frag, fresque, dx, dy, da):
    ##### INIT
    img_frag = np.asarray(Image.open(frag))
    img_fresque = np.asarray(Image.open(fresque))
    img_fresque2 = img_fresque.copy()

    ##### MODIFY
    hf, wf = dimensions(img_fresque2)
    h,w = dimensions(img_frag)
    for i in range(w):
        progress_bar(i/w, 'Copying image', 0)
        for j in range(h):
            # Calc new point
            a = [[int(i)],[int(j)],[1]]
            H = [[-cos(da), sin(da), dx],[-sin(da), -cos(da), dy],[0,0,1]]
            #H = [[cos(da), sin(da), dx],[-sin(da), cos(da), dy],[0,0,1]]
            fresque_pix = np.dot(H,a)
            # Round and apply
            newx = int(round(abs(fresque_pix[0,0])))
            newy = int(round(abs(fresque_pix[1,0])))
            #print([newx, newy])
            r = (img_frag[j, i, 0])
            g = (img_frag[j, i, 1])
            b = (img_frag[j, i, 2])
            if (not((r == 0) & (g == 0) & (b == 0))):
                #if (i%20 == 0):
                    #print("replacing pixel :")
                    #print(img_fresque2[newy+dy, newx+dx])
                    #print("by -> ", [r,g,b])
                if((newy < hf) & (newx < wf)):
                    if((newy >= 0) & (newx >= 0)):
                        img_fresque2[newy, newx] = [r,g,b]
                #np_fresque[newy+dy, newx+dx] = [r,g,b]
                #print("replacing pixel at ", newy+dy, ", ", newx+dx)
                #print([r,g,b])
    
    ##### SAVE
    print("Saving image : " + fresque)
    img = Image.fromarray(np.uint8(img_fresque2))
    img.save(fresque)
    return

def copy_image_into_image_OpenCV(frag, fresque, H):
    ##### INIT
    img_frag = np.asarray(Image.open(frag))
    img_fresque = np.asarray(Image.open(fresque))
    img_fresque2 = img_fresque.copy()

    ##### MODIFY
    hf, wf = dimensions(img_fresque2)
    h,w = dimensions(img_frag)
    for i in range(w):
        progress_bar(i/w, 'Copying image', 0)
        for j in range(h):
            # Calc new point
            a = [[int(i)],[int(j)],[1]]
            fresque_pix = np.dot(H,a)
            # Round and apply
            newx = int(round(abs(fresque_pix[0,0])))
            newy = int(round(abs(fresque_pix[1,0])))
            #print([newx, newy])
            r = (img_frag[j, i, 0])
            g = (img_frag[j, i, 1])
            b = (img_frag[j, i, 2])
            if (not((r == 0) & (g == 0) & (b == 0))):
                #if (i%20 == 0):
                    #print("replacing pixel :")
                    #print(img_fresque2[newy+dy, newx+dx])
                    #print("by -> ", [r,g,b])
                if((newy < hf) & (newx < wf)):
                    if((newy >= 0) & (newx >= 0)):
                        img_fresque2[newy, newx] = [r,g,b]
                #np_fresque[newy+dy, newx+dx] = [r,g,b]
                #print("replacing pixel at ", newy+dy, ", ", newx+dx)
                #print([r,g,b])
    
    ##### SAVE
    print("Saving image : " + fresque)
    img = Image.fromarray(np.uint8(img_fresque2))
    img.save(fresque)
    #print("done save.")
    return

def copy_image_into_image(frag, source, H, destination=0):
    ##### INIT
    img_frag = np.asarray(Image.open(frag))
    img_fresque = np.asarray(Image.open(source))
    img_fresque2 = img_fresque.copy()
    point = np.empty((1,2))
    point2 = np.empty_like(point)

    ##### MODIFY
    hf, wf = dimensions(img_fresque2)
    h,w = dimensions(img_frag)
    for i in range(w):
        progress_bar(i/w, 'Copying image', 0)
        for j in range(h):
            point[0,0] = i
            point[0,1] = j
            point2 = H.predict(point)
            newx = int(round(point2[0,0])) #int((H2[0,0]*i + H2[0,1]*j)) #+H2[0,2]
            newy = int(round(point2[0,1])) #int((H2[1,0]*i + H2[1,1]*j)) #+H2[1,2]
            r = (img_frag[j, i, 0])
            g = (img_frag[j, i, 1])
            b = (img_frag[j, i, 2])
            if (not((r == 0) & (g == 0) & (b == 0))):
                if((newy < hf) & (newx < wf)):
                    img_fresque2[newy, newx] = [r,g,b]

    ##### SAVE
    img = Image.fromarray(np.uint8(img_fresque2))
    if (destination == 0):
        img.save(source)
    else:
        img.save(destination)
    return

def rectify_H_Regressor(H, verbose=False):
    # Extract info
    Mat_rotation = H.coef_
    Mat_translation = H.intercept_
    Ang_acos1 = acos(max(min(Mat_rotation[0,0], 1), -1))
    Ang_acos2 = acos(max(min(Mat_rotation[1,1], 1), -1))
    Ang_asin1 = asin(max(min(-Mat_rotation[0,1], 1), -1))
    Ang_asin2 = asin(max(min(Mat_rotation[1,0], 1), -1))
    if (verbose):
        print(Mat_rotation)
        print(Mat_translation)
        print("Acos : ", Ang_acos1, ", ", Ang_acos2)
        print("Asin : ", Ang_asin1, ", ", Ang_asin2)

    # Calculate right angle
    angle = Ang_acos1
    if (angle == 0):
        goodmatch = False
    else:
        goodmatch = True

    # Redo Matrix
    H.coef_[0,0] = cos(angle)
    H.coef_[1,1] = cos(angle)
    H.coef_[0,1] = -sin(angle)
    H.coef_[1,0] = sin(angle)
    return goodmatch

def convert_image(source, destination):
    # Get Image
    im = mpimg.imread(source)
    im = im[:,:,:3].copy()

    # Save as normal image
    img = Image.fromarray(im, 'RGB')
    img.save(destination)
    return


def dimensions(image):
    w = int(image[0].size / 3)
    h = int(image.size / (3 * w))
    return h,w

    
def progress_bar(percentage, text='', size=40):
    if(size == 0):
        print(' ' + text + ' -> [' + str(int(percentage*100)) + '%]', end='\r')
        return
    a = int(size*percentage)
    b = size - a
    print(' ' + text + ' ->  [' + a*('+') + b*('-') + ']' + 10*' ', end='\r')
    return


def pixel_set(image, x, y, r=1, g=1, b=1):
    h,w = dimensions(image)
    if ((x <= w) & (y <= h)):
        image[y,x,0] = r
        image[y,x,1] = g
        image[y,x,2] = b        
        return True
    else:
        return False


"""

def filtre_sur_fresque(image, r, g, b):
    h,w = dimensions(image)
    for i in range(w):
        for j in range(h):
            ri = image[j, i, 0]
            gi = image[j, i, 1]
            bi = image[j, i, 2]
            image[j, i] = [ri+r, gi+g, bi+b]

def do_ransac_on_data(x1, x2):
    # Apply ransac algorithm
    ransac = linear_model.RANSACRegressor(None, 2, None, None, None, 100, np.inf, np.inf, np.inf, 0.99, 'absolute_loss')
    ransac.fit(x1, x2)
    # Predict data of estimated models
    x1_ransac = np.arange(x1.min(), x1.max())[:, np.newaxis]
    x2_ransac = ransac.predict(x1_ransac)
    return x1_ransac, x2_ransac, ransac

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