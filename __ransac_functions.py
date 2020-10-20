import numpy as np
import array as arr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, datasets

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
    print(tmp)
    return tmp

def do_ransac_on_data(x1, x2):
    # Apply ransac algorithm
    ransac = linear_model.RANSACRegressor()
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
    plt.scatter(x1[inlier_mask], x2[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
    plt.scatter(x1[outlier_mask], x2[outlier_mask], color='gold', marker='.', label='Outliers')
    plt.plot(x1_ransac, x2_ransac, color='cornflowerblue', linewidth=3, label='RANSAC regressor')
    plt.legend(loc='lower right')
    plt.xlabel("Input")
    plt.ylabel("Response")
    #plt.show()




def copy_image_into_image(frag, fresque, x=0, y=0, angle=0):
    img_frag = mpimg.imread(frag) #-> recup l'image en var
    img_frag = img_frag[:,:,:3].copy() #-> on enleve l'alpha si present
    img_fresque = mpimg.imread(fresque)
    img_fresque2 = img_fresque[:,:,:3].copy()

    #filtre_sur_fresque(img_fresque2, 30, 10, 0)
    h,w = dimensions(img_frag)
    for i in range(w):
        progress_bar(i/w, 'Copying image')
        for j in range(h):
            r = img_frag[j, i, 0]
            g = img_frag[j, i, 1]
            b = img_frag[j, i, 2]
            if (not((r == 0) & (g == 0) & (b == 0))):
                img_fresque2[j+y, i+x] = [r,g,b]
            #pixel_set(img_fresque2, i+x, j+y, r, g, b)   
    plt.imshow(img_fresque2)
    plt.savefig("images/fresque_new.png")
    plt.show()
    return

def filtre_sur_fresque(image, r, g, b):
    h,w = dimensions(image)
    for i in range(w):
        for j in range(h):
            ri = image[j, i, 0]
            gi = image[j, i, 1]
            bi = image[j, i, 2]
            image[j, i] = [ri+r, gi+g, bi+b]

def dimensions(image):
    w = int(image[0].size / 3)
    h = int(image.size / (3 * w))
    return h,w

    
def progress_bar(percentage, text='', size=40):
    a = int(size*percentage)
    b = size - a
    print(' ' + text + ' ->  [' + a*('+') + b*('-') + ']' + 10*' ', end='\r')
    return

    
def pixel_set(image, x, y, r=1, g=1, b=1):
    h,w = dimensions(image)
    if ((x < w) & (y < h)):
        image[y,x] = [r, g, b]

