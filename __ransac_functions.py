import numpy as np
import array as arr
from matplotlib import pyplot as plt
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
    plt.show()
