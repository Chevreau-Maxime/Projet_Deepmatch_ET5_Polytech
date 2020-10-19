import sys
import numpy as np
import array as arr
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import __ransac_functions as r_f


# READ DATA FROM TXT FILE IN PARAM
nb_paires = r_f.get_line_number(sys.argv[1])
nb_infos = 6
valeurs_txt = r_f.get_data_from_file(sys.argv[1])
#print("Paired points amount : " + str(nb_paires))
#print(valeurs_txt)

x1 = np.zeros((nb_paires, 1))
y1 = np.zeros((nb_paires, 1))
x2 = np.zeros((nb_paires, 1))
y2 = np.zeros((nb_paires, 1))
for i in range(nb_paires):
	x1[i] = valeurs_txt[i][0]
	y1[i] = valeurs_txt[i][1]
	x2[i] = valeurs_txt[i][2]
	y2[i] = valeurs_txt[i][3]


# GET RANSAC DATA
tmp = r_f.do_ransac_on_data(x1, x2)
x1_ransac = tmp[0]
x2_ransac = tmp[1]
x_data_ransac = tmp[2]

tmp = r_f.do_ransac_on_data(y1, y2)
y1_ransac = tmp[0]
y2_ransac = tmp[1]
y_data_ransac = tmp[2]

# DISPLAY
plt.subplot(121)
r_f.print_ransac(x1, x2, x1_ransac, x2_ransac, x_data_ransac)
plt.subplot(122)
r_f.print_ransac(y1, y2, y1_ransac, y2_ransac, y_data_ransac)
plt.show()




""" OLD CODE


n_samples = 50
n_outliers = 5
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)


# Add outlier data
# np.random.seed(0)
# X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
# y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(x1, x2)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_x1 = np.arange(x1.min(), x1.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_x1)
lw=2
plt.scatter(x1[inlier_mask], x2[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(x1[outlier_mask], x2[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_x1, line_y_ransac, color='cornflowerblue', linewidth=lw, label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
"""