import sys
import numpy as np
import array as arr
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import __ransac_functions as r_f
import __ransac_homography as r_h


#print("BEGIN TREATMENT")
#######################################################
#  READ DATA FROM TXT FILE IN PARAM
#######################################################
#print("READ DATA")
if (len(sys.argv) >= 1):
	param1 = sys.argv[1]
else:
	param1 = "resultats2/1.txt"
nb_paires = r_f.get_line_number(param1)

"""
if (nb_paires < 20):
	print("Paires insuffisantes ("+str(nb_paires)+") pour l'image ", param1)
	quit()
else:
	quit()
"""

nb_infos = 6
valeurs_txt = r_f.get_data_from_file(param1)
#print("Paired points amount : " + str(nb_paires))
#print(valeurs_txt)

#valeurs_txt = r_f.filter_matches(valeurs_txt) #filter matches
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


#######################################################
#  METHOD 1 : HOMOGRAPHY
####################################################### 
#H = r_f.homography(x1, x2, y1, y2)
"""
# GET RANSAC DATA
# -> x
tmp = r_f.do_ransac_on_data(x1, x2)
x1_ransac = tmp[0]
x2_ransac = tmp[1]
x_data_ransac = tmp[2]
# -> y
tmp = r_f.do_ransac_on_data(y1, y2)
y1_ransac = tmp[0]
y2_ransac = tmp[1]
y_data_ransac = tmp[2]

# CALCULATE TRANSFORMATION FOR FRAGMENT
#trans_x_slope, trans_x_inter = r_h.get_transformation(x1_ransac, x2_ransac, x_data_ransac)
#trans_y_slope, trans_y_inter = r_h.get_transformation(y1_ransac, y2_ransac, y_data_ransac)
#transformation = [trans_x_slope, trans_x_inter, trans_y_slope, trans_y_inter]
#dx, dy, da = r_h.get_transformation(x1, x2, y1, y2, x_data_ransac, y_data_ransac)
"""

#######################################################
#  METHOD 2 : RANSAC
#######################################################
print("CALCULATE RANSAC")
H = r_f.execute_ransac(x1, x2, y1, y2)
#print(H)

print("COPY FRAGMENT")
dx = 0
dy = 0
da = 0
# COPY FRAGMENT INTO
frag_ppm = r_f.get_frag_name(param1)
frag_png = "images/frag_tmp.png"
r_f.convert_image(frag_ppm, frag_png)
r_f.copy_image_into_image(frag_png, "images/fresque_copy.png", dx, dy, da, H)
r_f.copy_image_into_image(frag_png, "images/fresque_empty.png", dx, dy, da, H)

# DISPLAY RANSAC PAIRS
#plt.subplot(121)
#r_f.print_ransac(x1, x2, x1_ransac, x2_ransac, x_data_ransac)
#plt.subplot(122)
#r_f.print_ransac(y1, y2, y1_ransac, y2_ransac, y_data_ransac)
#plt.show()






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