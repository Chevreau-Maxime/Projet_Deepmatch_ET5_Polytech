import sys
import numpy as np
import array as arr
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import __ransac_functions as r_f
import __test_manual_ransac as r_h


#######################################################
#  READ DATA FROM TXT FILE IN PARAM
#######################################################

# CONSTANTES D'EXECUTION
useOpenCV = False
Picasso = False # False pour des "fantomes", True pour un "picasso"

# Debug Mode :
if (len(sys.argv) >= 2):
	param1 = sys.argv[1]
else:
	param1 = "resultats3/1.txt"


print("\n--- Fragment "+str(param1)+" ---")
print("- Extraction...")
nb_total = r_f.get_line_number(param1)
valeurs_txt, nb_paires = r_f.get_data_from_file(param1)
print("valid pairs : " + str(nb_paires) + " / " + str(nb_total))

# Filtre Nb de paires
if (nb_paires < 50):
	print("Paires insuffisantes ("+str(nb_paires)+") pour l'image ", param1)
	#quit()

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
print("- Calculate Ransac...")
if (useOpenCV):
	H = r_f.execute_openCV_ransac(x1, x2, y1, y2, False)
else:
	H = r_f.execute_ransac(x1, x2, y1, y2, False)
print("ok")


#######################################################
#  COPY FRAGMENT
#######################################################

print("- Copying Image...")
frag_ppm = r_f.get_frag_name(param1)
frag_png = "images/frag_tmp.png"
r_f.convert_image(frag_ppm, frag_png)

if (useOpenCV):
	if (Picasso):
		dx,dy,da,goodmatch = r_f.getDaDxDyFromH(H, 0.25, False)
		r_f.copy_image_into_image_Transform(frag_png, "images/fresque_empty.png", dx, dy, da)
		r_f.copy_image_into_image_Transform(frag_png, "images/fresque_copy.png", dx, dy, da)
	else:
		r_f.copy_image_into_image_OpenCV(frag_png, "images/fresque_empty.png", H)
		r_f.copy_image_into_image_OpenCV(frag_png, "images/fresque_copy.png", H)
else:
	if (Picasso):
		r_f.rectify_H_Regressor(H)
	r_f.copy_image_into_image(frag_png, "images/fresque_copy.png", H)
	r_f.copy_image_into_image(frag_png, "images/fresque_empty.png",H)
print("ok")