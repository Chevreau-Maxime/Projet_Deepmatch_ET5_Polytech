import sys
import numpy as np
import array as arr
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
import __ransac_functions as r_f
import __ransac_homography as r_h

lastsource = "images/fresque_copy.png"
for index in range(1, 315):
    txtname = "resultats3/"+str(index)+".txt"
    print("Now doing : ", txtname, "...")

    # Extraction : 
    nb_total = r_f.get_line_number(txtname)
    valeurs_txt, nb_paires = r_f.get_data_from_file(txtname)

    x1 = np.zeros((nb_paires, 1))
    y1 = np.zeros((nb_paires, 1))
    x2 = np.zeros((nb_paires, 1))
    y2 = np.zeros((nb_paires, 1))
    for i in range(nb_paires):
        x1[i] = valeurs_txt[i][0]
        y1[i] = valeurs_txt[i][1]
        x2[i] = valeurs_txt[i][2]
        y2[i] = valeurs_txt[i][3]

    # Ransac :
    H = r_f.execute_ransac(x1, x2, y1, y2)

    # Copy :
    nextsource = "images/iterations/"+str(index)+".png"
    frag_ppm = r_f.get_frag_name(txtname)
    frag_png = "images/frag_tmp.png"
    r_f.convert_image(frag_ppm, frag_png)
    goodmatch = r_f.rectify_H_Regressor(H) # to hopefully correct scattered fragments...
    if(goodmatch):
        r_f.copy_image_into_image(frag_png, lastsource, H, nextsource)
    lastsource = nextsource

    print("\nok.")
    