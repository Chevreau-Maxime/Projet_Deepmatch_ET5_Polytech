import numpy as np
import cv2 as cv
import __ransac_functions as rf
import sys



"""
def Affine(u1,v1,x1,y1,u2,v2,x2,y2,u3,v3,x3,y3,u4,v4,x4,y4): #(x,y) origin => (u,v)
    Mt=np.array(
        [x1,y1,1,0,0,0,-u1*x1,-u1*y1,0,0,0,x1,y1,1,-v1*x1,-v1*y1,
        x2,y2,1,0,0,0,-u2*x2,-u2*y2,0,0,0,x2,y2,1,-v2*x2,-v2*y2
        ,x3,y3,1,0,0,0,-u3*x3,-u3*y3,0,0,0,x3,y3,1,-v3*x3,-v3*y3
        ,x4,y4,1,0,0,0,-u4*x4,-u4*y4,0,0,0,x4,y4,1,-v4*x4,-v4*y4])
    Mt.resize(8,8)
    k=np.array([u1,v1,u2,v2,u3,v3,u4,v4]).T
    M=np.copy(Mt.T)
    MMt=np.dot(M,Mt)
    print (np.linalg.det(MMt))
    h=np.linalg.inv(MMt)
    h1=np.dot(h,M)
    h2=np.dot(h1,k)
    return h2"""

# READ DATA FROM TXT FILE IN PARAM
#param1 = sys.argv[1] 
param = "resultats/1.txt"
#param2 = sys.argv[2] #"images/frag/frag_eroded_0001.ppm"
nb_paires = rf.get_line_number(param)
nb_infos = 6
valeurs_txt = rf.get_data_from_file(param)
#print("Paired points amount : " + str(nb_paires))
#print(valeurs_txt)

x1 = np.zeros((nb_paires, 1))
y1 = np.zeros((nb_paires, 1))
x2 = np.zeros((nb_paires, 1))
y2 = np.zeros((nb_paires, 1))
for i in range(nb_paires):
	x1[i] = [i][0]
	y1[i] = valeurs_txt[i][1]
	x2[i] = valeurs_txt[i][2]
	y2[i] = valeurs_txt[i][3]
points1 = np.array([x1,y1]).T
points2 = np.array([x2,y2]).T
h, mask = cv.findHomography(points1,points2,cv.RANSAC)

   
print(h)