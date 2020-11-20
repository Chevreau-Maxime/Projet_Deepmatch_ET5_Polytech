import sys
import os
from math import asin, acos
import numpy as np
import array as arr
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, multioutput
from __create_fragments import create_fragments

print("Loop through all images : ")

#fullpath = (os.getcwd())
root_dir = "./autres_fresques"
#print(os.listdir(root_dir))
for folder in os.listdir(root_dir):
    #os.chdir(fullpath + "\\" + folder)
    os.system("cd " + folder)
    create_fragments()
