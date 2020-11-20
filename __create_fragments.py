import sys
import os
from math import asin, acos
import numpy as np
import array as arr
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, multioutput




def create_fragments(folder = "./"):
    print("Creating fragments from ", os.getcwd())
    fresque = open_file(folder + "fresque.png")
    fresque = fragment_file(fresque)
    save_file(fresque, folder + "fresque_fragmentee.png")
    return


def fragment_file(F):
    return F

def open_file(name):
    img_fresque = np.asarray(Image.open(name))
    img_fresque2 = img_fresque.copy()
    return img_fresque2 

def save_file(fresque_array, name):
    img = Image.fromarray(np.uint8(fresque_array))
    img.save(name)
    return