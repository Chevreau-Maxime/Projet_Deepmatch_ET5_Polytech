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

root_dir = "autres_fresques/"
for folder in os.listdir(root_dir):
    path = root_dir + str(folder) + "/"
    print(path)
    create_fragments(path)