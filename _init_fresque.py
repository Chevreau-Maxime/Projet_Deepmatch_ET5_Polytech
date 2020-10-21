import numpy as np
import array as arr
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, datasets


print("Begin Initialization")
fresque_base = mpimg.imread("images/fresque.ppm")
fresque_base = fresque_base[:,:,:3].copy()

plt.imshow(fresque_base)
plt.savefig("images/fresque_base.png")


w = int(fresque_base[0].size / 3)
h = int(fresque_base.size / (3 * w))    
for i in range(w):
    for j in range(h):
        fresque_base[j, i] = [0,0,0]
plt.imshow(fresque_base)
#fresque_base.save("images/fresque_empty.png")

print("End Initialization")



def dimensions(image):
    w = int(image[0].size / 3)
    h = int(image.size / (3 * w))
    return h,w
