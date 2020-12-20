import numpy as np
import array as arr
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, datasets


print("Begin Initialization")
fresque = mpimg.imread("images/fresque.ppm")
fresque = fresque[:,:,:3].copy()
#plt.imshow(fresque_base)
#plt.savefig("images/fresque_base.png")
fresque_base = fresque.copy()

w = int(fresque[0].size / 3)
h = int(fresque.size / (3 * w))    
for i in range(w):
    for j in range(h):
        fresque[j, i] = [0,0,0]

#plt.imshow(fresque_base)
#fresque_base.save("images/fresque_empty.png")

# Save as normal image
img = Image.fromarray(fresque, 'RGB')
img.save('images/fresque_empty.png')
img.save('images/fresque_empty_fantomes.png')
#img.show()
img = Image.fromarray(fresque_base, 'RGB')
img.save('images/fresque_copy.png')
print("End Initialization")



def dimensions(image):
    w = int(image[0].size / 3)
    h = int(image.size / (3 * w))
    return h,w
