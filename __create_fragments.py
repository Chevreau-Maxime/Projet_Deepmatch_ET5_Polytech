import sys
import os
from math import asin, acos
import numpy as np
import array as arr
import random
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn import linear_model, multioutput
from __ransac_functions import dimensions, pixel_set, progress_bar



def create_fragments(folder):
    print("Creating fragments from ", folder)
    fresque = open_file(folder + "fresque.png")
    fresque = fragment_file(fresque, 100)
    save_file(fresque, folder + "fresque_fragmentee.png")
    fresque = save_fragments(fresque, folder)
    save_file(fresque, folder + "fresque_videe.png")
    return







def save_fragments(F, folder):
    h,w = dimensions(F)
    done = False
    while(not(done)):
        start = [-1,-1]
        for x in range(w):
            for y in range(h):
                if (F[y,x].any()):
                    start = [x,y]
        if (start[0] == -1):
            done = True
        else:
            save_single_fragment(F, start[0], start[1])
            done = True #DELETE THIS LINE LATER
    return F
       
def save_single_fragment(F, startx, starty):
    h,w = dimensions(F)
    visit = np.zeros((2))
    visit[0] = startx
    visit[1] = starty
    idx = 0
    count = 0
    done = False
    while(not(done)):
        ### Treat visit[idx]
        currentx = int(visit[idx + 0])
        currenty = int(visit[idx + 1])
        print("current pixel : ", [currentx, currenty])
        count = count + 1
        pixel_set(F, currentx, currenty, 0,0,0)
        ### Check neighbours of visit[idx]
        for x in range(-1, 2, 1):
            for y in range(-1, 2, 1):
                newx = int(currentx+x)
                newy = int(currenty+y)
                if ((newx < w) & (newx >= 0)):
                    if ((newy < h) & (newy >= 0)):
                        #print("    looking at : ", [newx, newy])
                        tmp = F[newy, newx]
                        #print(tmp)
                        if ((tmp[0] != 0) or (tmp[1] != 0) or (tmp[2] != 0)):
                            #print("yes")
                            visit = np.append(visit, [newx, newy])
                            #visit = np.concatenate((visit, newx))
                            #visit = np.concatenate((visit, newy))
                            #visit = np.insert(visit, len(visit)-1, [newx, newy])
                            #value = np.zeros_like(visit[0])
                            #value[0] = newx
                            #value[1] = newy
                            #visit = np.vstack((visit, value))
                            #print("ok !")
        #print(visit)
        idx = idx + 2
        #print(idx)
        if (len(visit) == idx*2):
            done = True
    print("Changed " + str(count) + " pixels")
        


    # TODO Implement A* type algorithm to retrieve all pixels in zones
    # np.append (visited, [a,b])

# Lines to cut in fragments
def fragment_file(F, fragment_size):
    h,w = dimensions(F)

    ##### Modify all (0,0,0) pixels to (1,1,1)
    clean_saturated_pixels(F)

    ##### Create array of intersections
    nb_x = int(w / fragment_size)+1
    nb_y = int(h / fragment_size)+1
    inter = create_intersections(nb_x, nb_y, fragment_size, w, h)

    ##### Draw lines between intersections
    for i in range(nb_x-1):
        for j in range(nb_y-1):
            progress_bar(((i*(nb_y-1))+j)/((nb_x-1)*(nb_y-1)), "Drawing lines to fragment", 0)
            ### Horizontal
            draw_line(F, inter[i,j,0], inter[i+1, j, 0], inter[i,j,1], inter[i+1, j, 1]) 
            ### Vertical
            draw_line(F, inter[i,j,0], inter[i, j+1, 0], inter[i,j,1], inter[i, j+1, 1]) 
    print("\nFragmentation finished.")
    return F

def clean_saturated_pixels(F):
    h,w = dimensions(F)
    count = 0
    for x in range(w):
        progress_bar(x/w, "Preparing image for fragmentation", 0)
        for y in range(h):
            if (not(F[y,x].any())):
                pixel_set(F, x, y, 1, 1, 1)
                count = count + 1
    print("\n(changed "+ str(count) +" pixels)")
    return

def create_intersections(nb_x, nb_y, size, w, h, var=0.5):
    inter = np.zeros((nb_x+1, nb_y+1, 2))
    for i in range(nb_x):
        for j in range(nb_y):
            x = int(min(i * size + (random.random()*var*size), w-1))
            y = int(min(j * size + (random.random()*var*size), h-1))
            if (i == 0):
                x = 0
            if (j == 0):
                y = 0
            if (i == nb_x-1):
                x = w-1
            if (j == nb_y-1):
                y = h-1
            inter[i, j] = [x, y]
    return inter

def draw_line(F, xs, xe, ys, ye):
    x = xs
    x2 = xe
    y = ys
    y2 = ye
    done = False
    while (not(done)):
        if ((x == x2) & (y == y2)):
            done = True
        if (abs(x2 - x) > abs(y2 - y)):
            if (x2 > x):
                x = x+1
            else:
                x = x-1
        else:
            if (y2 > y):
                y = y+1
            else:
                y = y-1
        pixel_set(F, int(x), int(y), 0, 0, 0)
    return

def open_file(name):
    img_fresque = np.asarray(Image.open(name))
    img_fresque2 = img_fresque[:,:,:3].copy()
    return img_fresque2 

def save_file(fresque_array, name):
    img = Image.fromarray(np.uint8(fresque_array))
    img.save(name)
    return