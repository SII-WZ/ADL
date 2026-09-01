
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy
import random

def turn(imag_arry):
    npy_img=[]
    for i in range(imag_arry.shape[0]):
        img_cv = imag_arry[i,]
        img_new = cv.resize(img_cv, (32, 32), interpolation=cv.INTER_CUBIC)
        #   print(img_cv)
        # plt.imshow(img_new, cmap=plt.cm.gray)
        # plt.show()
        npy_img.append((np.array(copy.deepcopy(img_new))))
    return np.array(npy_img)

load_path="your data load_path"

x=np.load(load_path+"speckles.npy")

x=turn(x)
print(x.shape)
np.save(load_path+'x_32', x)

print(np.array(x).shape)

