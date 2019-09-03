from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from model1 import *
import cv2

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def LoadRealImage(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.jpg"%i),as_gray = as_gray)
        img = img / 255.0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)

        yield (img,[1])

def labelVisualize(num_class,color_dict,img):
    img = img[:, :, 0] if len(img.shape) == 3 else img

    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255.0

def LoadFakeImage(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.jpg"%i),as_gray = as_gray)
        img = img / 255.0
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)

        yield img

def PreFakeImage(test_path,G_model,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    #需要的对象是(256,256,1)
        result = G_model.predict_generator( LoadFakeImage(test_path,num_image,target_size,as_gray = True),30)
        #生成对象
        for i, item in enumerate(result):
            img = labelVisualize(2, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
            img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)

            #io.imshow(item)
            #plt.show()
            yield (img,[0])