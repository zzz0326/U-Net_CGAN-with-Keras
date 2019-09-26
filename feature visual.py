
import numpy as np
import skimage.transform as trans
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

model = load_model('unet_membrane2.hdf5')  # replaced by your model name
# Get all our test images.
image = '0.jpg'
images = cv2.imread('0.jpg')
#cv2.imshow("Image", images)
#cv2.waitKey(0)
# Turn the image into an array.
image_arr = trans.resize(images,(256,256,1))  # 根据载入的训练好的模型的配置，将图像统一尺寸
image_arr = np.expand_dims(image_arr, axis=0)

# 设置可视化的层
layer_1 = K.function([model.layers[0].input], [model.layers[37].output])
#layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
f1 = layer_1([image_arr])[0]
for _ in range(2):
    show_img = f1[:, :, :, _]
    show_img.shape = [256, 256]
    plt.subplot(1, 2, _ + 1)
    plt.subplot(1, 2, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
plt.show()
# conv layer: 299
'''
layer_1 = K.function([model.layers[0].input], [model.layers[299].output])
f1 = layer_1([image_arr])[0]
for _ in range(81):
    show_img = f1[:, :, :, _]
    show_img.shape = [8, 8]
    plt.subplot(9, 9, _ + 1)
    plt.imshow(show_img, cmap='gray')
    plt.axis('off')
plt.show()
'''