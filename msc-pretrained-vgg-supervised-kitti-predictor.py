import keras
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import vgg16
import matplotlib.image as mpimg
from PIL import Image

# Model predictoins


# Data source

base_dir = "StereoDatakitti"
train_left_dir = os.path.join(base_dir, 'train_left')
train_right_dir = os.path.join(base_dir, 'train_right')
train_target_dir = os.path.join(base_dir, 'train_target')
validation_left_dir = os.path.join(base_dir, 'validation_left')
validation_right_dir = os.path.join(base_dir, 'validation_right')
validation_target_dir = os.path.join(base_dir, 'validation_target')
test_left_dir = os.path.join(base_dir, 'test_left')
test_right_dir = os.path.join(base_dir, 'test_right')

# Data Generator
test_left_datagen = ImageDataGenerator(rescale=1. / 255)
test_right_datagen = ImageDataGenerator(rescale=1. / 255)


def multi_test_generator(test_left_datagen, test_right_datagen):
    test_left_generator = test_left_datagen.flow_from_directory(test_left_dir, target_size=(750, 750), batch_size=1,
                                                                class_mode=None)
    test_right_generator = test_right_datagen.flow_from_directory(test_right_dir, target_size=(750, 750), batch_size=1,
                                                                  class_mode=None)
    while True:
        x = test_left_generator.__next__()
        y = test_right_generator.__next__()
        yield [x, y]


# Model loading and predicting

model = models.load_model('Training_Models/model_vgg16_supervised_kitti.h5')
model.summary()
preds = model.predict_generator(multi_test_generator(test_left_datagen, test_right_datagen), steps=200, verbose=0)

# image display

for img in range(len(preds)):

    plt.figure()
    plt.title("predicted output of image {} ".format(img))

    out = preds[img]
    minim = out.min()
    maxim = out.max()
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=minim,vmax=maxim)
    out = norm(out)
    out = out * 255
    out = out.astype('uint8')
    plt.grid(False)
    import cv2

    #out = cv2.applyColorMap(out, cv2.COLORMAP_HOT)
    plt.imshow(out)

#plt.imshow((out.reshape(624, 624, 3)) * norm_coeff / 255)

plt.show()



