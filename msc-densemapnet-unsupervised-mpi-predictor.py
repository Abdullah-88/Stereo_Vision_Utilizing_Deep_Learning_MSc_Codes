# Model predictoins

import keras
import os , shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras .preprocessing import image
import matplotlib.image as mpimg
from PIL import Image



# Data source

base_dir = "StereoDatampi"
#base_dir = "StereoDatakitti"

train_left_dir = os.path.join(base_dir, 'train_left')
train_right_dir = os.path.join(base_dir, 'train_right')
train_target_dir = os.path.join(base_dir, 'train_target')
validation_left_dir = os.path.join(base_dir, 'validation_left')
validation_right_dir = os.path.join(base_dir, 'validation_right')
validation_target_dir = os.path.join(base_dir, 'validation_target')
test_left_dir = os.path.join(base_dir, 'test_left')
test_right_dir = os.path.join(base_dir, 'test_right')

#with tf.device('/gpu:0'):


# Data generator
train_left_datagen = ImageDataGenerator(rescale=1. / 255)
train_right_datagen = ImageDataGenerator(rescale=1. / 255)
train_target_datagen = ImageDataGenerator(rescale=1. / 255)
validation_left_datagen = ImageDataGenerator(rescale=1. / 255)
validation_right_datagen = ImageDataGenerator(rescale=1. / 255)
validation_target_datagen = ImageDataGenerator(rescale=1. / 255)
test_left_datagen = ImageDataGenerator(rescale=1. / 255)
test_right_datagen = ImageDataGenerator(rescale=1. / 255)


def multi_test_generator(test_left_datagen, test_right_datagen):
    test_left_generator = test_left_datagen.flow_from_directory(test_left_dir, target_size=(672, 672), batch_size=1,
                                                                class_mode=None)
    test_right_generator = test_right_datagen.flow_from_directory(test_right_dir, target_size=(672, 672), batch_size=1,
                                                                  class_mode=None)
    while True:
        x = test_left_generator.__next__()
        y = test_right_generator.__next__()
        yield x


dropout = 0.2

input_img_left = Input(shape=(672, 672, 3))
input_img_right = Input(shape=(672, 672, 3))






xin = layers.SeparableConv2D(filters=128, kernel_size=5, padding='same')(input_img_left)

# image reduced by 8
x8 = layers.MaxPooling2D(8)(xin)
x8 = layers.BatchNormalization()(x8)
x8 = layers.Activation('relu', name='downsampled_stereo')(x8)

dilation_rate = 1
y = x8
# correspondence network
# parallel cnn at increasing dilation rate
for i in range(4):
    a = layers.SeparableConv2D(filters=128,
               kernel_size=5,
               padding='same',
               dilation_rate=dilation_rate)(x8)
    a = layers.Dropout(dropout)(a)
    y = keras.layers.concatenate([a, y])
    dilation_rate += 1

dilation_rate = 1
x = layers.MaxPooling2D(8)(xin)
# disparity network
# dense interconnection inspired by DenseNet
for i in range(4):
    x = keras.layers.concatenate([x, y])
    y = layers.BatchNormalization()(x)
    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(filters=128,
               kernel_size=1,
               padding='same')(y)

    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(filters=64,
               kernel_size=5,
               padding='same',
               dilation_rate=dilation_rate)(y)
    y = layers.Dropout(dropout)(y)
    dilation_rate += 1

# disparity estimate scaled back to original image size
x = keras.layers.concatenate([x, y], name='upsampled_disparity')
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.SeparableConv2D(filters=128, kernel_size=1, padding='same')(x)
x = layers.UpSampling2D(8)(x)

#x = layers.ZeroPadding2D(padding=(2, 0))(x)


y = layers.BatchNormalization()(x)
y = layers.Activation('relu')(y)
y = layers.SeparableConv2D(filters=64, kernel_size=5, padding='same')(y)

x = layers.concatenate([x, y])
y = layers.BatchNormalization()(x)
y = layers.Activation('relu')(y)
y = layers.SeparableConv2D(filters=3, kernel_size=9, padding='same')(y)


yout = layers.Activation('sigmoid', name='disparity_output')(y)

# densemapnet model
densemap_model = Model(input_img_left, yout)




densemap_model.load_weights('Training_Models/model_densemapnet_unsupervised_mpi.h5')

# img_decoded = decoder_model(input_img_left)


# Model predicting


densemap_model.summary()

preds = densemap_model.predict_generator(multi_test_generator(test_left_datagen, test_right_datagen), steps=150,
                                        verbose=0)

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
plt.show()