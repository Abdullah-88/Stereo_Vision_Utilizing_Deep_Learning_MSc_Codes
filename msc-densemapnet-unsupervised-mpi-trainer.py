# Depth Variational Auto Encoders Continued Training


import keras
import os, shutil
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import Input
from keras import Model
from keras import regularizers
from keras import backend as K

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.image as mpimg
from PIL import Image

# Data source

base_dir = "StereoDatampi"
train_left_dir = os.path.join(base_dir, 'train_left/image_2')
train_right_dir = os.path.join(base_dir, 'train_right/image_3')
# train_target_dir = os.path.join(base_dir,'train_target')
# validation_left_dir = os.path.join(base_dir,'validation_left')
# validation_right_dir = os.path.join(base_dir,'validation_right')
# validation_target_dir = os.path.join(base_dir,'validation_target')
# test_left_dir = os.path.join(base_dir,'test_left')
# test_right_dir = os.path.join(base_dir,'test_right')


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

img_decoded = densemap_model(input_img_left)


# Layer for VAE loss

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, input_img_left, input_img_right, img_decoded):
        baseline = 0.54
        foc = 0.035
        input_img_left = K.flatten(input_img_left)
        input_img_right = K.flatten(input_img_right)
        img_decoded = K.flatten(img_decoded)

        geometric_loss = K.sum(K.square(K.abs(input_img_left - (input_img_right + ((foc * baseline) / img_decoded)))),
                               axis=-1)

        return geometric_loss

    def call(self, inputs):
        left = inputs[0]
        right = inputs[1]
        dec = inputs[2]
        loss = self.vae_loss(left, right, dec)
        self.add_loss(loss, inputs=inputs)

        return [left, right]


geo_match = CustomVariationalLayer()([input_img_left, input_img_right, img_decoded])

# Training VAE


vae = Model([input_img_left, input_img_right], geo_match)

vae.compile(optimizer='adam', loss=None)

vae.summary()

# model training


epochs = 1000

for e in range(epochs):

    print('epoch:', e + 1)

    path_list = os.listdir(train_left_dir)
    random.shuffle(path_list)


    for image_name in path_list:


        left_image_path = os.path.join(train_left_dir, image_name)
        right_image_path = os.path.join(train_right_dir, image_name)

        left_img = image.load_img(left_image_path, target_size=(672, 672))
        right_img = image.load_img(right_image_path, target_size=(672, 672))

        left = image.img_to_array(left_img)
        left = np.expand_dims(left, axis=0)
        left = left / 255.

        right = image.img_to_array(right_img)
        right = np.expand_dims(right, axis=0)
        right = right / 255.

        history = vae.fit([left, right], y=0, verbose=0)

    densemap_model.save_weights('Training_Models/model_densemapnet_unsupervised_mpi.h5')


