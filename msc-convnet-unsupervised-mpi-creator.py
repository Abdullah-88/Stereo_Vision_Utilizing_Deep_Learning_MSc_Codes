# Depth Variational Auto Encoders


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




input_img_left = Input(shape=(672, 672, 3))
input_img_right = Input(shape=(672, 672, 3))

# build network encoder

conv_first = layers.SeparableConv2D(64, (7, 7), padding='same')(input_img_left)
batchnorm_first = layers.BatchNormalization()(conv_first)
activation_first = layers.Activation('relu')(batchnorm_first)
maxpooling_first = layers.MaxPooling2D((2, 2))(activation_first)

conv_second = layers.SeparableConv2D(128, (7, 7), padding='same')(maxpooling_first)
batchnorm_second = layers.BatchNormalization()(conv_second)
activation_second = layers.Activation('relu')(batchnorm_second)
maxpooling_second = layers.MaxPooling2D((2, 2))(activation_second)

conv_third = layers.SeparableConv2D(256, (5, 5), padding='same')(maxpooling_second)
batchnorm_third = layers.BatchNormalization()(conv_third)
activation_third = layers.Activation('relu')(batchnorm_third)
maxpooling_third = layers.MaxPooling2D((2, 2))(activation_third)

conv_fourth = layers.SeparableConv2D(512, (3, 3), padding='same')(maxpooling_third)
batchnorm_fourth = layers.BatchNormalization()(conv_fourth)
activation_fourth = layers.Activation('relu')(batchnorm_fourth)
maxpooling_fourth = layers.MaxPooling2D((2, 2))(activation_fourth)

# build network Decoder


decoder_conv_first = layers.SeparableConv2D(512, (3, 3), padding='same')(maxpooling_fourth)
decoder_batchnorm_first = layers.BatchNormalization()(decoder_conv_first)
decoder_activation_first = layers.Activation('relu')(decoder_batchnorm_first)
decoder_upsample_first = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(
    decoder_activation_first)

decoder_conv_second = layers.SeparableConv2D(256, (5, 5), padding='same')(decoder_upsample_first)
decoder_batchnorm_second = layers.BatchNormalization()(decoder_conv_second)
decoder_activation_second = layers.Activation('relu')(decoder_batchnorm_second)
decoder_upsample_second = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(
    decoder_activation_second)

decoder_conv_third = layers.SeparableConv2D(128, (7, 7), padding='same')(decoder_upsample_second)
decoder_batchnorm_third = layers.BatchNormalization()(decoder_conv_third)
decoder_activation_third = layers.Activation('relu')(decoder_batchnorm_third)
decoder_upsample_third = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(
    decoder_activation_third)

decoder_conv_fourth = layers.SeparableConv2D(64, (7, 7), padding='same')(decoder_upsample_third)
decoder_batchnorm_fourth = layers.BatchNormalization()(decoder_conv_fourth)
decoder_activation_fourth = layers.Activation('relu')(decoder_batchnorm_fourth)
decoder_upsample_fourth = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(
    decoder_activation_fourth)

decoder_out = layers.SeparableConv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_upsample_fourth)

decoder_model = Model(input_img_left, decoder_out)

img_decoded = decoder_model(input_img_left)


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


epochs = 5

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

    decoder_model.save_weights('Training_Models/model_convnet_unsupervised_mpi.h5')





