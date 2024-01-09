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

decoder_model.load_weights('Training_Models/model_convnet_unsupervised_mpi.h5')

# img_decoded = decoder_model(input_img_left)


# Model predicting


decoder_model.summary()

preds = decoder_model.predict_generator(multi_test_generator(test_left_datagen, test_right_datagen), steps=150,
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