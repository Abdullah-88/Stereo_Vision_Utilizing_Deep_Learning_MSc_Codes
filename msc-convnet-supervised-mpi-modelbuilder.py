import keras
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import metrics
from keras import Input
from keras import Model
from keras import activations
from keras.regularizers import l2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.image as mpimg
from PIL import Image

# Data source

base_dir = "StereoDatampi"
train_left_dir = os.path.join(base_dir, 'train_left')
train_right_dir = os.path.join(base_dir, 'train_right')
train_target_dir = os.path.join(base_dir, 'train_target')
validation_left_dir = os.path.join(base_dir, 'validation_left')
validation_right_dir = os.path.join(base_dir, 'validation_right')
validation_target_dir = os.path.join(base_dir, 'validation_target')
test_left_dir = os.path.join(base_dir, 'test_left')
test_right_dir = os.path.join(base_dir, 'test_right')

# layers definition for sharing

conv_32 = layers.SeparableConv2D(32, (7, 7), padding='same')
batchnormalization_32 = layers.BatchNormalization()
activation_relu_32 = layers.Activation('relu')

conv_64 = layers.SeparableConv2D(64, (7, 7), padding='same')
batchnormalization_64 = layers.BatchNormalization()
activation_relu_64 = layers.Activation('relu')

conv_128 = layers.SeparableConv2D(128, (5, 5), padding='same')
batchnormalization_128 = layers.BatchNormalization()
activation_relu_128 = layers.Activation('relu')

conv_256 = layers.SeparableConv2D(256, (3, 3), padding='same')
batchnormalization_256 = layers.BatchNormalization()
activation_relu_256 = layers.Activation('relu')

maxpooling_32 = layers.MaxPooling2D((2, 2))
maxpooling_64 = layers.MaxPooling2D((2, 2))
maxpooling_128 = layers.MaxPooling2D((2, 2))
maxpooling_256 = layers.MaxPooling2D((2, 2))

maxpooling_8 = layers.MaxPooling2D((8, 8))


# left convnet


input_img_left = Input(shape=(672, 672, 3))

left_block1_conv1 = conv_32(input_img_left)
left_block1_batchnorm = batchnormalization_32(left_block1_conv1)
left_block1_activation = activation_relu_32(left_block1_batchnorm)
left_block1_maxpool = maxpooling_32(left_block1_activation)

left_block2_conv1 = conv_64(left_block1_maxpool)
left_block2_batchnorm = batchnormalization_64(left_block2_conv1)
left_block2_activation = activation_relu_64(left_block2_batchnorm)
left_block2_maxpool = maxpooling_64(left_block2_activation)

left_block3_conv1 = conv_128(left_block2_maxpool)
left_block3_batchnorm = batchnormalization_128(left_block3_conv1)
left_block3_activation = activation_relu_128(left_block3_batchnorm)
left_block3_maxpool = maxpooling_128(left_block3_activation)

left_skip_maxpool = maxpooling_8(left_block1_activation)
left_skip_concat = layers.concatenate([left_skip_maxpool, left_block3_maxpool], axis=-1)

left_block4_conv1 = conv_256(left_skip_concat)
left_block4_batchnorm = batchnormalization_256(left_block4_conv1)
left_block4_activation = activation_relu_256(left_block4_batchnorm)
left_block4_maxpool = maxpooling_256(left_block4_activation)

left_out = left_block4_maxpool

# right convnet

input_img_right = Input(shape=(672, 672, 3))

right_block1_conv1 = conv_32(input_img_right)
right_block1_batchnorm = batchnormalization_32(right_block1_conv1)
right_block1_activation = activation_relu_32(right_block1_batchnorm)
right_block1_maxpool = maxpooling_32(right_block1_activation)

right_block2_conv1 = conv_64(right_block1_maxpool)
right_block2_batchnorm = batchnormalization_64(right_block2_conv1)
right_block2_activation = activation_relu_64(right_block2_batchnorm)
right_block2_maxpool = maxpooling_64(right_block2_activation)

right_block3_conv1 = conv_128(right_block2_maxpool)
right_block3_batchnorm = batchnormalization_128(right_block3_conv1)
right_block3_activation = activation_relu_128(right_block3_batchnorm)
right_block3_maxpool = maxpooling_128(right_block3_activation)

right_skip_maxpool = maxpooling_8(right_block1_activation)
right_skip_concat = layers.concatenate([right_skip_maxpool, right_block3_maxpool], axis=-1)

right_block4_conv1 = conv_256(right_skip_concat)
right_block4_batchnorm = batchnormalization_256(right_block4_conv1)
right_block4_activation = activation_relu_256(right_block4_batchnorm)
right_block4_maxpool = maxpooling_256(right_block4_activation)

right_out = right_block4_maxpool

# combining with dense layers

merge_layer = layers.multiply([left_out, right_out])
# merge_layer = layers.concatenate([left_out,right_out],axis=-1)
expand_layer_conv = layers.Conv2D(256, (4, 1), padding='same')(merge_layer)
expand_layer_batchnorm = layers.BatchNormalization()(expand_layer_conv)
expand_layer_activation = layers.Activation('relu')(expand_layer_batchnorm)
expand_layer_maxpooling = layers.MaxPooling2D((2, 2))(expand_layer_activation)

# decoder network

decoder_conv_match = layers.SeparableConv2D(256, (4, 1), padding='same')(expand_layer_maxpooling)
decoder_batchnorm_match = layers.BatchNormalization()(decoder_conv_match)
decoder_activation_match = layers.Activation('relu')(decoder_batchnorm_match)
upsample_match = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder_activation_match)
decoder_skip_concat_match = layers.concatenate([upsample_match, expand_layer_activation], axis=-1)

decoder_conv_first = layers.SeparableConv2D(256, (3, 3), padding='same')(decoder_skip_concat_match)
decoder_batchnorm_first = layers.BatchNormalization()(decoder_conv_first)
decoder_activation_first = layers.Activation('relu')(decoder_batchnorm_first)
upsample_first = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder_activation_first)
decoder_skip_concat_first = layers.concatenate([upsample_first, left_block4_activation], axis=-1)

decoder_conv_second = layers.SeparableConv2D(128, (5, 5), padding='same')(decoder_skip_concat_first)
decoder_batchnorm_second = layers.BatchNormalization()(decoder_conv_second)
decoder_activation_second = layers.Activation('relu')(decoder_batchnorm_second)
upsample_second = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder_activation_second)
decoder_skip_concat_second = layers.concatenate([upsample_second, left_block3_activation], axis=-1)

decoder_conv_third = layers.SeparableConv2D(64, (7, 7), padding='same')(decoder_skip_concat_second)
decoder_batchnorm_third = layers.BatchNormalization()(decoder_conv_third)
decoder_activation_third = layers.Activation('relu')(decoder_batchnorm_third)
upsample_third = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder_activation_third)
decoder_skip_concat_third = layers.concatenate([upsample_third, left_block2_activation], axis=-1)

decoder_conv_fourth = layers.SeparableConv2D(32, (7, 7), padding='same')(decoder_skip_concat_third)
decoder_batchnorm_fourth = layers.BatchNormalization()(decoder_conv_fourth)
decoder_activation_fourth = layers.Activation('relu')(decoder_batchnorm_fourth)
upsample_fourth = layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder_activation_fourth)
decoder_skip_upsample = layers.UpSampling2D(size=(32, 32), data_format=None, interpolation='nearest')(
    decoder_activation_match)
decoder_skip_concat_fourth = layers.concatenate([upsample_fourth, left_block1_activation, decoder_skip_upsample],
                                                axis=-1)
decoder_out = layers.SeparableConv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_skip_concat_fourth)

# Model building


depth_mapper = Model([input_img_left, input_img_right], decoder_out)
callback_list = [keras.callbacks.ModelCheckpoint(
    filepath='Training_Models/model_convnet_supervised_mpi.h5',
    monitor='val_loss', save_best_only=False), keras.callbacks.TensorBoard(log_dir='Logs')]
depth_mapper.compile(optimizer=optimizers.RMSprop(lr=1e-3, decay=1e-6), loss='binary_crossentropy', metrics=['acc'])
depth_mapper.summary()

#plot_model(depth_mapper, to_file='depthmapper.png', show_shapes=True)

# Data generator

train_left_datagen = ImageDataGenerator(rescale=1. / 255)
train_right_datagen = ImageDataGenerator(rescale=1. / 255)
train_target_datagen = ImageDataGenerator(rescale=1. / 255)
validation_left_datagen = ImageDataGenerator(rescale=1. / 255)
validation_right_datagen = ImageDataGenerator(rescale=1. / 255)
validation_target_datagen = ImageDataGenerator(rescale=1. / 255)
test_left_datagen = ImageDataGenerator(rescale=1. / 255)
test_right_datagen = ImageDataGenerator(rescale=1. / 255)

"""""# Data generator 
train_left_datagen = ImageDataGenerator(rescale=1./255)
train_right_datagen = ImageDataGenerator(rescale=1./255)
train_target_datagen = ImageDataGenerator(rescale=1./255)
validation_left_datagen = ImageDataGenerator(rescale=1./255)
validation_right_datagen = ImageDataGenerator(rescale=1./255)
validation_target_datagen = ImageDataGenerator(rescale=1./255)
test_left_datagen = ImageDataGenerator(rescale=1./255)
test_right_datagen = ImageDataGenerator(rescale=1./255)"""""""""


def multi_train_generator(train_left_datagen, train_right_datagen, train_target_datagen):
    train_left_generator = train_left_datagen.flow_from_directory(train_left_dir, target_size=(672, 672), batch_size=2,shuffle=True,
                                                                  class_mode=None)
    train_right_generator = train_right_datagen.flow_from_directory(train_right_dir, target_size=(672, 672),
                                                                    batch_size=2,shuffle=True, class_mode=None)
    train_target_generator = train_target_datagen.flow_from_directory(train_target_dir, target_size=(672, 672),
                                                                      batch_size=2,shuffle=True, class_mode=None)
    while True:
        x = train_left_generator.__next__()
        y = train_right_generator.__next__()
        z = train_target_generator.__next__()
        yield [x, y], z


def multi_validation_generator(validation_left_datagen, validation_right_datagen, validation_target_datagen):
    validation_left_generator = validation_left_datagen.flow_from_directory(validation_left_dir, target_size=(672, 672),
                                                                            batch_size=1, shuffle=True,class_mode=None)
    validation_right_generator = validation_right_datagen.flow_from_directory(validation_right_dir,
                                                                              target_size=(672, 672), batch_size=1,shuffle=True,
                                                                              class_mode=None)
    validation_target_generator = validation_target_datagen.flow_from_directory(validation_target_dir,
                                                                                target_size=(672, 672), batch_size=1,shuffle=True,
                                                                                class_mode=None)
    while True:
        x = validation_left_generator.__next__()
        y = validation_right_generator.__next__()
        z = validation_target_generator.__next__()
        yield [x, y], z


def multi_test_generator(test_left_datagen, test_right_datagen):
    test_left_generator = test_left_datagen.flow_from_directory(test_left_dir, target_size=(672, 672), batch_size=1,
                                                                class_mode=None)
    test_right_generator = test_right_datagen.flow_from_directory(test_right_dir, target_size=(672, 672), batch_size=1,
                                                                  class_mode=None)
    while True:
        x = test_left_generator.__next__()
        y = test_right_generator.__next__()
        yield [x, y]


# model training

history = depth_mapper.fit_generator(
    multi_train_generator(train_left_datagen, train_right_datagen, train_target_datagen), steps_per_epoch=900, epochs=1000,
    validation_data=multi_validation_generator(validation_left_datagen, validation_right_datagen,
                                               validation_target_datagen), validation_steps=150,
    callbacks=callback_list)

# plotting results

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label="Validation acc")
plt.title('Training and validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label="Validatin loss")
plt.title('Training and validatin Loss')
plt.legend()

plt.show()


