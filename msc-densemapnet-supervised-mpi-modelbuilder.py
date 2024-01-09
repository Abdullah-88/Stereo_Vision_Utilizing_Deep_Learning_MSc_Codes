from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import os, shutil
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
import matplotlib.image as mpimg
from PIL import Image

from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose, SeparableConv2D
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model


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



dropout = 0.5

left = Input(shape=(748, 744, 3))
right = Input(shape=(748, 744, 3))

# left image as reference

x = SeparableConv2D(filters=128, kernel_size=5, padding='same')(left)
xleft = SeparableConv2D(filters=1,
                        kernel_size=5,
                        padding='same',
                        dilation_rate=2)(left)

# left and right images for disparity estimation
xin = keras.layers.concatenate([left, right])
xin = SeparableConv2D(filters=128, kernel_size=5, padding='same')(xin)

# image reduced by 8
x8 = MaxPooling2D(8)(xin)
x8 = BatchNormalization()(x8)
x8 = Activation('relu', name='downsampled_stereo')(x8)

dilation_rate = 1
y = x8
# correspondence network
# parallel cnn at increasing dilation rate
for i in range(4):
    a = SeparableConv2D(filters=64,
                        kernel_size=5,
                        padding='same',
                        dilation_rate=dilation_rate)(x8)
    a = Dropout(dropout)(a)
    y = keras.layers.concatenate([a, y])
    dilation_rate += 1

dilation_rate = 1
x = MaxPooling2D(8)(x)
# disparity network
# dense interconnection inspired by DenseNet
for i in range(4):
    x = keras.layers.concatenate([x, y])
    y = BatchNormalization()(x)
    y = Activation('relu')(y)
    y = SeparableConv2D(filters=64,
                        kernel_size=1,
                        padding='same')(y)

    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters=64,
               kernel_size=5,
               padding='same',
               dilation_rate=dilation_rate)(y)
    y = Dropout(dropout)(y)
    dilation_rate += 1

# disparity estimate scaled back to original image size
x = keras.layers.concatenate([x, y], name='upsampled_disparity')
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(filters=64, kernel_size=1, padding='same')(x)
x = UpSampling2D(8)(x)
# if not self.settings.nopadding:
x = ZeroPadding2D(padding=(2, 0))(x)

# left image skip connection to disparity estimate
x = keras.layers.concatenate([x, xleft])
y = BatchNormalization()(x)
y = Activation('relu')(y)
y = SeparableConv2D(filters=64, kernel_size=5, padding='same')(y)

x = keras.layers.concatenate([x, y])
y = BatchNormalization()(x)
y = Activation('relu')(y)
y = SeparableConv2D(filters=3, kernel_size=9, padding='same')(y)

# prediction
# if self.settings.otanh:
#   yout = Activation('tanh', name='disparity_output')(y)
# else:
yout = Activation('sigmoid', name='disparity_output')(y)

# densemapnet model
test_model = Model([left, right], yout)

# if self.settings.model_weights:
#   print("Loading checkpoint model weights %s...."
#        % self.settings.model_weights)
# self.model.load_weights(self.settings.model_weights)

# if self.settings.otanh:
callback_list = [
    keras.callbacks.ModelCheckpoint(filepath='Training_Models/model_densemapnet_supervised-mpi.h5',
                                    monitor='val_loss', save_best_only=False),
    keras.callbacks.TensorBoard(log_dir='Logs')]
test_model.compile(loss='binary_crossentropy',
                   optimizer=RMSprop(lr=1e-3, decay=1e-6), metrics=['acc'])
# else:
#   self.model.compile(loss='mse',
#                     optimizer=RMSprop(lr=lr))

print("DenseMapNet Model:")
test_model.summary()
# plot_model(self.model, to_file='densemapnet.png', show_shapes=True)

# return self.model

# Data generator

train_left_datagen = ImageDataGenerator(rescale=1. / 255)
train_right_datagen = ImageDataGenerator(rescale=1. / 255)
train_target_datagen = ImageDataGenerator(rescale=1. / 255)
validation_left_datagen = ImageDataGenerator(rescale=1. / 255)
validation_right_datagen = ImageDataGenerator(rescale=1. / 255)
validation_target_datagen = ImageDataGenerator(rescale=1. / 255)
test_left_datagen = ImageDataGenerator(rescale=1. / 255)
test_right_datagen = ImageDataGenerator(rescale=1. / 255)


def multi_train_generator(train_left_datagen, train_right_datagen, train_target_datagen):
    train_left_generator = train_left_datagen.flow_from_directory(train_left_dir, target_size=(748, 744), batch_size=2,shuffle=True,
                                                                  class_mode=None)
    train_right_generator = train_right_datagen.flow_from_directory(train_right_dir, target_size=(748, 744),
                                                                    batch_size=2,shuffle=True, class_mode=None)
    train_target_generator = train_target_datagen.flow_from_directory(train_target_dir, target_size=(748, 744),
                                                                      batch_size=2,shuffle=True, class_mode=None)
    while True:
        x = train_left_generator.__next__()
        y = train_right_generator.__next__()
        z = train_target_generator.__next__()
        yield [x, y], z


def multi_validation_generator(validation_left_datagen, validation_right_datagen, validation_target_datagen):
    validation_left_generator = validation_left_datagen.flow_from_directory(validation_left_dir, target_size=(748, 744),
                                                                            batch_size=1,shuffle=True, class_mode=None)
    validation_right_generator = validation_right_datagen.flow_from_directory(validation_right_dir,
                                                                              target_size=(748, 744), batch_size=1,shuffle=True,
                                                                              class_mode=None)
    validation_target_generator = validation_target_datagen.flow_from_directory(validation_target_dir,
                                                                                target_size=(748, 744), batch_size=1,shuffle=True,
                                                                                class_mode=None)
    while True:
        x = validation_left_generator.__next__()
        y = validation_right_generator.__next__()
        z = validation_target_generator.__next__()
        yield [x, y], z


def multi_test_generator(test_left_datagen, test_right_datagen):
    test_left_generator = test_left_datagen.flow_from_directory(test_left_dir, target_size=(748, 744), batch_size=1,
                                                                class_mode=None)
    test_right_generator = test_right_datagen.flow_from_directory(test_right_dir, target_size=(748, 744), batch_size=1,
                                                                  class_mode=None)
    while True:
        x = test_left_generator.__next__()
        y = test_right_generator.__next__()
        yield [x, y]


# model training

history = test_model.fit_generator(multi_train_generator(train_left_datagen, train_right_datagen, train_target_datagen),
                                   steps_per_epoch=900, epochs=1000,
                                   validation_data=multi_validation_generator(validation_left_datagen,
                                                                              validation_right_datagen,
                                                                              validation_target_datagen),
                                   validation_steps=150, callbacks=callback_list)

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