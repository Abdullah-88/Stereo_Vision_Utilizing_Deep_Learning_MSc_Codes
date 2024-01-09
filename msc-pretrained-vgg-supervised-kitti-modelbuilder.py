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
from keras.applications import VGG16
import matplotlib.image as mpimg
from PIL import Image
# Data source

base_dir = "StereoDatakitti"
train_left_dir = os.path.join(base_dir,'train_left')
train_right_dir = os.path.join(base_dir,'train_right')
train_target_dir = os.path.join(base_dir,'train_target')
validation_left_dir = os.path.join(base_dir,'validation_left')
validation_right_dir = os.path.join(base_dir,'validation_right')
validation_target_dir = os.path.join(base_dir,'validation_target')
test_left_dir = os.path.join(base_dir,'test_left')
test_right_dir = os.path.join(base_dir,'test_right')

conv_base = VGG16(weights='imagenet',include_top=False)

#left convnet

input_img_left = Input(shape=(750,750,3))
pre_trained_vgg19_left = conv_base(input_img_left)
left_batchnorm = layers.BatchNormalization()(pre_trained_vgg19_left)
left_out = left_batchnorm




#right convnet

input_img_right = Input(shape=(750,750,3))
pre_trained_vgg19_right = conv_base(input_img_right)
right_batchnorm = layers.BatchNormalization()(pre_trained_vgg19_right)
right_out = right_batchnorm



#combining with dense layers

merge_layer = layers.concatenate([left_out,right_out],axis=-1)
#dropout_layer = layers.Dropout(0.5)(merge_layer)
#encoder_dense_first = layers.Dense(2048,activation='relu')(dropout_layer)
#encoder_dense_second = layers.Dense(1024,activation='relu')(encoder_dense_first)

#decoder network

#decoder_dense_first = layers.Dense(1024,activation='relu')(merge_layer)
#unflatten = layers.Reshape((2,2,256))(decoder_dense_first)
decoder_conv_first = layers.Conv2D(64,(3,3),activation='relu',padding='same')(merge_layer)
upsample_first = layers.UpSampling2D(size=(4,4), data_format=None, interpolation='nearest')(decoder_conv_first)
decoder_conv_second = layers.Conv2D(32,(3,3),activation='relu',padding='same')(upsample_first)
upsample_second = layers.UpSampling2D(size=(4,4), data_format=None, interpolation='nearest')(decoder_conv_second )
decoder_conv_third = layers.Conv2D(16,(3,3),activation='relu',padding='same')(upsample_second)
upsample_third = layers.UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(decoder_conv_third)
decoder_out = layers.Conv2D(3,(3,3),activation='sigmoid',padding='same')(upsample_third)
#upsample_forth = layers.UpSampling2D(size=(2,2), data_format=None, interpolation='nearest')(unconv_forth)
#decoder_out = layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(upsample_forth)



# Model building


depth_mapper = Model([input_img_left,input_img_right],decoder_out)

for layer in conv_base.layers[:16]:
   layer.trainable = False
for layer in conv_base.layers[16:]:
   layer.trainable = True

callback_list = [keras.callbacks.ModelCheckpoint(filepath='Training_Models/model_vgg16_supervised_kitti.h5',monitor='val_loss',save_best_only=True),keras.callbacks.TensorBoard(log_dir='Logs')]
depth_mapper.compile(optimizer=optimizers.RMSprop(lr=1e-3,decay=1e-6),loss='binary_crossentropy',metrics=['acc'])
depth_mapper.summary()



# Data generator
train_left_datagen = ImageDataGenerator(rescale=1./255)
train_right_datagen = ImageDataGenerator(rescale=1./255)
train_target_datagen = ImageDataGenerator(rescale=1./255)
validation_left_datagen = ImageDataGenerator(rescale=1./255)
validation_right_datagen = ImageDataGenerator(rescale=1./255)
validation_target_datagen = ImageDataGenerator(rescale=1./255)
test_left_datagen = ImageDataGenerator(rescale=1./255)
test_right_datagen = ImageDataGenerator(rescale=1./255)

def multi_train_generator(train_left_datagen, train_right_datagen, train_target_datagen):
    train_left_generator = train_left_datagen.flow_from_directory(train_left_dir, target_size=(750, 750), batch_size=2,
                                                                  shuffle=True,class_mode=None)
    train_right_generator = train_right_datagen.flow_from_directory(train_right_dir, target_size=(750, 750),
                                                                    batch_size=2,shuffle=True, class_mode=None)
    train_target_generator = train_target_datagen.flow_from_directory(train_target_dir, target_size=(736, 736),
                                                                      batch_size=2, shuffle=True,class_mode=None)
    while True:
        x = train_left_generator.__next__()
        y = train_right_generator.__next__()
        z = train_target_generator.__next__()
        yield [x, y], z


def multi_validation_generator(validation_left_datagen, validation_right_datagen, validation_target_datagen):
    validation_left_generator = validation_left_datagen.flow_from_directory(validation_left_dir, target_size=(750, 750),
                                                                            shuffle=True,batch_size=1, class_mode=None)
    validation_right_generator = validation_right_datagen.flow_from_directory(validation_right_dir,
                                                                              target_size=(750, 750),shuffle=True, batch_size=1,
                                                                              class_mode=None)
    validation_target_generator = validation_target_datagen.flow_from_directory(validation_target_dir,
                                                                                target_size=(736, 736), shuffle=True,batch_size=1,
                                                                                class_mode=None)
    while True:
        x = validation_left_generator.__next__()
        y = validation_right_generator.__next__()
        z = validation_target_generator.__next__()
        yield [x, y], z


def multi_test_generator(test_left_datagen, test_right_datagen):
    test_left_generator = test_left_datagen.flow_from_directory(test_left_dir, target_size=(750, 750), batch_size=1,
                                                                class_mode=None)
    test_right_generator = test_right_datagen.flow_from_directory(test_right_dir, target_size=(750, 750), batch_size=1,
                                                                  class_mode=None)
    while True:
        x = test_left_generator.__next__()
        y = test_right_generator.__next__()
        yield [x, y]


# model training

history =depth_mapper.fit_generator(multi_train_generator(train_left_datagen,train_right_datagen,train_target_datagen),steps_per_epoch=150,epochs=1000,validation_data=multi_validation_generator(validation_left_datagen,validation_right_datagen,validation_target_datagen),validation_steps=50,callbacks=callback_list)

# plotting results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label="Validation acc")
plt.title('Training and validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label="Validatin loss")
plt.title('Training and validatin Loss')
plt.legend()

plt.show()


