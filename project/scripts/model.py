import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
K.set_image_dim_ordering('th')

#dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = '/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/img/test/'
validation_data_dir = '/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/img/train/'
#train_data_dir = 'img/test'
#validation_data_dir = 'img/train'

nb_train_samples = 120 #1000 #2000
nb_validation_samples = 30 #60 #800


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
'''
model.add(Convolution2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
'''
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3)) ####### previously it was 1
model.add(Activation('softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
##this is the augmentation configuration we will use for training

trainer_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        rescale=1./255,
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=20,
    class_mode='categorical') ########

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical') ########

nb_epoch = 2


model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples/8,
    #samples_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/2)
print(model.evaluate_generator(validation_generator, nb_validation_samples/5))

model.save("/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/final_model_now.h5")