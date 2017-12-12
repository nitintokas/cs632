import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

K.set_image_dim_ordering('th')

#dimensions of our images.
img_width, img_height = 64, 64
#img_width, img_height = 256, 256

train_data_dir = '/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/img/test/'
validation_data_dir = '/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/img/train/'
#train_data_dir = 'img/test'
#validation_data_dir = 'img/train'

nb_train_samples = 120 #1000 #2000
nb_validation_samples = 30 #60 #800

model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (3, img_width, img_height))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:8]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=24,
    class_mode='categorical') ########

validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=5,
    class_mode='categorical') ########

nb_epoch = 3


model_final.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples/12,
    #samples_per_epoch=nb_train_samples,
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples/3)
print(model_final.evaluate_generator(validation_generator, nb_validation_samples/5))

#model.save("/Applications/XAMPP/xamppfiles/htdocs/projects/deep_learning/scripts/final_model_now.h5")