# Author: Angelo C. Guiam
# Exercise 02, CMSC 265, 2S 2020-2021

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

TRAIN_DIR = 'data_training'
VALIDATION_DIR = 'data_validation'
CLASSES = 4

#train generator will augment and feed images into the model transfer learning
train_generator = ImageDataGenerator(
                    rescale = 1.0/255.0,
                    rotation_range = 40,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True)

train_generator = train_generator.flow_from_directory(
                    TRAIN_DIR,
                    batch_size = 20,
                    class_mode = 'categorical',
                    target_size = (150, 150))

#train generator will augment and feed images for the validation in the transfer learning
valid_generator = ImageDataGenerator(rescale = 1.0/255.0)

valid_generator = valid_generator.flow_from_directory(
                    VALIDATION_DIR,
                    batch_size = 20,
                    class_mode = 'categorical',
                    target_size = (150, 150))

#created a model based from InceptionV3 architecture with weights from imagenet dataset
base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

#setting configuration for the transfer learning
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(CLASSES, activation="sigmoid")(x)

model = models.Model(base_model.input, x)
model.compile(optimizer = optimizers.RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#retrain model from the training dataset
history = model.fit(
    x = train_generator,
    epochs = 10,
    steps_per_epoch = train_generator.n//train_generator.batch_size,
    validation_data = valid_generator,
    validation_steps = valid_generator.n//valid_generator.batch_size)
#run model validation
model.evaluate(
    x = valid_generator,
    steps = valid_generator.n//valid_generator.batch_size)
#export model to inception folder for retrieval
model.save("inception")