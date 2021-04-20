# Author: Angelo C. Guiam
# Exercise 02, CMSC 265, 2S 2020-2021

import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers

TEST_FOLDERS = 21
TRAIN_DIR = 'data_training'

#this will synchronize the labels from the model training
train_generator = ImageDataGenerator().flow_from_directory(
                    TRAIN_DIR,
                    batch_size = 20,
                    class_mode = 'categorical',
                    target_size = (150, 150))

#load model exported after retraining
model = models.load_model("inception")

#iterate all test folders
for i in range (1, TEST_FOLDERS+1):
    #this will create an image generator that contains the test images
    test_generator = ImageDataGenerator(
                        rescale = 1.0/255.0)

    test_generator = test_generator.flow_from_directory(
                        f"data_test/{i:03}",
                        batch_size = 1,
                        class_mode = None,
                        shuffle = False, #turn off shuffling of images for a consistent result
                        target_size = (150, 150))
    #generate prediction list from test data
    prediction = model.predict(x = test_generator)
    #map the prediction to labels/classes
    predicted_class_indices = np.argmax(prediction, axis = 1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    #write result to file
    filename = f"{i:03}.txt"
    file = open(filename,"w")
    filenames = test_generator.filenames

    for idx, prediction in enumerate(predictions):
        file.write(f"{idx+1} {prediction}\n")
    
    file.close()