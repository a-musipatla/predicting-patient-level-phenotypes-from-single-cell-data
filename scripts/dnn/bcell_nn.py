import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def define_model():
    # Define Sequential model with 3 layers
    model = keras.Sequential(
        [
            layers.Dense(20, activation="relu", name="layer1"),
            layers.Dense(7, activation="relu", name="layer2"),
            layers.Dense(1, name="layer3"),
        ]
    )
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def save_model(model, model_filename):
    # Saves a model at specified location. If default location, will not overwrite previous 
    # saved models.
    # Input:
    #       model: the keras model
    #       model_filename: 
    # Return: 
    #       Filepath of saved model

    # Increment file name to prevent overwrites
    counter = 0
    temp_filename = model_filename + "{}"
    while os.path.isfile(temp_filename.format(counter)):
        counter += 1
    temp_filename = temp_filename.format(counter)

    model.save(temp_filename)

    return temp_filename

def fit_model(model, train_dataset, val_dataset, test_dataset, k=1, epochs=15):
    # Trains a model on input dataset
    # Input:
    #       model: the keras model
    #       train_dataset: tensorslice dataset
    #       test_dataset: tensorslice dataset
    #       k: folds
    #       epochs: number for epochs to train
    # Return: 
    #       Fitted model

    model.fit(
        train_dataset, 
        epochs=epochs,
        validation_data=val_dataset
    )

    return model
