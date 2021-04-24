import os

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def define_model(shape=None, dropout=0.1):

    ###
    # inputs:
    #   shape: the number of nodes per each layer of neural network. First element is the input size
    #   dropout: dropout layer will be added to the penultimate layer, dropout size
    # 
    # assumes that the last layer will have dimension 1 due to binary classification
    ###

    model = None
    if not shape:
        # Define Sequential model with 3 layers
        model = keras.Sequential(
            [
                layers.Dense(20, activation="relu", name="layer1"),
                layers.Dense(7, activation="relu", name="layer2"),
                layers.Dropout(dropout),
                layers.Dense(1, activation='sigmoid', name="layer3"),
            ]
        )
    
    else:
        model = keras.Sequential()
        for i, x in enumerate(shape):
            model.add(layers.Dense(x, activation='relu', name='layer%d' % (i+1)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(1, activation='sigmoid', name='layer%d' % len(shape)))


    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),      # removed from_logits = True, seems to solve gradiency issue
                metrics=['accuracy'])

    return model

def save_model(model, model_filename, history=None):
    # Saves a model at specified location. If default location, will not overwrite previous 
    # saved models.
    # Input:
    #       model: the keras model
    #       model_filename: 
    #       history: pass in a history object to save with the same name
    # Return: 
    #       Filepath of saved model

    # Increment file name to prevent overwrites
    counter = 0
    temp_filename = model_filename + "{}"
    while os.path.isfile(temp_filename.format(counter)):
        counter += 1
    temp_filename = temp_filename.format(counter)

    model.save(temp_filename)
    if history:
        np.save(temp_filename, history.history)               

    return temp_filename

def fit_model(model, train_dataset, val_dataset, test_dataset, k=1, batch_size=1024, epochs=15, patience=5):
    # Trains a model on input dataset
    # Input:
    #       model: the keras model
    #       train_dataset: tensorslice dataset
    #       test_dataset: tensorslice dataset
    #       k: folds
    #       epochs: number for epochs to train
    # Return: 
    #       Fitted model

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        train_dataset, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_dataset,
        callbacks=[callback]
    )

    return model, history
