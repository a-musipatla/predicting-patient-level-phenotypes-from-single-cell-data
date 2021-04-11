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
    pass