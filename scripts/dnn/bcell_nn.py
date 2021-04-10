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
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)

    return model