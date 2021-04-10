# data management
import numpy as np
import pandas as pd

# flow cytometry libraries
import cytoflow as flow

# TensorFlow and 
import tensorflow as tf

def tube_to_df(cytometry_experiment):
    # wrapper function for extracting data for cytoflow experiment
    cyto_df = cytometry_experiment.data
    cyto_df = cyto_df.drop(columns=['EventNum', 'Time', 'Cell Length'])
    return cyto_df

def df_to_train_tensor(cyto_df):
    # function that returns tensor features and categories
    y = cyto_df.pop('bcr')
    dataset = tf.data.Dataset.from_tensor_slices((cyto_df.values, y.values))
    return dataset

def df_to_test_tensor(cyto_df):
    # function that returns tensor features
    cyto_df = cyto_df
    dataset = tf.data.Dataset.from_tensor_slices((cyto_df.values))
    return dataset