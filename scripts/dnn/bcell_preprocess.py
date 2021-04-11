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
    # shuffle and batch
    train_dataset = dataset.shuffle(len(cyto_df)).batch(1)
    return train_dataset

def df_to_test_tensor(cyto_df):
    # function that returns tensor features
    cyto_df = cyto_df
    dataset = tf.data.Dataset.from_tensor_slices((cyto_df.values))
    return dataset

def compute_scale_vector(cyto_df):
    # function that computes a scale vector for normalizing a dataset
    # Input:
    #       Base data frame to normalize against. This can be the training set,
    #       or a subset of the training set. 
    # Output:
    #       A vector that indicates what to scale each feature in a train and test
    #       dataset with in order to normalize it. 
    return cyto_df