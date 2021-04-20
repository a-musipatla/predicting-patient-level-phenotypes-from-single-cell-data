# data management
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# flow cytometry libraries
import cytoflow as flow

# math libraries
import random

# TensorFlow
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

def split_dataset(dataset: tf.data.Dataset, test_data_fraction: float):
    # Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    #   rounded up to two decimal places.
    # Input:
    #       dataset: the input dataset to split.
    #       validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    # Return: 
    #       a tuple of two tf.data.Datasets as (training, validation)
    # Source: https://stackoverflow.com/questions/59669413/what-is-the-canonical-way-to-split-tf-dataset-into-test-and-validation-subsets

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset