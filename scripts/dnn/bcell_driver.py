# System arguments
import argparse
# data management
import numpy as np
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
# flow cytometry libraries
import cytoflow as flow
# user defined functions
import bcell_preprocess as bpreprocess
import bcell_plot as bplot
import bcell_nn as bnn
import json


# read command line flags
parser = argparse.ArgumentParser(description='Run a DNN on cytometry data.')
parser.add_argument('-v', '--verbose', 
                    help='Print information and status to console.',
                    action='store_true')
parser.add_argument('-p', '--plot', 
                    help='Display plots.',
                    action='store_true')
parser.add_argument('-k', '--k_fold',
                    type=int, 
                    default=1, 
                    help="Number of folds in k-fold cross validation.")
parser.add_argument('-b', '--batch_size',
                    type=int, 
                    default=1024, 
                    help="Batch size")
parser.add_argument('-s', '--test_split', 
                    type=float, 
                    default=0.1, 
                    help="Fraction of dataset held out as test.")
parser.add_argument('-vs', '--val_split', 
                    type=float, 
                    default=0.1, 
                    help="Fraction of dataset held out as val.")
parser.add_argument('-f', '--frac', 
                    type=float, 
                    default=0.5, 
                    help="Fraction of total dataset we should use")
parser.add_argument('-mp', '--model_params', 
                    type=str, 
                    default=None, 
                    help="Model params to pass into the model function")
parser.add_argument('--model_filename',
                    default='models/checkpoint_',
                    help="File path and name to save output model.")
args = parser.parse_args()

# specify data files
marrow_basal_file = 'data/B_cell_data/Marrow1_01_Basal1.fcs'
marrow_bcr_file   = 'data/B_cell_data/Marrow1_06_BCR.fcs'

# using the cytoflow package
basal_tube = flow.Tube(file = marrow_basal_file,
                  conditions = {'bcr' : 0.0})
bcr_tube   = flow.Tube(file=marrow_bcr_file,
                  conditions = {'bcr' : 1.0})

import_op = flow.ImportOp(conditions = {'bcr' : 'float'},
                          tubes = [basal_tube, bcr_tube])


ex = import_op.apply()

# Check channels in the experiment
if args.verbose:
    # print information about the channels in the experiment
    print('\n')
    print('Number of channels: ', len(ex.channels))
    print("Channels in this experiment: \n", ex.channels)
    print('\n')

# Extract cytometry data in pandas database format
cells_df   = bpreprocess.tube_to_df(ex)
if args.verbose:
    # print information about the dataframes with cytometry data
    print('\n')
    print('Information on the cytometry measurement dataframe:')
    print(cells_df.head)
    print(cells_df.dtypes)
    print('\n')

# EDA plots
if args.plot:
    # display a plot showing the count per category
    bplot.value_count_plot(cells_df).plot(kind = 'bar')
    plt.title("Cell Count by Category")
    plt.xlabel("BCR Stimulated/Unstimulated")
    plt.ylabel("Number of Cells")
    plt.show(block=True)

# Format data as tensorslicedataset
#       https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
cyto_dataset = bpreprocess.df_to_train_tensor(cells_df, use=args.frac)
if args.verbose:
    # print information on the cell data tensors
    print('\n')
    print('Tensor Dataset:\t', cyto_dataset)
    for feat, targ in cyto_dataset.take(5):
        print('Information on the cytometry measurement tensors:')
        print('Tensor shape:\t', feat.shape)
        print('Features: {}, Target: {}'.format(feat, targ))
        print('\n')

# Train/Test split
if args.verbose:
    # print information on split
    print('\n')
    print('Using', args.test_split, 'of all available data')
    print('Holding out ', args.test_split, 'of dataset for testing.')
    print('Holding out ', args.val_split, 'of dataset for validation.')
    print('\n')
train_dataset, val_dataset, test_dataset = bpreprocess.split_dataset(cyto_dataset, args.val_split, args.test_split)


if args.model_params:
    with open(args.model_params) as json_file:
        params = json.load(json_file)
    model = bnn.define_model(**params)
else:
    # Initialize base DNN model 
    model = bnn.define_model()

# Train model
_, history = bnn.fit_model(model, train_dataset, val_dataset, train_dataset, batch_size=args.batch_size, epochs=15)

# Calculate Acc
score, acc = model.evaluate(test_dataset)
print('\n')
print('Test score:', score)
print('Test accuracy:', acc)
print('\n')

# Save model
model_filename = bnn.save_model(model, args.model_filename, history=history)
print("Model saved to: ", model_filename)