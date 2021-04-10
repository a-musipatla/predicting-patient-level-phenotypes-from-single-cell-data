# data management
import numpy as np
import pandas as pd

# flow cytometry libraries
import cytoflow as flow

def tube_to_df(cytometry_experiment):
    # wrapper function for extracting data for cytoflow experiment
    cyto_df = cytometry_experiment.data
    cyto_df = cyto_df.drop(columns=['EventNum', 'Time', 'Cell Length'])
    return cyto_df