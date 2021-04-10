# data management
import numpy as np
import pandas as pd

# flow cytometry libraries
import cytoflow as flow

def tube_to_db(cytometry_experiment):
    # wrapper function for extracting data for cytoflow experiment
    cyto_db = cytometry_experiment.data
    cyto_db = cyto_db.drop(columns=['EventNum', 'Time', 'Cell Length'])
    return cyto_db