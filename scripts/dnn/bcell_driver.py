# data management
import numpy as np
import pandas as pd

# flow cytometry libraries
import cytoflow as flow

# user defined functions
import bcell_preprocess as preprocess

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
print('\n')
print("Channels in this experiment: \n", ex.channels)
print(len(ex.channels))
print('\n')

# Extract cytometry data in pandas database format
cells_db   = preprocess.tube_to_db(ex)
print(cells_db.head)


