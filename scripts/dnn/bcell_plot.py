# data management
import numpy as np
import pandas as pd

# visualization
import seaborn as sns

def value_count_plot(cyto_df):
    return cyto_df['bcr'].value_counts()
