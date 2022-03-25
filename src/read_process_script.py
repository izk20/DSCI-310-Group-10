# author: Ahmed Rizk
# date: 2022-03-24
#
# A script that sets a universal seed, reads the dataset used for the analysis
# from CIS-72M0003-E-2017-Annual_F1.csv, and processes the data for the EDA.
#
# Usage:

# Options:
# --read_path=<read_path>    '''Path of the file to read'''
# --out_path=<out_path>    '''path to the cleaned dataset object created'''
# --cols_kept=<cols_kept>  '''array of columns to keep from dataset'''

import numpy as np
import pandas as pd
from docopt import docopt

np.random.seed(1)

opt = docopt(__doc__)

def main(read_path, out_name, cols_kept):

    data = pd.read_csv(read_path, header=None)
    data = data[cols_kept]
    data.to_csv(data, index = False)
    
main(opt["--read_path"], opt["--out_name"], opt[["--cols_kept"]])


