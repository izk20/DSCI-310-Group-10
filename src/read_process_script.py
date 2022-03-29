# author: Ahmed Rizk
# date: 2022-03-24

''' A script that sets a universal seed, reads the dataset used for the analysis
 from specified path, and processes the data for the EDA.

 Usage:

 Options:
 --read_path=<read_path>    Path of the file to read
 --processed_path=<processed_path>    path to the processed dataset object created, before the training/testing split
 --train_path=<train_path> path to the training data
 --test_path=<test_path> path to the testing data

''' 

import numpy as np
import pandas as pd
from docopt import docopt

np.random.seed(1)

opt = docopt(__doc__)

def read_trim(read_path, chosen_cols):
    data = pd.read_csv(read_path, header=None)
    reduced_data = data[chosen_cols]
    return reduced_data


def process(data, processed_path):
    data = data.drop(columns = 'USHRWK') 
    data = data.loc[reduced_data['ATINC'] != 99999996] 
    processed = data[["EFINVA","EFSIZE","EFMJIE"]]
    return processed    # should I split into 3?



def train_test_drop(data, dropped_col, train_path, test_path):
    X_train, Y_train, X_test, Y_test = split_drop(data, 0.3, 123, dropped_col)
    return X_train, Y_train, X_test, Y_test
    

def write_to_csv(data, path, filename):
    data.to_csv(path + filename, index=True)
    
    

def main(read_path, processed_path, train_path, test_path):
    reduced_data = read_trim(read_path, ['EFSIZE', 'USHRWK', 'ATINC', 'HLEV2G', 'EFINVA', 'EFMJIE', 'EFATINC', 'EFMJSI'])
    processed = process(reduced_data)
    X_train, Y_train, X_test, Y_test = train_test_drop(processed, "EFINVA")
    write(X_train, train_path,"X_train.csv")
    write(Y_train, train_path, "Y_train.csv")
    write(X_test, test_path, "X_test.csv")
    write(Y_test, test_path ,"Y_test.csv")
    write(processed, processed_path, "processed.csv")

    
    
main(opt["--read_path"], opt["--out_path"], opt[["--cols_kept"]])


