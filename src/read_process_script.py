# author: Ahmed Rizk
# date: 2022-03-24

''' A script that sets a universal seed, reads the dataset used for the analysis
 from specified path, and processes the data for the EDA.

 Usage:

 Options:
 --read_path=<read_path>    Path of the file to read
 --out_path=<out_path>    path to the cleaned dataset object created
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
    # write to file data
    return processed    # should I split into 3?



def train_test_drop(data, dropped_col, train_path, test_path):
    X_train, Y_train, X_test, Y_test = split_drop(data, 0.3, 123, dropped_col)
    # write to file data/train and data/test
    return X_train, Y_train, X_test, Y_test


def create_pipeline(binary, category):
    binary_fea = binary
    cate_fea = category
    cate_trans = make_pipeline(OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans,cate_fea)
)
    return preprocessor
    
    
def create_reduced_dataframe(train, preprocessor):
    
    train_processed = preprocessor.fit_transform(train)
    



def main(read_path, out_path, processed_path, train_path, test_path):
    reduced_data = read_trim(read_path, ['EFSIZE', 'USHRWK', 'ATINC', 'HLEV2G', 'EFINVA', 'EFMJIE', 'EFATINC', 'EFMJSI'])
    processed = process(reduced_data)
    X_train, Y_train, X_test, Y_test = train_test_drop(processed, "EFINVA")
    preprocessor = create_pipeline("EFMJIE", "EFSIZE")
    reduced_dataframe = create_reduced_dataframe(X_train, preprocessor)
    
    
    

    
    

    reduced_dataframe.to_csv(out_path, index = False)
    
    
main(opt["--read_path"], opt["--out_path"], opt[["--cols_kept"]])


