# author: Ahmed Rizk
# date: 2022-03-24

''' A script that sets a universal seed, reads the dataset used for the analysis
 from specified path, and processes the data for the EDA.

 Usage:

 Options:
 --read_path=<read_path>    '''Path of the file to read'''
 --out_path=<out_path>    '''path to the cleaned dataset object created'''
 --cols_kept=<cols_kept>  '''array of columns to keep from dataset'''

''' 

import numpy as np
import pandas as pd
from docopt import docopt

np.random.seed(1)

opt = docopt(__doc__)

def main(read_path, out_path, cols_kept):

    data = pd.read_csv(read_path, header=None)
    first_five = data.head() # save
    reduced_data = data[cols_kept]
    reduced_info = reduced_data.info() # save
    reduced_describe = reduced_data.describe # save
    data.to_csv(out_path, index = False)
    reduced_data = reduced_data.drop(columns = 'USHRWK')  # too specific
    reduced_data = reduced_data.loc[reduced_data['ATINC'] != 99999996]  # too specific
    
    
    #### The previous section is up to "data splitting and processing"
    ### The following is from "data splitting and processing" till the first model
    
    processed = reduced_data[["EFINVA","EFSIZE","EFMJIE"]]
    binary_fea =["EFMJIE"]
    cate_fea = ["EFSIZE"]
    cate_trans = make_pipeline(OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans,cate_fea)
)
    train_processed = preprocessor.fit_transform(X_train)
    pd.DataFrame(train_processed, columns = ["EFMJIE","EFSIZE"])

    
main(opt["--read_path"], opt["--out_path"], opt[["--cols_kept"]])


