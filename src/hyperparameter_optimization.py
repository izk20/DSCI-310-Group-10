# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/read_process_script.py --data=<reduced_data>
Options:
--data<reduced_data>   The data set which previously processed before.
"""

from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  OneHotEncoder,OrdinalEncoder
from sklearn.compose import make_column_transformer
from analysis.alpha_tuning import ridge_alpha_tuning
from analysis.split_drop import split_drop

opt = docopt(__doc__)
def main(reduced_data):
    processed = reduced_data[["EFINVA","EFSIZE","EFMJIE"]]
    X_train, Y_train, X_test, Y_test = split_drop(processed, 0.3, 123, "EFINVA")
    binary_fea =["EFMJIE"]
    cate_fea = ["EFSIZE"]
    cate_trans = make_pipeline(OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans,cate_fea))
    train_processed = preprocessor.fit_transform(X_train)
    alphas = list(10.0 ** np.arange(-2, 5, 1))
    best_alpha = ridge_alpha_tuning(alphas,preprocessor,X_train,Y_train)
    print(best_alpha)
    train_processed.to_csv('raw_data.csv', index=False)

if __name__ == "__main__":
  main(opt["--data"])
