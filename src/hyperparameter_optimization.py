# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/read_process_script.py --xtrain=<xtrain> --ytrain=<ytrain> --variables=<variables>

Options:
--xtrain=<xtrain>:    csv file previously saved in the previous script the training data for the x-axis of ridge regression
--ytrain=<ytrain>:     csv file previously saved int the previous script the training data for the y-axis of ridge regression
--variables = <variables>:  A list of variables which contains both binary and categories feature of processor.
"""
import pandas as pd
from docopt import docopt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from analysis.alpha_tuning import ridge_alpha_tuning

opt = docopt(__doc__)

def make_processor(binary_fea, cate_fea):
    cate_trans = make_pipeline(OrdinalEncoder(categories=[[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans, cate_fea))
    return preprocessor


def main(xtrainpath, ytrainpath, variables):
    xtrain = pd.read_csv(xtrainpath)
    ytrain = pd.read_csv(ytrainpath)
    # raise ImportError(ytrain)
    preprocessor = make_processor(variables[0], variables[1])
    # raise ImportError(variables)
    train_processed = preprocessor.fit_transform(xtrain)
    alphas = list(10.0 ** np.arange(-2, 5, 1))
    best_alpha = ridge_alpha_tuning(alphas, preprocessor, xtrain, ytrain)
    print(best_alpha)
    train_processed.to_csv(train_processed, "train_processed")
    return best_alpha,preprocessor


if __name__ == "__main__":
    main(opt["--xtrain"], opt["--ytrain"],opt["--variables"])
