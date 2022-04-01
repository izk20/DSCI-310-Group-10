# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/read_process_script.py --xtrainpath=<xtrainpath> --ytrainpath=<ytrainpath> --variables=sss<variables>
python src/read_process_script.py --xtrainpath='' --ytrainpath=<ytrainpath> --variables=<variables>

Options:
--xtrainpath=<xtrainpath>:    csv file previously saved in the previous script the training data for the x-axis of ridge regression
--ytrainpath=<ytrainpath>:     csv file previously saved int the previous script the training data for the y-axis of ridge regression
--variables=<variables>:  A list of variables which contains both binary and categories feature of processor.
"""

import pandas as pd
from docopt import docopt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from analysis.alpha_tuning import ridge_alpha_tuning
import pickle

opt = docopt(__doc__)

def make_processor(binary_fea, cate_fea):
    cate_trans = make_pipeline(OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans, cate_fea)
    )
    return preprocessor

def pickle_save(name, variable):
    with open(name,"wb") as f:
        pickle.dump(variable, f)


def main(xtrainpath, ytrainpath, variables):
    variables = [x for x in variables.split(',')]
    ytrain = pd.read_csv(ytrainpath)
    xtrain = pd.read_csv(xtrainpath)
    ytrain = ytrain.squeeze()
    preprocessor = make_processor([variables[0]],[variables[1]])
    train_processed = preprocessor.fit_transform(xtrain)
    alphas = list(10.0 ** np.arange(-2, 5, 1))
    best_alpha = ridge_alpha_tuning(alphas, preprocessor, xtrain, ytrain)
    with open("best_alpha","wb") as f:
        pickle.dump(best_alpha, f)
    with open("trained_preprocessor","wb") as f:
        pickle.dump(train_processed, f)
    with open("preprocessor", "wb") as f:
        pickle.dump(preprocessor, f)


if __name__ == "__main__":
    main(opt["--xtrainpath"], opt["--ytrainpath"],opt["--variables"])
