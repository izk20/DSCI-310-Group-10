# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/hyperparameter_optimization.py --xtrain=<xtrain> --ytrain=<ytrain>

Options:
--xtrain=<xtrain>:    csv file previously saved in the previous script the training data for the x-axis of ridge regression
--ytrain=<ytrain>:     csv file previously saved int the previous script the training data for the y-axis of ridge regression
"""

from docopt import docopt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import src.hyperparameter_optimization as ho

opt = docopt(__doc__)


def ridge_pipline(processor,alpha):
    # This function is a helper function to create a specific case for pipline creating a ridge pipline
    # Parameter:
    #   --processor:
    ridge_pipeline = make_pipeline(processor, Ridge(alpha=alpha))
    return ridge_pipeline

def cross_validation(ridgepip,xtrain,ytrain):
    cross_validate(ridgepip, xtrain, ytrain, cv=10, return_train_score=True)

def write_csv(pd,out_dir):
    pd.to_csv(out_dir, index=True)


def main(xtrain, ytrain):
    ridge_pipeline = make_pipeline(ho.preprocessor, Ridge(alpha=ho.best_alpha))
    cv_ridge = pd.DataFrame(cross_validate(ridge_pipeline, xtrain, ytrain, cv=10, return_train_score=True))
    write_csv(cv_ridge,"../data/cv_ridge.csv")


if __name__ == "__main__":
    main(opt["--xtrain"], opt["--ytrain"])
