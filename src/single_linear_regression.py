# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/hyperparameter_optimization.py --xtrain=<xtrainpath> --ytrain=<ytrainpath>

Options:
--xtrain=<xtrainpath>:    csv file previously saved in the previous script the training data for the x-axis of ridge regression
--ytrain=<ytrainpath>:     csv file previously saved int the previous script the training data for the y-axis of ridge regression
"""

from docopt import docopt
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

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

def make_plot(cv_ridge):
    ridge = plt.plot(np.arange(len(cv_ridge)),
                     cv_ridge['test_score'],
                     '-0')
    plt.title('Figure 3: RidgeCV Folds = 10')
    plt.xlabel('CV Fold Iterations')
    plt.ylabel('CV Accuracy')
    return ridge

def main(xtrainpath, ytrainpath):
    xtrain = pd.read_csv(xtrainpath)
    ytrain = pd.read_csv(ytrainpath)
    preprocessor = pickle.load(open("results/preprocessor.pickle", "rb"))
    best_alpha = pd.read_csv("results/best_alpha.csv")
    ridge_pipeline = make_pipeline(preprocessor, Ridge(alpha=best_alpha))
    cv_ridge = pd.DataFrame(cross_validate(ridge_pipeline, xtrain, ytrain, cv=10, return_train_score=True))
    write_csv(cv_ridge,"../data/cv_ridge.csv")
    cv_plot = make_plot(cv_ridge)
    cv_plot.savefig('cv_plot.png')


if __name__ == "__main__":
    main(opt["--xtrain"], opt["--ytrain"])
