# author: Harry Zhang
# date: 2022-03-25

"""
Usage: src/down_data.py --preprocessor=<preprocessor> --trainx=<xtrain> --trainy=<ytrain>
Options:
--preprocessor=<preprocessor>   The preprocessor which generated in previous scirpt
--trainx=<xtrain>                The splited X-train dataframe
--trainy=<ytrain>               The splited Y_train dataframe
""" -> __doc__

from docopt import docopt
import requests
import os
import pandas as pd
import feather

opt = docopt(__doc__)

def main(preprocessor, xtrain, out_file):
  alphas = list(10.0 ** np.arange(-2, 5, 1))
  best_alpha = ridge_alpha_tuning(alphas,preprocessor,X_train,Y_train)
  display('The best alpha from ridge hyperparameter tuning is:', best_alpha)
  ridge_pipeline = make_pipeline(preprocessor, Ridge(alpha=best_alpha))
  cv_ridge = pd.DataFrame(cross_validate(ridge_pipeline, X_train, Y_train, cv=10, return_train_score=True))

if __name__ == "__main__":
  main(opt["--preprocessor"], opt["--trainx"], opt["--trainy"])
