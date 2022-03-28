# author: Ahmed Rizk
# date: 2022-03-25
#
''' A script that performs the train test split, finds the best k value, visualizes results and fits the best KNN model using the best K
Usage:

   Options:
 --processed=<processed>     Path (including filename) to processed data (which needs to be saved as a csv file)
 --out_dir=<out_dir> Path to directory where the model should be written
   
   ''' -> doc

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from src.analysis.KNN_tuning import KNN_tuning
from src.analysis import inv_outcome_plot
from src.analysis.split_drop import split_drop

from docopt import docopt

opt = docopt(__doc__)


def create_categ(processed, new_var, var):
    processed[new_var] = np.array(processed[var]) > processed[var].median()
    return processed

def train_test_drop(processed, drop_col):
    train_2, test_2 = train_test_split(processed, test_size = 0.3, random_state=123)
    X_train_2, Y_train_2, X_test_2, Y_test_2 = split_drop(processed, 0.3, 123, drop_col)
    return train_2, test_2, X_train_2, Y_train_2, X_test_2, Y_test_2  



def tune_model(train1, train2):
    param_grid = {"n_neighbours": np.arange(1,50,5)}
    results_df = KNN_tuning(preprocessor,train1,train2,param_grid) # preprocessor, which script will contain this?
    return results_df

    
    
def elbow_plot(results):
    elbow_plt = plt.plot(results['n_neighbours'], 
                    results_df['mean_cv_score'], 
                    '-0')
    plt.title('Figure 4: KNN K-tuning Results')
    plt.xlabel('K Neighbours')
    plt.ylabel('CV Accuracy')
    plt.show()
    
    
def make_pipeline(train1, train2, test1, test2):
    pipe_final = make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=26))
    pipe_final.fit(train1, train2)
    pipe_final.score(test1, test2)
    

def main(processed, out_dir):

    
    processed = create_categ(processed, 'EFINVA_Made_Money', 'EFINVA')
    train_2, test_2, X_train_2, Y_train_2, X_test_2, Y_test_2 = train_test_drop(processed, "EFINVA_Made_Money")
    results_df = tune_model(X_train_2,Y_train_2)
    elbow_plot(results_df)
    make_pipeline(X_train_2, Y_train_2, X_test_2, Y_test_2)
    
    

    
main(opt["--train"], opt["--out_dir"])


