# author: Ahmed Rizk
# date: 2022-03-25
#
''' A script that performs the train test split, finds the best k value, visualizes results and fits the best KNN model using the best K

   Usage: knn_classification_script.py --processed=<processed> --out_dir=<out_dir>

   Options:
 --processed=<processed>     Path (including filename) to processed data (which needs to be saved as a csv file)
 --out_dir=<out_dir> Path to directory where the model should be written
   
   '''

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from analysis.KNN_tuning import KNN_tuning
from analysis import inv_outcome_plot
from analysis.split_drop import split_drop

from docopt import docopt

opt = docopt(__doc__)


def create_categ(processed, new_var, var):
    processed = pd.read_csv(processed)
    processed[new_var] = np.array(processed[var]) > processed[var].median()
    return processed

def train_test_drop(processed, drop_col):
    train_2, test_2 = train_test_split(processed, test_size = 0.3, random_state=123)
    X_train_2, Y_train_2, X_test_2, Y_test_2 = split_drop(processed, 0.3, 123, drop_col)
    return train_2, test_2, X_train_2, Y_train_2, X_test_2, Y_test_2  


def create_preprocessor(binary, category):
    X_train = pd.read_csv("../data/Training/X_train.csv")
    binary_fea = [binary]
    cate_fea = [category]
    cate_trans = make_pipeline(OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7]], dtype=int))
    binary_trans = make_pipeline(OneHotEncoder(drop="if_binary"))
    preprocessor = make_column_transformer(
        (binary_trans, binary_fea),
        (cate_trans,cate_fea)
)
    train_processed = preprocessor.fit_transform(X_train)
    pd.DataFrame(train_processed, columns = ["EFMJIE","EFSIZE"])
    return preprocessor



def tune_model(train1, train2, preprocessor):
    param_grid = {"n_neighbours": np.arange(1,50,5)}
    results_df = KNN_tuning(preprocessor,train1,train2,param_grid)
    return results_df

    
    
def elbow_plot(results, path, filename):
    elbow_plt = plt.plot(results['n_neighbours'], 
                    results['mean_cv_score'], 
                    '-0')
    plt.title('Figure 4: KNN K-tuning Results')
    plt.xlabel('K Neighbours')
    plt.ylabel('CV Accuracy')
    plt.savefig(path + filename)
    
    
def create_pipeline(train1, train2, test1, test2, preprocessor):
    pipe_final = make_pipeline(preprocessor, KNeighborsClassifier(n_neighbors=26))
    pipe_final.fit(train1, train2)
    pipe_final.score(test1, test2)
    
    
def conf_mat(pipeline, data1, data2, path, filename):
    plot_confusion_matrix(pipeline, data1, data2, cmap=plt.cm.Blues, colorbar=False)
    plt.savefig(path + filename)


    

def main(processed, out_dir):
    print("1")
    processed = create_categ(processed, "EFINVA_Made_Money", "EFINVA")
    print("2")
    train_2, test_2, X_train_2, Y_train_2, X_test_2, Y_test_2 = train_test_drop(processed, "EFINVA_Made_Money")
    print("2.5")
    preprocessor_x = create_preprocessor("EFMJIE", "EFSIZE")
    print("3")
    results_df = tune_model(X_train_2, Y_train_2, preprocessor_x)
    #to write 
    with open("results_df", "wb") as f:
        pickle.dump(results_df, f) 
    print("4")
    # elbow_plot(results_df, out_dir, "elbow_plot.png")
    print("5")
    pipe_final = create_pipeline(X_train_2, Y_train_2, X_test_2, Y_test_2, preprocessor_x)
    print("6")
    conf_mat(pipe_final, X_test_2, Y_test_2, out_dir, "conf_mat.png")
    print("7")
    
main(opt["--processed"], opt["--out_dir"])


