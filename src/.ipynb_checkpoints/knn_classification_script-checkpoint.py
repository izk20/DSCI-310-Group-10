# author: Ahmed Rizk
# date: 2022-03-25
#
# A script that performs the train test split, finds the best k value, visualizes results and fits the best KNN model
# usin the best K
#
# Usage:

# Options:
# --train=<train>     Path (including filename) to training data (which needs to be saved as a csv file)
# --out_dir=<out_dir> Path to directory where the model should be written
" -> doc

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

def main(train, out_dir):

   
    
main(opt["--train"], opt["--out_dir"])


