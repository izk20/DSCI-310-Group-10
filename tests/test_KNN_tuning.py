import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.analysis.KNN_tuning import KNN_tuning

def test_KNN_fullfuc():
    results_dict = {
        "n_neighbours": [],
        "mean_train_score": [],
        "mean_cv_score": []
    }
    param = {"n_neighbours": [1,2,3]}
    dat = pd.DataFrame({
        'x': [12,23,34,45,65,56,12,23,34,45,65,56],
        'x2': [2, 3, 4, 5, 6, 7,2, 3, 4, 5, 6, 7],
        'x3': [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
        'x34': [2, 3, 4, 5, 6, 7,4,5,6,7,8,9],
        'x35': [2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9],
        'y': [1,2,1,2,1,2,1,2,1,2,1,2]

    })
    train, test = train_test_split(dat, test_size=.2, random_state=123)
    train_x,train_y = train.drop(columns='y'), train['y']
    for k in param["n_neighbours"]:
        knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
        scores = cross_validate(knn, train_x, train_y, return_train_score=True)
        results_dict["n_neighbours"].append(k)
        results_dict["mean_train_score"].append(np.mean(scores['train_score']))
        results_dict["mean_cv_score"].append(np.mean(scores['test_score']))
    data = pd.DataFrame(results_dict)
    df2 = KNN_tuning(StandardScaler(),train_x,train_y,param)
    assert_frame_equal(data, df2, check_dtype=False)

def test_KNN_trainx():
    results_dict = {
        "n_neighbours": [],
        "mean_train_score": [],
        "mean_cv_score": []
    }
    param = {"n_neighbours": [1, 2, 3]}
    dat = pd.DataFrame({
        'x': [12, 23, 34, 45, 65, 56, 12, 23, 34, 45, 65, 56],
        'x2': [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
        'x3': [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
        'x34': [2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9],
        'x35': [2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9],
        'y': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    })
    train, test = train_test_split(dat, test_size=.2, random_state=123)
    train_x, train_y = train.drop(columns='y'), train['y']
    train_x = "teo"
    with pytest.raises(TypeError) as e_info:
        KNN_tuning(StandardScaler(), train_x, train_y, param)
    assert "train_x should be data frame" in str(e_info.value)

def test_KNN_trainy():
    results_dict = {
        "n_neighbours": [],
        "mean_train_score": [],
        "mean_cv_score": []
    }
    param = {"n_neighbours": [1, 2, 3]}
    dat = pd.DataFrame({
        'x': [12, 23, 34, 45, 65, 56, 12, 23, 34, 45, 65, 56],
        'x2': [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
        'x3': [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7],
        'x34': [2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9],
        'x35': [2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9],
        'y': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    })
    train, test = train_test_split(dat, test_size=.2, random_state=123)
    train_x, train_y = train.drop(columns='y'), train['y']
    train_y = [1,2,3,4,5,6,7,8,9]
    with pytest.raises(TypeError) as e_info:
        KNN_tuning(StandardScaler(), train_x, train_y, param)
    assert "train_y should be dataSeries" in str(e_info.value)