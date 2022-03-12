from pandas import DataFrame
from sklearn.model_selection import train_test_split
import pytest
from src.analysis.alpha_tuning import ridge_alpha_tuning
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV

def test_ridgealphatuning_fullfunc():
    alpha = [1, 5, 12]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']
    cv_pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alpha, cv=2))
    cv_pipe.fit(trainx, trainy)
    best_a = cv_pipe.named_steps['ridgecv'].alpha_
    print(best_a)
    assert ridge_alpha_tuning(alpha, StandardScaler(), trainx, trainy, cv=2) == best_a

def test_ridgealphatuning_alpha():
    alpha = 1
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']
    with pytest.raises(TypeError) as e_info:
        ridge_alpha_tuning(alpha, StandardScaler(), trainx, trainy, cv=2)
    assert "alpha is not a list" in str(e_info.value)

def test_ridgealphatuning_trainx():
    alpha = [1, 10, 100]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']
    trainx = 1
    with pytest.raises(TypeError) as e_info:
        ridge_alpha_tuning(alpha, StandardScaler(), trainx, trainy, cv=2)
    assert "train_x should be data frame" in str(e_info.value)

def test_ridgealphatuning_trainy():
    alpha = [1, 10, 100]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']
    trainy = 1213
    with pytest.raises(TypeError) as e_info:
        ridge_alpha_tuning(alpha, StandardScaler(), trainx, trainy, cv=2)
    assert "train_y should be data frame" in str(e_info.value)

def test_ridgealphatuning_cv():
    alpha = [1, 10, 100]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']
    with pytest.raises(TypeError) as e_info:
        ridge_alpha_tuning(alpha, StandardScaler(), trainx, trainy, cv="two")
    assert "cv should be an integer" in str(e_info.value)