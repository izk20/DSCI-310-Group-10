from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.analysis.alpha_tuning import ridge_alpha_tuning
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV

def test_ridgealphatuning_1():
    #Make preprocessor
    #1. Manually test
        #make pipeline
        #fit model
        #Determine alpha
    #2Run ridge model
        #pass the same items into ridge_alpha tuning
    #3. checking if they are equal
        #assert if 4 equals (1,2,3)

    alpha = [1,10,100]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4,6,7,8,9,0],
        'x2': [1, 2, 3, 4, 5, 6 , 7,8,10],
        'y': [2, 3, 4, 5,6 ,7 ,7, 8, 9]
    })
    train, test = train_test_split(toy_dataset,test_size=.4,random_state=123)
    trainx, trainy = train.drop(columns = 'y'),train['y']
    # pip = Pipeline(steps= [('passthrough')])
    cv_pipe =  make_pipeline(StandardScaler(), RidgeCV(alphas=alpha, cv=2))
    cv_pipe.fit(trainx,trainy)
    best_a = cv_pipe.named_steps['ridgecv'].alpha_

    assert ridge_alpha_tuning(alpha,StandardScaler(),trainx,trainy,cv=2) == best_a

def test_ridgealphatuning_2():
    alpha = [1, 10, 100]
    toy_dataset = DataFrame({
        'x1': [1, 2, 3, 4, 6, 7, 8, 9, 0],
        'x2': [1, 2, 3, 4, 5, 6, 7, 8, 10],
        'y': [2, 3, 4, 5, 6, 7, 7, 8, 9]
    })
    train, test = train_test_split(toy_dataset, test_size=.4, random_state=123)
    trainx, trainy = train.drop(columns='y'), train['y']

    ridge_alpha_tuning()