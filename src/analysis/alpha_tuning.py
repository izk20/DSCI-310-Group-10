# Importing the appropriate packages
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV


def ridge_alpha_tuning(alpha, processor, trainx, trainy):
    ridge_cv_pipe = make_pipeline(processor, RidgeCV(alphas=alpha, cv=10))
    ridge_cv_pipe.fit(trainx, trainy)
    best_alpha = ridge_cv_pipe.named_steps["ridgecv"].alpha_
    return best_alpha
