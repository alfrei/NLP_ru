from sklearn import metrics
import numpy as np


def fit_model(X_train,X_test,y_train,y_test,model):
    """ 
    Fit model & predict probabilities
    Return hard prediction if probabilities not allowed 
    """
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print('-'*60)
    print(model.__class__)
    print('acc:%s' % metrics.accuracy_score(y_test, predict))
    print(metrics.confusion_matrix(y_test, predict))
    if 'predict_proba' in dir(model): predict = model.predict_proba(X_test)
    return predict


def max_index(arr):
    """ return first index (or axis for 2D array) of maximum value """
    return np.where(arr == np.max(arr))[0][0]

