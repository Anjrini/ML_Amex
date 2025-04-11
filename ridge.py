# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 11:29:37 2025

@author: anjrini
"""


import numpy as np
import sklearn.model_selection as skm
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def ridge(x,y):
    ridge=skl.ElasticNet(l1_ratio=0,max_iter=60000)
    v=skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
    scaler= StandardScaler()
    pipe=Pipeline([("scaler",scaler),("ridge",ridge)])
    
    lamx= 10**(np.linspace(-2, 8,num=50))/np.std(y)

    grid=skm.GridSearchCV(pipe, {"ridge__alpha":lamx},cv=v
                 ,scoring="neg_mean_squared_error")
    grid.fit(x,y)
    mse=(-grid.cv_results_["mean_test_score"]).min()
    z=[grid.best_estimator_.predict(x),mse]
    
    return z