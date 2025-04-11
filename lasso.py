# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 15:48:36 2025

@author: anjrini
"""

import pandas as pd
import numpy as np
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import sklearn.model_selection as skm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


lasso=skl.ElasticNet(l1_ratio=1)
v=skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
scaler= StandardScaler()
pipe=Pipeline([("scaler",scaler),("lasso",lasso)])

def lasso(x,y):
    lamx= 10**(np.linspace(-2, 8,num=50))/np.std(y)

    grid=skm.GridSearchCV(pipe, {"lasso__alpha":lamx},cv=v
                 ,scoring="neg_mean_squared_error")
    grid.fit(x,y)
    mse=(-grid.cv_results_["mean_test_score"]).min()
    z=[grid.best_estimator_.predict(x),mse]
    
    return z