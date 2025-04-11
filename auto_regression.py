# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 15:49:53 2025

@author: anjrini
"""

import numpy as np
import sklearn.model_selection as skm
import sklearn.linear_model as skl

def lm_rnn(x,y):
    v=skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
    lm=skl.LinearRegression()
    xx=np.asarray(x)
    y1= np.asarray(y)
    
    for idx_train,idx_test in v.split(xx):
        xtrain=xx[idx_train,:]
        ytrain=y1[idx_train]
        xtest=xx[idx_test,:]
        ytest=y1[idx_test]
        lm.fit(xtrain,ytrain)
        yy=lm.predict(xtest)
        mse=np.mean((yy-ytest)**2)
   
    return [lm.predict(xx),mse]
