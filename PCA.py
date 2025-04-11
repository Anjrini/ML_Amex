# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:49:01 2025

@author: anjrini
"""
import numpy as np
from sklearn.decomposition import PCA
import sklearn.linear_model as skl
import sklearn.model_selection as skm



def pca(x,y):
    pca1= PCA()
    linreg= skl.LinearRegression()
    v= skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
    mse=np.zeros(x.shape[1])
    xx=np.asarray(x)
    y1=np.asanyarray(y)
    for idx_train,idx_test in v.split(xx):
        xtrain=xx[idx_train,:]
        ytrain=y1[idx_train]
        xtest=xx[idx_test,:]
        ytest=y1[idx_test]
        x_train=(xtrain-np.mean(xtrain,0))/np.std(xtrain,0)
        x_test=(xtest-np.mean(xtrain,0))/np.std(xtrain,0)
        pca1.fit(x_train,ytrain)
        lm=linreg.fit(pca1.transform(x_train),ytrain)
        trans=pca1.transform(x_test)
        for i in range(1,x.shape[1]+1):
            yy=np.dot(trans[:,:i],lm.coef_[:i])+lm.intercept_
            mse[i-1]= np.mean((yy-ytest)**2)
            
    idx_=np.where(mse.min()==mse)[0][0]
    xs=(xx-np.mean(xx,0))/np.std(xx,0)
    y3=np.dot(pca1.transform(xs)[:,:idx_+1],lm.coef_[:idx_+1])+lm.intercept_
    
    return [y3,mse.min()]
