# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:45:32 2025

@author: anjrini
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import sklearn.model_selection as skm

def pls(x,y):
    xx=np.asarray(x)
    y1=np.asarray(y)
    v=skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
    mse=np.zeros(x.shape[1])
    coefs=[0]*x.shape[1]
    inters=[0]*x.shape[1]
    for idx_train,idx_test in v.split(xx):
        xtrain=xx[idx_train,:]
        ytrain=y1[idx_train]
        xtest=xx[idx_test,:]
        ytest=y1[idx_test]
        x_train= (xtrain-np.mean(xtrain,0))/np.std(xtrain,0)
        x_test=  (xtest-np.mean(xtrain,0))/np.std(xtrain,0)
        
    #since the y is taken into account while computing the component we have to make
    #sure of the criterion in the presence of a small data set that the coefficients
    #are in line with the dimensions of the x_columns and length of the y vector
    
    z1=xtrain.shape[1]
    z2=len(ytrain)
    z3=0
    if z1>z2:
        mse=np.zeros(z2)
        coefs=[0]*z2
        inters=[0]*z2
        z3=z2
    else:
        mse=np.zeros(z1)
        coefs=[0]*z1
        inters=[0]*z1
        z3=z1
    
    for i in range(1,z3+1):
        pls1= PLSRegression(n_components=i)
        pls1.fit(x_train,ytrain)
        y4=np.dot(x_test,pls1.coef_.T)+pls1.intercept_
        coefs[i-1]=pls1.coef_
        inters[i-1]=pls1.intercept_
        mse[i-1]= np.mean((y4.T-ytest)**2)

    idx_=np.where(mse.min()==mse)[0][0]
    xs= (xx-np.mean(xx,0))/np.std(xx,0)
    r_y=np.dot(xs,coefs[idx_].T)+inters[idx_]
    
    return [r_y.T,mse.min()]
