# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:46:49 2025

@author: anjrini
"""
import pandas as pd
import numpy as np
import sklearn.model_selection as skm
import sklearn.linear_model as skl

def lm(df,y_column):
    check=df.apply(lambda x: np.sum(x==0))
    idx_c=df.columns[np.where(check>0)][0]
    y_year=y_column
    y=df[y_year]
    df=df.drop(columns=[idx_c,y_year])
    
    
    #df_lm=df_lm.rename(columns={0:"v0",1:"v1",2:"v2",3:"v3",4:"v4",5:"y"})
    lm3=skl.LinearRegression()
    
    valid=skm.ShuffleSplit(n_splits=1,test_size=0.5,random_state=0)
    for (idx_train,idx_test) in valid.split(df):
        lm3.fit(df.iloc[idx_train,:],y[idx_train])
        p3=lm3.predict(df.iloc[idx_test,:])
        mse=np.mean((p3-y[idx_test])**2)
    
    z=[lm3.predict(df),mse]
    return z


