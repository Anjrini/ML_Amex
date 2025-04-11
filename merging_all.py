# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:09:08 2025

@author: anjrini
"""

#importing the required libraries
import linear
import RNN_Preparation_Function as RNN
import pandas as pd
import numpy as np
import lasso
import ridge
import PCA
import pls
import auto_regression
import neural_network

def m_all(data_frame,x_columns,y_column,lag):
    #getting the required data frame with the lagging value
    x,y=RNN.AR_prep(data_frame, x_columns, y_column, lag,False)
    
    #Matrix of the calculations
    m=np.zeros((data_frame.shape[0]+1,7))
    m=pd.DataFrame(m)
    m.columns=["Linear","Lasso","Ridge","PCA","PLS","Auto Regression","Neural Network"]
    x_label=["MSE","Jan","Feb","Mar","April","May","Jun","Jul","Aug","Sept","Okt"
             ,"Nov","Dec"]
    m.index=x_label
    
    #calculating the MSE and y hat using the normal linear model
    m.iloc[1:,0],m.iloc[0,0]=linear.lm(data_frame, y_column)
    
    #calculating the MSE and y hat using the lasso model
    m.iloc[lag+1:,1],m.iloc[0,1]=lasso.lasso(x, y)
    
    #calculating the MSE and y hat using the Ridge model
    m.iloc[lag+1:,2],m.iloc[0,2]=ridge.ridge(x, y)
    
    #calculating the MSE and y hat using the PCA model
    m.iloc[lag+1:,3],m.iloc[0,3]=PCA.pca(x, y)
    
    #calculating the MSE and y hat using the PLS model
    m.iloc[lag+1:,4],m.iloc[0,4]= pls.pls(x, y)
    
    #calculating the MSE and y hat using the Auto Regression model
    m.iloc[lag+1:,5],m.iloc[0,5]= auto_regression.lm_rnn(x,y)
    
    #calculating the MSE and y hat using the Nueral Network Model
    m.iloc[lag+1:,6],m.iloc[0,6]=neural_network.f_rnn(data_frame, x_columns, y_column, lag)
    
    #Summary of the MSE 
    MSEs=m.loc["MSE"].sort_values()
    
    return [MSEs,m]