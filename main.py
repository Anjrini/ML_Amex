# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:49:52 2025

@author: Mustafa anjrini
"""

import os
#please choose the path where the files are stored on your computer
#between the quotations below
os.chdir("C:/Users/anjrini/Desktop/p/amex2")
import pandas as pd
import numpy as np
import preparation
import plotting
import merging_all


#reading the excel file
df=pd.read_excel("data.xlsx")

# chossing the company of intereset or choosing all for the whole data
var="lidl" #please write the name of the company or "all" for the whole costs

#creating the data frame from the previous inputs
df=preparation.prep(df, var)

#Calculating the MSEs as well as y hats

df# getting the predictors as well as the data avilable

# selecting the predictors of interest in this case the years for the x columns
# please select the columns of predictors
x_columns= ["2023","2024"] #always as list

#selecting the response of interest
# please select the column of the response
y_column="2024"

#selecting the lagging time
# please select the lagging time in months
lag=2

#Calculating the MSEs as well as y hats
MSEs,y_preds=merging_all.m_all(df,x_columns, y_column, lag)

#checking the mean squared errors for the methods implemented
#methods implemented: Ridge, Lasso, Linear,Neural Network, PCA, PLS, Auto Regression
MSEs

#choosing a method to get the y hat
#methods to choose from are: 
#    Ridge, Lasso, Linear,Neural Network, PCA, PLS, Auto Regression
    
Method= "Ridge" # please write the name of the method between the quotation marks

i=np.where(y_preds.columns==Method)[0] #getting the index of the method
ii=np.where(df.columns==y_column)[0] # getting the index of the y column

#showing the results
if Method!="Linear":
    #showing the resulted y hat from the method chosen
    print("the below shows the predicted y hat")
    y_pred=y_preds.iloc[lag+1:,i] 
    print(y_pred)
    #comparison with the y values
    print("\nthe below shows the actual y values")
    print(df.iloc[lag:,ii]) #showing the resulted y values
   
else:
    #showing the resulted y hat from the method chosen
    print("the above shows the predicted y hat")
    y_pred=y_preds.iloc[1:,i] 
    print(y_pred)
    #comparison with the y values
    print("\nthe above shows the actual y values")
    print(df.iloc[:,ii]) #showing the resulted y values
    

#plotting the results
if Method!="Linear":
    x=np.arange(lag,y_pred.shape[0]+lag)
    plotting.plotting(df, var, x,y_pred,Method)
else:
    x=np.arange(12)
    plotting.plotting(df, var, x,y_pred,Method)

