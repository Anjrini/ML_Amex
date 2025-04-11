# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:34:52 2024

@author: Mustafa Anjrini
"""
#impoting two libraries Numpy and Pandas for the implementation
import numpy as np
import pandas as pd


def AR_prep(data_frame,x_col_names,y_col_name,lag,RNN=True):
    
    ll=list(x_col_names)
    #recreating the data frame containing only the columns related
    df= data_frame.reindex(columns=ll)
    
    #creating a new data frame for the implementation as well as
    #the position of the response
    df_new= np.zeros((df.shape[0]-lag,df.shape[1]*lag))
    n=np.where(data_frame.columns==y_col_name)[0][0]

    for i in range(df.shape[1]):
        for j in range(1,lag+1):
            df_new[:,j+i*lag-1]=df.iloc[j-1:-lag+j-1,i]
    
    #if RNN is the target, then this code will be executed
    if RNN==True:
        df_new_rnn=df_new.reshape((df_new.shape[0],df.shape[1],lag))        
        df_new_rnn=np.transpose(df_new_rnn,axes=[0,2,1])
        
        #creaing the new response vector
        y1= np.array(data_frame.iloc[lag:,n])
        
        #creating a list of the results(new data frame and response)
        z=list([df_new_rnn,y1])
    else:
        # if the implementation of the Auto Regression is the target, then
        # this code will be executed
        #creating the name of the columns of the new data frame
        hh=[]
        for j in df.columns:
            for i in np.arange(1,lag+1):   
                hh.append("{0}_{1}".format(j,i))
        
        #creaing the new response vector
        y1= data_frame.iloc[lag:,n]
        df_new= pd.DataFrame(df_new,columns=hh,index=y1.index)
        
        #creating a list of the results(new data frame and response)
        z=list([df_new,y1])
    return z

