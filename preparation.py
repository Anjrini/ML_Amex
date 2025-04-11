# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:57:12 2025

@author: anjrini
"""
import re as re
import pandas as pd
import numpy as np

def prep(df,var):
    
    
    x_label=["Jan","Feb","Mar","April","May","Jun","Jul","Aug","Sept","Okt"
             ,"Nov","Dec"]
    all_j=pd.unique(df["Datum"].apply(lambda x: x[-4:]))
    all_j=sorted(all_j)
    col=len(all_j)
    
    v1=np.zeros((col,12)) 
    df_new=[0]*12
    for j in range(col):
        jahr=all_j[j]  
        for i in range(1,13):
            if i<10:
                df_new[i-1]=df.loc[df["Datum"].apply(lambda x: bool(re.search(str(0)+str(i)+"/"+jahr, x))),:]
            else:
                df_new[i-1]=df.loc[df["Datum"].apply(lambda x: bool(re.search(str(i)+"/"+jahr, x))),:]
        
        if var=="all":    
            #all costs in the specified year
            for i in range(12):
                v1[j,i]=np.sum(df_new[i].loc[df_new[i]["Betrag"]>0,"Betrag"])
        else:
            #specific cost in the specified year
            for i in range(12):
                v1[j,i]=np.sum(df_new[i].loc[df_new[i]["Beschreibung"].apply(lambda x: bool(re.search(var.upper(),x))),"Betrag"])
    
    r=pd.DataFrame(v1.T,index=x_label,columns=all_j)
    return r