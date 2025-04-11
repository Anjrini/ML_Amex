# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:20:43 2025

@author: anjrini
"""
import numpy as np
from matplotlib.pyplot import subplots


def plotting(df,var,x_pred,y_pred,method):

    x_label=["Jan","Feb","Mar","April","May","Jun","Jul","Aug","Sept","Okt"
       ,"Nov","Dec"]
    
    x=np.arange(len(x_label))
    w=0.33
    
    fig,ax= subplots(figsize=(8,6))
    rect1=ax.bar(x-w,df.iloc[:,0],label="2023",width=w)
    rect2=ax.bar(x,df.iloc[:,1],label="2024",width=w)
    rect3=ax.bar(x+w,df.iloc[:,2],label="2025",width=w)
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(x_label)
    ax.set_xlabel("Month")
    ax.set_ylabel("Expenditure")
    ax.set_title(var.capitalize()+" Expenditure per Year")
    fig
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height())
            if height==0:
                next
            else:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2.5),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rect1)
    autolabel(rect2)
    autolabel(rect3)

    ax.plot(x_pred,y_pred,label=method,c="black",linewidth = '3')
    ax.legend()
    return fig


