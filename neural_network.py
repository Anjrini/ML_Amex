# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:01:56 2025

@author: anjrini
"""

import numpy as np
import RNN_Preparation_Function
import sklearn.model_selection as skm
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset


from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
seed_everything(0,workers=True)
torch.use_deterministic_algorithms(True , warn_only=True)

from ISLP.torch import SimpleDataModule,SimpleModule,rec_num_workers,ErrorTracker


def f_rnn(data_frame,x_columns,y_column,lag):# x is the whole data frame and y is the column name in that data frame
    m=np.mean(data_frame[y_column])
    s=np.std(data_frame[y_column])
    df_n=(data_frame-np.mean(data_frame,0))/np.std(data_frame,0)
    x_n,y_n=RNN_Preparation_Function.AR_prep(df_n, x_columns, y_column, lag,True)
    
    in_=len(x_columns)
    
    class RNNmodel(nn.Module):
        def __init__(self):
            super(RNNmodel, self).__init__()
            
            self.rnn= nn.RNN(in_, in_*3,batch_first=True)
            self.drop=nn.Dropout(0.4)
            self.linear= nn.Linear(in_*3, 1)
            
        def forward(self,x):
            val,h_n=self.rnn(x)
            val= self.linear(self.drop(val[:,-1]))
            return torch.flatten(val)
            
    rnn_model=RNNmodel()
    
    x_train,x_test,y_train,y_test=skm.train_test_split(x_n,y_n,test_size=0.5,random_state=0)
    
    data=[]
    x_train=torch.tensor(x_train.astype(np.float32))
    y_train=torch.tensor(y_train.astype(np.float32))
    data.append(TensorDataset(x_train,y_train))
    x_test=torch.tensor(x_test.astype(np.float32))
    y_test=torch.tensor(y_test.astype(np.float32))
    data.append(TensorDataset(x_test,y_test))
    d_train,d_test=data
    
    max_rec_workers=rec_num_workers()
    rnn_dm=SimpleDataModule(d_train,d_test,batch_size=1
                                ,num_workers=min(2,max_rec_workers)
                                ,validation=d_test)
    
    
    
    opt= RMSprop(rnn_model.parameters(),lr=0.0001)
    rnn_module=SimpleModule.regression(rnn_model,optimizer=opt)
    
    rnn_trainer= Trainer(max_epochs=64,callbacks=[ErrorTracker()]
                         ,deterministic=True)
    
    rnn_trainer.fit(rnn_module,datamodule=rnn_dm)
    #rnn_trainer.test(rnn_module,datamodule=rnn_dm)
    
    rnn_model.eval()
    
    y_pred=rnn_model(x_test)*s+m
    y_pred=np.asarray(y_pred.detach())
    y_test=y_test*s+m
    y_test=np.asarray(y_test.detach())
    mse=np.mean((y_test-y_pred)**2)
    
    rnn_model.eval()
    xx=torch.tensor(x_n.astype(np.float32))
    y_pred=rnn_model(xx)*s+m
    y_pred=np.asarray(y_pred.detach())
    
    return (y_pred,mse)
