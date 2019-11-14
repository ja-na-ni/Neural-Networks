# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:37:14 2019

@author: Nithin
"""

import pandas as pd
import numpy as np 


z=pd.read_excel('D:\\dataset\\z-score.xlsx')

z['x1']=z['Capital']/z['Asset']
z['x2']=z['Retained']/z['Asset']
z['x3']=z['EBIT']/z['Asset']
z['x4']=z['Market']/z['Debt']
z['x5']=z['Sales']/z['Asset']


for i in range(len(z)):
    z['score'][i]= (1.2 * z['x1'][i] ) + (1.4 * z['x2'][i]) + (3.3 * z['x1'][i]) + (0.6 * z['x1'][i]) + (0.999 * z['x1'][i])
    
#adding bias
z['bias']=-1

z['result']=z['score'] > 1.80
z['result']=z['result'].map( {True:1,False:0}) 

z_new=z.drop(['year','Asset','Capital','Retained','EBIT','Debt','Market','Sales'],axis=1)

n=z_new.values
n[0].size
n=np.insert(n,n[0].size, -1, axis=1)

y=z.loc[:,'result'].values.reshape(-1,1)

N, D_in , D_out = 8,5,1

np.random.seed()
w1=np.random.normal(0,1,(D_in + 1, D_out ))


def sig_array(x):
    return 1/ (1 + np.exp(-x))

#compute differentiation of sigmoid

def sigmoid_array(x):
    return x*(1-x)

# N is batch size
# D_in is input dimesnison
# D_out is output dimension



# epoch defines the number of epochs required

epoch=20000


learning_rate=0.1

for t in range(epoch):
    #compute predicted y
    y_pred = sig_array(np.dot(n,w1))
    
    #compute an print loss
    loss=np.square(y_pred-y).sum()
    print(t,loss)

    #compute gradients of w1 with respect to loss
    y_pred_err= y-y_pred
    grad_y_pred_sig= y_pred_err * sigmoid_array(y_pred)
    grad_w1= n.T.dot(grad_y_pred_sig)
    
    #update weights
    w1 += learning_rate * grad_w1

#examine the model

print(w1)
print(y_pred)





