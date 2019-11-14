# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:22:32 2019

@author: win10
"""
import numpy as np

#compute sigmoid of the input
## sigmoid slope is assumed to be 1
def sigmoid_array(x):
    return 1/(1+np.exp(-x))

#compute differentation of sigmoid
def sigmoid_prime_array(x):
    return x*(1-x)

#N is batch size;
#D_in is input dimension;
#D_out is output dimension;
N, D_in,D_out = 4,2,1

#epoch defines the number of epochs rwquired;
epoch = 10000
#epoch = 50000
#epoch = 100000
#epoch = 500000

#create input vecotrs and expected ouptut label
x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype= np.float_)
y = np.array([[0],[1],[1],[1]],dtype=np.float_)

#Add bias input (-1) to all the instances
x = np.insert(x,x[0].size,-1,axis = 1)

#randomly initiialize weights--- normal distribution -- mean 0, std 1
w1 = np.random.randn(D_in + 1,D_out)
#w1 = np.random.default_rng().normal(0,1,(D_in + 1,D_out)) #Note '+!'

#set the learning rate (0<lr <=1)
learning_rate = 0.1 #experiment with this

#start the learning process
for t in range(epoch):
    #compute predited y
    y_pred = sigmoid_array(np.dot(x,w1))
    #compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t,loss)
    
    #compute gradients of w1 with respect to loss
    y_pred_err = y - y_pred
    grad_y_pred_sig = y_pred_err * sigmoid_prime_array(y_pred)
    grad_w1 = x.T.dot(grad_y_pred_sig)
    
    #updated weights
    w1 += learning_rate * grad_w1
    
#Let's examine the model
print(w1)  
print(y_pred)  

-----------------------------------BACKPROPAGATION ALGORITHM--------------------------------------------

import numpy as np

def sigmoid_array(x):
    return i/(1+np.exp(-x))

def sigmoid_prime_array(x):
    return x*(1-x)

N, D_in,H,D_out = 4,2,2,1

#epoch defines the number of epochs rwquired;
epoch = 10000
#epoch = 50000
#epoch = 100000
#epoch = 500000

#create input vecotrs and expected ouptut label
x = np.array([[0,0],[0,1],[1,0],[1,1]],dtype= np.float_)
y = np.array([[0],[1],[1],[0]],dtype=np.float_)

#Add bias input (-1) to all the instances
x_bias = np.insert(x,x[0].size,-1,axis = 1)

#randomly initiialize weights--- normal distribution -- mean 0, std 1
w1 = np.random.default_rng().normal(0,1,(D_in +1, H))
w2 = np.random.default_rng().normal(0,1,(H +1, H))
#w1 = np.random.default_rng().normal(0,1,(D_in + 1,D_out)) #Note '+!'

#set the learning rate (0<lr <=1)
learning_rate = 0.2 #experiment with this

for t in range(epoch):
    #Forward pass:
    h_sig = sigmoid_array(np.dot(x_bias,w1))
    #add bias to output of hidden layer
    h_sig_bias = np.insert(h_sig,h_sig[0].size,-1,axis=1)
    # compute prdicted value for y
    y_pred = sigmoid_array(np.dot(h_sig_bias,w2))
    #y_pred = np.dot(h_sig_bias,w2)
    
    #compute the loss
    loss = np.square(y_pred - y).sum()
    print(t,loss)
    
    #compute gradients of w1 with respect to loss
    y_pred_err = y - y_pred
    grad_y_pred_sig = y_pred_err * sigmoid_prime_array(y_pred)
    grad_w2 = h_sig_bias.T.dot(grad_y_pred_sig)
    
    grad_h_sig = grad_y_pred_Sig.dot(w2.T)* sigmoid_prime_array(y_pred)
    grad_h_sig_bias = np.delete(grad_h_sig,-1,axis =1)
    grad_w1 = x_bias.T.dot(grad_h_sig_bias)
    #updated weights
    w1 += learning_rate * grad_w1
    w2 += learning_rate * grad_w2
    
#Let's examine the model
print(w1)  
print(w2)
print(y_pred)  

------------------------------------------svm-------------------------------------------------------
from svmutil import *

train = [[-1,-1],[-1,1],[1,-1],[1,1]]
target = 
 model = svm_train(problem, param)

svm_save_model('xor.model',model)
xor_model = svm_load_model('xor.model')

test = [[-0.9,-0.82],[-0.79,0.92],[0.9,-0.9],[0.8,0.94]]

p_label,p_acc, p_vals = svm_predict(target,test,xor_model)

print()





















