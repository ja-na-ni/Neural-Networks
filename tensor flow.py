# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:43:29 2019

@author: win10
"""

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.__version__
print(tf.__version__)
print(tf.version.VERSION)
a = tf.constant("hello")
a
b = tf.constant("world")

with tf.Session() as sess:
    res = sess.run(a+b)
print(res)

myones = tf.ones((4,4))
sess = tf.InteractiveSession()
sess.run(myones)

rand_dist = tf.random_normal((4,4),mean=0,stddev = 1.0)
rand = sess.run(rand_dist)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a+b

sess.run(add_op, feed_dict= {a:10,b:20})

#simple linear regression
import numpy as np
x_data = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
y_data = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)

import matplotlib.pyplot as plt
plt.plot(x_data,y_data,'*')

m = tf.Variable(0.85)
b = tf.Variable(0.4)

y_hat = m*x_data+b
error = tf.reduce_mean(tf.abs(y_data - y_hat))

#error = tf.reduce_mean((y_data - y_hat)**2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

train = optimizer.minimize(error)
init = tf.global_variables_initializer()
sess.run(init)

epochs = 50

for i in range(epochs):
    sess.run(train)
    
slope,intercept = sess.run([m,b])
print(slope,intercept)

x_test = np.linspace(-1,11,10)

y_pred = slope*x_test + intercept

plt.plot(x_test,y_pred, 'r')
plt.plot(x_data,y_data,'*')
---------------------------------------------------------------------------------
import pandas as pd
data = pd.read_csv('Churn_Modelling.csv')    
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x[:,1]= label.fit_transform(x[:,1])
x[:,2]= label.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[1])                       

x = ohe.fit_transform(x).toarray()
x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.25,random_state =0)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units =6,activation= 'relu',input_dim=11))

classifier.add(Dense(units=6,activation = 'relu'))

classifier.add(Dense(units=1,activation='sigmoid'))


from keras.optimizers import SGD

sgd = SGD(lr=0.01)

classifier.compile(optimizer = 'adasm',loss = 'binary_crossentroopy',metrics = ['accuracy'])

classifier.fit(x-train,y_train,batch_size = 10,epochs = 10) #change the epochs and see

y_pred = classifier.predict(x_test)

import numpy as np

y_pred = np.where(y_pred>0.5,1.0)

y_pred = pd.DataFrame
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
























