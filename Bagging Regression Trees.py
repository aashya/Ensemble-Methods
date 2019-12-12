#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

T=100
xaxis = np.linspace(0,2*np.pi, T)
yaxis = np.sin(xaxis)

N = 30
idx = np.random.choice(T, size = N, replace = False)

Xtrain = xaxis[idx].reshape(N,1)
Ytrain = yaxis[idx]

model = DecisionTreeRegressor()
model.fit(Xtrain, Ytrain)
prediction = model.predict(xaxis.reshape(T, 1))

print("score for one tree")
print( model.score(xaxis.reshape(T,1), yaxis))
                           
plt.plot(xaxis, prediction)
plt.plot(xaxis,yaxis)
plt.show()
                        
class BaggedTreeRegressor:
    def __init__(self, B):
        self.B = B
                           
                           
    def fit(self,X,Y):
        N = len(X)
        self.models = []
        for b in range(self.B):
            idx = np.random.choice(N, size=N, replace = True)
            Xb = X[idx]
            Yb = Y[idx]
                           
            model = DecisionTreeRegressor()
            model.fit(Xb, Yb)
            self.models.append(model)
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.B
    
    def score(self,X,Y):
        d1 = Y - self.predict(X)
        d2 = Y - Y.mean()
        return 1 - d1.dot(d1)/d2.dot(d2)

model = BaggedTreeRegressor(200)
model.fit(Xtrain, Ytrain)
print ("score for bagged trees")
print (model.score(xaxis.reshape(T,1), yaxis))
       
prediction = model.predict(xaxis.reshape(T, 1))
plt.plot(xaxis, prediction)
plt.plot(xaxis,yaxis)
plt.show()


# In[ ]:




