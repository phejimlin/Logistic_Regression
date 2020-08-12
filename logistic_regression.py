#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# # Data preparing
# 

# In[2]:


iris = load_iris(as_frame=True)
# We only take two label.
X = iris.data.loc[iris["target"] !=2].values
Y = iris.target.loc[iris["target"] !=2].values
print(X[:5])
print(Y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# # Logistic Regression with greadient descent optimizer.
# 

# Linear regression model as: h(x) = θ x
# Apply sigmoid function to output of it, make it become logistic regression. 
# 
# h(x) = sigmoid(θ x)
# 
# For loss function, we apply log_loss to calculate loss.
# The idea is to give a lot of penalties when the model predicts the wrong label.
# e.g. When actual class is 1 and the model predicts 0, we make loss very high.
# We can apply the characteristic of log in 0~1 to achieve this situation.
# 
# 
# So, there are two scenario for loss:
#     1. -log(h(x)),     if y=1
#     2. -log(1 - h(x)), if y=0
# Combine these two, loss function will become: -ylog(h(x)) - (1-y)log(1-h(x))

# In[3]:


class LogisticRegression:
    """
    Implement Logistic Regression with greadient descent optimizer.
    """
    def __init__(self, learning_rate=0.1, steps=100000, verbose=False):
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, Y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.steps):
            h = self.__sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - Y)) / Y.size
            self.theta -= self.learning_rate * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                print(f'Steps {i} loss: {self.__loss(h, Y)} \t')
    
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


# In[4]:


LR_model = LogisticRegression(verbose=True)


# In[5]:


LR_model.fit(X_train, y_train)


# In[6]:


y_pred = LR_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))


# # Logistic Regression with adagrad optimizer.
# 

# The main idea of Adagrad is to make the weights that have low gradient or infrequent updates should speed up learning rate and make them train faster. On the other hand, the weights that have higher gradient or frequent updates should slow the learning rate and make them not miss the minimum value.
# 
# In each step, the current gradient divides the sum of squares of all previous gradients of the weight and multiple learning rate. And theta(weight) will keep updating by minus equal to it.
# 

# In[7]:


class LogisticRegressionAdagrad:
    """
    Implement Logistic Regression with adagrad optimizer.
    """
    def __init__(self, learning_rate=0.1, steps=100000, verbose=False):
        self.learning_rate = learning_rate
        self.steps = steps
        self.verbose = verbose
        self.epsilon = 0.0001
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
    def fit(self, X, Y):
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        gradient_sums = np.zeros(self.theta.shape[0]) # The sum of previous gradient in each t
        for i in range(self.steps):
            h = self.__sigmoid(np.dot(X, self.theta))
            # adagrad optimizer
            gradient = np.dot(X.T, (h - Y)) / Y.size
            gradient_sums += gradient ** 2
            gradient_update = gradient / (np.sqrt(gradient_sums + self.epsilon))
            self.theta -= self.learning_rate * gradient_update
            
            if(self.verbose == True and i % 10000 == 0):
                print(f'Steps {i} loss: {self.__loss(h, Y)} \t')
    
    def predict_prob(self, X):
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


# In[8]:


LR_model_with_adagrad = LogisticRegressionAdagrad(verbose=True)


# In[9]:


LR_model_with_adagrad.fit(X_train, y_train)


# In[10]:


y_pred = LR_model_with_adagrad.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))


# # Logistic Regression by Sklearn.
# 

# In[11]:


from sklearn.linear_model import LogisticRegression 


# In[12]:


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))


# In[ ]:




