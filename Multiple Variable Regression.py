#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Loading dataset
df=pd.read_csv(r'C:\Users\medik\Downloads\LR-1.csv')
len(df)


# In[3]:


df


# In[4]:


#importing ML libraries
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# In[5]:


#Separate input parameters and output parameters
#The dataset contains 5 columns (X1,X2,X3,X4,X5), and we are selecting only X2,X3,X4,X5 as input parameters and X1 as output parameters
X = df.iloc[:,df.columns != 'x1'] #Selecting all columns except X1 as input params 
Y = df.iloc[:, 0] #Selecting X1 as output param
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0) 


# In[6]:


#Training the model
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
print(len(X_train))


# In[7]:


y_pred = model.predict(X_test)
y_pred


# In[11]:


a=df.iloc[:,2]
b=df.iloc[:,0]

plt.scatter(a,b,label='line1')
#plt.show()
c=df.iloc[:,3]
plt.scatter(c,b,label='line2')
d=df.iloc[:,4]
plt.scatter(d,b,label='line3')
plt.xlim(0,10)
plt.ylim(3,8)

plt.show()


# In[9]:


l=[True,50,60]
l,sum(l)

