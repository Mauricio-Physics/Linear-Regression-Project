#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sbn
import math 
import numpy as np


# In[2]:


Customers_Data = pd.read_csv('C:/Users/MAURICIO/Desktop/Customers_Data.csv',encoding='utf-8')


# In[3]:


Customers_Data.head()


# In[4]:


Customers_Data.columns


# In[5]:


Customers_Data.info()


# In[6]:


sbn.jointplot(x='Time on Website',y='Yearly Amount Spent',data=Customers_Data)


# In[7]:


sbn.jointplot(x='Time on Website',y='Yearly Amount Spent',kind="hex",data=Customers_Data)


# In[8]:


sbn.set_palette("GnBu_d")
sbn.set_style('whitegrid')
sbn.pairplot(Customers_Data)


# In[9]:


## The target will be 'Yearly Amount Spent'


# In[10]:


sbn.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=Customers_Data)              ## fit regression


# In[11]:


linear_model = linear_model.LinearRegression()


# In[12]:


x = Customers_Data['Length of Membership'] 
y = Customers_Data['Yearly Amount Spent']     
print(len(x),len(y))


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=1)  


# In[54]:


print(len(x_train))
print(len(x_train)/len(x))


# In[55]:


print(x_train.shape)
print(y_train.shape)

print(type(x_train))
print(type(y_train))


# In[56]:


x_train = x_train.to_frame()
y_train = y_train.to_frame()

x_test = x_test.to_frame()
y_test = y_test.to_frame()


# In[57]:


print(x_train.shape)
print(y_train.shape)

print(type(x_train))
print(type(y_train))


# In[58]:


linear_model.fit(x_train,y_train)


# In[59]:


# Have a look at R sq to give an idea of the fit 
print('R^2 : ',linear_model.score(x_train,y_train))   ## Return the coefficient of determination R^2 of the prediction.

# and so the correlation is..
print('Correlation: ', math.sqrt(linear_model.score(x_train,y_train)))

# Equation coefficient and Intercept
print("Slope: ", linear_model.coef_)

print("Intercept: ", linear_model.intercept_)


# In[60]:


y_predicted = linear_model.predict(x_test)


# In[61]:


print('MAE :'," ", metrics.mean_absolute_error(y_test,y_predicted))
print('MSE :'," ", metrics.mean_squared_error(y_test,y_predicted))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_predicted)))


# In[66]:


plt.figure(figsize=(10,10))
plt.title('Comparison of Y values in test and the Predicted values')
plt.xlabel('Predicted values')
plt.xlim(275,800)
plt.ylabel('Test Set')
plt.ylim(275,800)
plt.scatter(y_predicted,y_test)
plt.xlabel('Y Predicted')
plt.ylabel('Y Test')


# In[63]:


plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_predicted, color='red', linewidth=2)
plt.show()


# In[64]:


sbn.distplot(y_test - y_predicted,bins=100)


# In[ ]:




