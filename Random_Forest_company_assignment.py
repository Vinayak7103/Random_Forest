#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[3]:


company= pd.read_csv('C:/Users/vinay/Downloads/Company_Data.csv')


# In[4]:


company.head()


# In[5]:


company["Sales"].min()


# In[6]:


company["Sales"].max()


# In[7]:


company["Sales"].value_counts()


# ## Checking for maximum and minimum values to decide what will be the cut off point
# 

# In[8]:


np.median(company["Sales"])


# In[9]:


##Knowing the middle value by looking into median so that i find the middle value to check to divide data into two levels.
company["sales"]= "<=7.49"
company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"


# In[10]:


company["sales"].unique()
company["sales"].value_counts()


# In[11]:


##Dropping Sales column from the data 
company.drop(["Sales"],axis=1,inplace = True)
company.isnull().sum() # no null value


# ## As, the fit does not consider the String data, we need to encode the data.
# 

# In[12]:


from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
for column_name in company.columns:
    if company[column_name].dtype == object:
        company[column_name] = le.fit_transform(company[column_name])
    else:
        pass


# In[13]:


features = company.iloc[:,0:10] 
labels = company.iloc[:,10]


# In[14]:


##Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=0.3,stratify = labels)


# In[15]:


##Looking into the class variable split
y_train.value_counts()
y_test.value_counts()


# In[16]:


##Building the model
from sklearn.ensemble import RandomForestClassifier as RF


# In[17]:


model =RF(n_jobs=4,n_estimators = 150, oob_score =True,criterion ='entropy') 
model.fit(x_train,y_train)
model.oob_score_


# In[20]:


##Predicting on training data
pred_train = model.predict(x_train)
pred_train


# In[22]:


##Accuracy on training data
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train,pred_train)
accuracy_train #100%


# In[24]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_train = confusion_matrix(y_train,pred_train)


# In[26]:


##Prediction on test data
pred_test = model.predict(x_test)
pred_test


# In[28]:


##Accuracy on test data
accuracy_test = accuracy_score(y_test,pred_test)
accuracy_test


# In[29]:


np.mean(y_test==pred_test)


# In[31]:


##Confusion matrix
confusion_test = confusion_matrix(y_test,pred_test)
confusion_test


# In[ ]:




