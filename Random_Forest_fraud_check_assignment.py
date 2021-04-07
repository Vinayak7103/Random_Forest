#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# In[2]:


fraud= pd.read_csv('C:/Users/vinay/Downloads/Fraud_check.csv')


# In[3]:


fraud


# In[4]:


fraud.rename(columns = {'Taxable.Income':'Taxable_Income','City.Population':'City_Population','Work.Experience':'Work_Experience','Marital.Status':'Marital_Status'},inplace = True) ;fraud


# In[5]:


fraud1 = pd.get_dummies(fraud,columns=['Undergrad','Marital_Status','Urban']);fraud1


# In[6]:


fraud1["income"]="<=30000"
fraud1.loc[fraud1["Taxable_Income"]>=30000,"income"]="Good"
fraud1.loc[fraud1["Taxable_Income"]<=30000,"income"]="Risky"


# In[7]:


fraud1


# In[8]:


fraud1.drop(["Taxable_Income"],axis=1,inplace=True)


# ## Model.fit doesnt not consider String. So, we should encode

# In[9]:


label_encoder = preprocessing.LabelEncoder()
fraud1['income']= label_encoder.fit_transform(fraud1['income']) 


# In[10]:


fraud1['income'].unique()


# In[11]:


fraud1.income.value_counts()


# In[12]:


colnames = list(fraud1.columns)
colnames


# In[13]:


x=fraud1.iloc[:,0:9]
y=fraud1['income']


# In[14]:


y


# In[15]:


# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# In[16]:


x_train


# ## Building model Random forest Classifier using Entropy Criteria

# In[17]:


from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[18]:


num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RF(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())


# In[19]:


model.fit(x_train,y_train)


# In[20]:


##Predictions on train data
prediction = model.predict(x_train)


# In[21]:


##Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
accuracy


# In[22]:


np.mean(prediction == y_train)


# In[23]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[24]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[25]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)
acc_test # 70%


# ## In random forest we can plot a Decision tree present in Random forest
