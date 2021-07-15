#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[120]:


df = pd.read_csv(r'C:\Users\mesha\Downloads\New folder (2)\data_billauth.csv',index_col=False)


# In[121]:


df #Printing the raw dataframe


# In[122]:


df.info() # Checking the data frames feature value types and the coloums in it


# In[123]:


df.isnull().sum()


# ## Data processing

# In[124]:


X = df.drop(['Class'], axis=1)
y = df['Class'] # Extracting the X(features) and the Y(Class) from the raw dataframe


# In[125]:


X # checking the X wheather all the features are extracted


# In[126]:


y # checking the y wheather all the class are extracted


# In[127]:


# Splitting into training and test data for both features and class
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33) 


# In[128]:


# Fitting the Decision Tree Classifier
from sklearn import tree
dt = tree.DecisionTreeClassifier()
model = dt.fit(X_train,y_train) 


# In[129]:


# Predicting the test data
predicted= model.predict(X_test)


# In[130]:


# Getting the accuracy score from the metrics
from sklearn.metrics import accuracy_score


# In[131]:


score = accuracy_score(y_test,predicted)


# In[132]:


print(score)


# In[133]:


# Getting the features name and the class name
names = X.columns
classes =  ['class 0','class 1']


# In[134]:


# Visualising the decission tree 
import graphviz
# DOT data
dot_data = tree.export_graphviz(model,feature_names=names,class_names=classes)
graph = graphviz.Source(dot_data,format="png") 
graph


# In[135]:


from sklearn.metrics import confusion_matrix # Visualising the confusion matrix to analyse how many False negative and False positive produced


# In[115]:


matrix = confusion_matrix(y_test,predicted)


# In[116]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test, y_test)


# In[ ]:




