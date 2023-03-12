#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


# In[2]:


train = pd.read_csv("train_titanic.csv")
test = pd.read_csv("test_titanic.csv")


# In[3]:


train.columns.difference(test.columns)


# In[4]:


train.info()


# In[5]:


df = pd.concat([train,test], ignore_index= True)


# In[6]:


df.head(1)


# In[7]:


df.tail(1)


# In[8]:


df.isnull().sum()


# In[9]:


df["Age"].fillna(df["Age"].mode()[0], inplace = True)


# In[10]:


df["Sex"].value_counts(dropna =False)


# In[11]:


df["Embarked"].value_counts(dropna =False)


# In[12]:


df["Embarked"].fillna(df["Embarked"].mode()[0], inplace = True)


# In[13]:


df.replace({"Sex":{"male":0,"female":1},"Embarked":{"S":1,"C":2,"Q":3}}, inplace = True)


# In[14]:


df.isnull().sum()


# In[15]:


pre_train = df[df["Survived"].notna()]
pre_test = df[df["Survived"].isna()]


# In[16]:


pre_train.head(5)


# In[17]:


pre_train.tail(5)


# In[18]:


x = pre_train.drop(["Survived","PassengerId","Name","Ticket","Cabin"], axis = 1)
y = pre_train["Survived"]


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[20]:


model = LogisticRegression()


# In[21]:


model.fit(x_train,y_train)


# In[22]:


y_pred = model.predict(x_train)
print('accuracy score',accuracy_score(y_train,y_pred))


# In[24]:


y_test_pred = model.predict(x_test)
print('confusion matrix\n',confusion_matrix(y_test,y_test_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




