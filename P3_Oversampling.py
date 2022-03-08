#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from collections import Counter


# In[2]:


df = pd.read_csv('../saved F(Drug-Disease).csv', header=None)


# In[3]:


df.iloc[:,2].value_counts()


# In[4]:


classOne = shuffle(df[df[2]==1])
classZero = shuffle(df[df[2]==0])
del df
testClassOne = classOne.iloc[:int(len(classOne)/10), :]
testClassZero = classZero.iloc[:int(len(classZero)/10), :]
testData = shuffle(pd.concat([testClassOne, testClassZero] , ignore_index=True))
del testClassZero
testData.to_csv('../Drug_disease_testData.csv', index=False)
del testData


# In[5]:


trainClassOne = classOne.iloc[int(len(classOne)/10):, :]
trainClassZero = classZero.iloc[int(len(classZero)/10):, :]
del classOne
del classZero
trainData = shuffle(pd.concat([trainClassOne, trainClassZero] , ignore_index=True))
del trainClassZero


# In[6]:


trainData.to_csv('../Drug_disease_trainData.csv', index=False)
print(trainData.shape)
del trainData


# In[7]:


# Dividing the data into X and Y
df = pd.read_csv('../Drug_disease_trainData.csv')
print(df.shape)
X_train = df.drop(['2'], axis=1)
y_train = df['2']
del df


# In[8]:


# y_train


# In[9]:


y_train.value_counts()


# In[10]:


# OverSampling
from imblearn.over_sampling import RandomOverSampler 

oversample = RandomOverSampler(sampling_strategy='minority')
X_oversample, y_oversample = oversample.fit_resample(X_train.values[:,2:], y_train)
print(X_oversample.shape)


# In[11]:


Counter(y_oversample)


# In[12]:


# X_oversample = pd.DataFrame(X_oversample)
# y_oversample = pd.DataFrame(y_oversample)


# In[12]:


df = pd.DataFrame(X_oversample)
df['y'] = y_oversample
df.shape


# In[13]:


df.head()


# In[14]:


df.to_csv('../F_Drug-Disease_oversampling.csv', index=False)
del df


# In[15]:


#smote
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

smote_sampler = SMOTE(n_jobs=-1)
# smote_pipeline = make_pipeline(RandomOverSampler(), SMOTE(n_jobs=-1))
print(smote_sampler.get_params())
X_smote , y_smote = smote_sampler.fit_resample(X_train.values[:,2:], y_train)
X_smote.shape


# In[16]:


Counter(y_smote)


# In[18]:


# X_smote = pd.DataFrame(X_smote)
# y_smote = pd.DataFrame(y_smote)


# In[18]:


dfsmote = pd.DataFrame(X_smote)
dfsmote['y'] = y_smote
dfsmote.head()


# In[19]:


print(dfsmote.shape)
del X_smote, y_smote


# In[20]:


dfsmote.to_csv('../F_Drug-Disease_SMOTE.csv', index=False)
del dfsmote


# In[21]:


# ADASYN
from imblearn.over_sampling import ADASYN
X_adasyn, y_adasyn = ADASYN().fit_resample(X_train.values[:,2:], y_train)

# X_adasyn = pd.DataFrame(X_adasyn)
# y_adasyn = pd.DataFrame(y_adasyn)
Counter(y_adasyn)


# In[22]:


X_adasyn.shape


# In[23]:


dfadasyn = pd.DataFrame(X_adasyn)
dfadasyn['y'] = y_adasyn
dfadasyn.head()


# In[24]:


print(dfadasyn.shape)
dfadasyn.to_csv('../F_Drug-Disease_Adasyn.csv', index=False)


# In[ ]:




