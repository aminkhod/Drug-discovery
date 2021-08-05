#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


df = pd.read_csv('../saved F(Drug-Disease).csv')


# In[ ]:


# OverSampling
#from imblearn.over_sampling import RandomOverSampler


# Dividing the data into X and Y
X=df.drop('2',axis=1)
y=df['2']

#oversample = RandomOverSampler(sampling_strategy='minority')
#X_oversample, y_oversample = oversample.fit_resample(X, y)


# In[ ]:


#y_over.value_counts()


# In[ ]:


#df=pd.concat([X_over, y_over], axis=1)
#df.shape


# In[ ]:


df.to_csv('../Actual Data_BalanceData_oversampling.csv', index=False)
del df


# In[ ]:


#smote
from imblearn.over_sampling import SMOTE
smote_sampler = SMOTE(random_state=42)
X_smote , y_smote = smote_sampler.fit_resample(X,y)
X_smote.shape


# In[ ]:


y.value_counts()


# In[ ]:


X_smote= pd.DataFrame(X_smote)
y_smote= pd.DataFrame(y_smote)
y_smote.value_counts()


# In[ ]:


dfsmote=pd.concat([X_smote, y_smote], axis=1)
dfsmote.to_excel('../Actual Data_BalanceData_SMOTE.xlsx', index=False)
del dfsmote


# In[ ]:


# ADASYN
from imblearn.over_sampling import ADASYN
X_adasyn, y_adasyn = ADASYN().fit_resample(X, y)
del X
X_adasyn= pd.DataFrame(X_adasyn)
y_adasyn= pd.DataFrame(y_adasyn)
y_adasyn.value_counts()


# In[ ]:


X_adasyn.shape


# In[ ]:


dfadasyn=pd.concat([X_adasyn, y_adasyn], axis=1)
dfadasyn.to_excel('../Actual Data_BalanceData_Adasyn.xlsx', index=False)


# In[ ]:




