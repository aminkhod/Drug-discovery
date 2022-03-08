#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[25]:


# two type of data 'C' and 'F'
# Let's start with smaller one, I mean the 'F' with dimensions of (313 Diseases, 593 Drugs).


# In[2]:


# excel_file='Rivise data/DDI_triple_2.csv'
# drug=pd.read_csv(excel_file, header=None)
# df='data/drug_name_568.txt'
# DF=pd.read_csv(df, header=None)
d='data/drug-disease-datasets/F-DrugSim.txt'
Drugsim=pd.read_csv(d, sep='\t', header=None)
print(Drugsim.shape)

Diseasesim = pd.read_csv('data/drug-disease-datasets/F-DiseaseSim.txt', sep='\t', header=None)
print(Diseasesim.shape)

# Drugs_disease_interaction = pd.read_csv('C-drug-disease interactions.txt',sep = '\t', header=None)
# # print(Drugs_disease_interaction.head())


# In[27]:


# ### Deleting missing Drugs
# druglist=[]
# mismatch=[]

# for j in range(len(D)):
#     druglist.insert(j,D.loc[j][0])
# for i in range(len(DF.loc[0])):
#     if not DF.loc[i][1] in (druglist):
#             print(i,DF.loc[i][1])
#             mismatch.insert(i,DF.loc[i][1])
#             DF=DF.drop([i])
#             drug=drug.drop([i],axis=0)
#             drug=drug.drop([i],axis=1)


# In[7]:


# Drugs_disease_interaction.iloc[1:,1:].to_csv('saved drug.csv',index=False)
drug_disease_int = pd.read_csv('data/drug-disease-datasets/F-drug-disease interactions.txt', sep='\t', header=None)
drug_disease_int.iloc[1:,0].to_csv("saved Diseasename.csv", index=False)
Diseasename = pd.read_csv("saved Diseasename.csv", header=0)

drug_disease_int.iloc[0,1:].to_csv("saved Drugname.csv", index=False)
Drugname = pd.read_csv("saved Drugname.csv", header=0)

print(drug_disease_int.head(), drug_disease_int.shape)

Drugsim.insert(0, "-1", Drugname.values)
Diseasesim.insert(0, "-1", Diseasename.values)


print(Drugsim.shape)
print(Diseasesim.shape)
# print(drug.head())


# In[8]:


Diseasename


# In[9]:


Drugname


# In[10]:


finalmatrix=[]
for i in range(len(drug_disease_int.iloc[:,0])-1): # putting disease names
    for j in range(len(drug_disease_int.iloc[0,:])-1): # putting drug names
        result = [drug_disease_int.iloc[0,j+1], drug_disease_int.iloc[i+1,0], drug_disease_int.loc[i+1][j+1]]
#         print(result)
        finalmatrix.append(result)


# In[11]:


len(finalmatrix)


# In[12]:


del drug_disease_int


# In[13]:


pd.DataFrame(finalmatrix).to_csv("saved F_pairs.csv",index=False)
del finalmatrix


# In[14]:


F = pd.read_csv("saved F_pairs.csv")
F


# In[15]:


# r, c = F.shape
# print(r,c)
# print(F.iloc[0,:])


# In[16]:


# j = 0
# for i in range(0, r, len(DF.iloc[:,0])):
#     try:
#         F = F.drop([i + j])
# #         print(i)
#     except:
#         1
# #         print(str(i)+ " can't be droped")
#     j += 1
# F.shape


# In[17]:


r, c = F.shape
print(r,c)
# list(range(0, r, len(DF.iloc[:,0])))
F.iloc[0,:]


# In[18]:


print(Drugsim.shape, Diseasesim.shape)


# In[19]:


Drugsim


# In[20]:


313 * 593


# In[21]:


# F =F.iloc[:,:3]
# F


# In[22]:


# print(len(Drugsim.iloc[0,:]),len(Diseasesim.iloc[0,:]))


# In[23]:


for k in range(len(Drugsim.iloc[0,:])-1):
    F[k + 3] = F['0'].map(Drugsim.set_index('-1')[k])
    k += 1
print(F.shape)
# F


# In[24]:


F


# In[25]:


for k in range(len(Diseasesim.iloc[0,:])-1):
    F[k - 1 + len(Drugsim.iloc[0,:]) + 3] = F['1'].map(Diseasesim.set_index('-1')[k])
    k += 1


# In[26]:


print(F.shape)
F


# In[27]:


del Diseasename
del Drugsim
del Drugname
del Diseasesim


# In[28]:


F.shape


# In[29]:


F.to_csv("../saved F(Drug-Disease).csv", index=False, header=None)


# In[30]:


del F


# In[31]:


# F.head()


# In[32]:


# df = pd.read_csv("../saved F(Drug-Disease).csv", header=None)


# In[33]:


# df


# In[34]:


# df[2]


# In[36]:


# # OverSampling
# from imblearn.over_sampling import RandomOverSampler


# # Dividing the data into X and Y
# X=df.drop(2,axis=1)
# y=df[2]

# oversample = RandomOverSampler(sampling_strategy='minority')
# X_oversample, y_oversample = oversample.fit_resample(X, y)


# In[38]:


# y_oversample.value_counts()


# In[41]:


# df=pd.concat([X_oversample, y_oversample ], axis=1)
# df.shape


# In[42]:


# df.to_csv('../F-Drug-DiseaseRandomOverSampler.csv', index=False)
# del df


# In[46]:





# In[47]:


# #smote
# from imblearn.over_sampling import SMOTE
# smote_sampler = SMOTE(random_state=42)
# X_smote , y_smote = smote_sampler.fit_resample(X.iloc[:,2:],y)
# X_smote.shape


# In[48]:


# y_smote.value_counts()


# In[49]:


# X_smote= pd.DataFrame(X_smote)
# y_smote= pd.DataFrame(y_smote)
# y_smote.value_counts()


# In[ ]:


# dfsmote=pd.concat([X_smote, y_smote], axis=1)
# dfsmote.to_csv('../BalanceData_SMOTE.csv', index=False)
# del dfsmote


# In[ ]:


# # ADASYN
# from imblearn.over_sampling import ADASYN
# X_adasyn, y_adasyn = ADASYN().fit_resample(X, y)
# del X
# X_adasyn= pd.DataFrame(X_adasyn)
# y_adasyn= pd.DataFrame(y_adasyn)
# y_adasyn.value_counts()


# In[ ]:


# X_adasyn.shapesds


# In[ ]:


# dfadasyn=pd.concat([X_adasyn, y_adasyn], axis=1)


# In[ ]:


# dfadasyn.to_csv('../BalanceData_Adasyn.csv', index=False)


# In[ ]:




