#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
import pandas as pd


# In[2]:


# two type of data 'C' and 'F'
# Let's start with smaller one, I mean the 'F' with dimensions of (313 Diseases, 593 Drugs).


# In[3]:


# excel_file='Rivise data/DDI_triple_2.csv'
# drug=pd.read_csv(excel_file, header=None)
# df='data/drug_name_568.txt'
# DF=pd.read_csv(df, header=None)
d='data/Drug disease/Drug_CosineSNF(chemStracture_DrugProtein_sideEffect).csv'
Drugsim=pd.read_csv(d, header=None)
print(Drugsim.shape)

Diseasesim = pd.read_csv('data/Drug disease/Disease_CosineSNF(DisPhenotype_DisProteinm).csv', header=None)
print(Diseasesim.shape)

# Drugs_disease_interaction = pd.read_csv('C-drug-disease interactions.txt',sep = '\t', header=None)
# # print(Drugs_disease_interaction.head())


# In[4]:


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


# In[5]:


# Drugs_disease_interaction.iloc[1:,1:].to_csv('saved drug.csv',index=False)
drug_disease_int = pd.read_csv('data/Drug disease/disease_drug_interaction.csv', header=None)

# drug_disease_int.iloc[1:,0].to_csv("saved Diseasename.csv", index=False)
# Diseasename = pd.read_csv("saved Diseasename.csv", header=0)

# drug_disease_int.iloc[0,1:].to_csv("saved Drugname.csv", index=False)
# Drugname = pd.read_csv("saved Drugname.csv", header=0)

Drugname = list(range(len(Drugsim)))
Diseasename = list(range(len(Diseasesim)))
print(drug_disease_int.head(), drug_disease_int.shape)

Drugsim.insert(0, -1, Drugname)
Diseasesim.insert(0, -1, Diseasename)


print(Drugsim.shape)
print(Diseasesim.shape)
# print(drug.head())
drug_disease_int.iloc[0:,0:].astype(int).sum()


# In[6]:


Drugsim[-1]


# In[7]:


Diseasename[:10]


# In[8]:


Drugname[:10]


# In[9]:


finalmatrix=[]
for i in range(len(Drugname)): # putting disease names
    for j in range(len(Diseasename)): # putting drug names
        result = [Drugname[i], Diseasename[j], drug_disease_int.loc[j][i]]
#         print(result)
        finalmatrix.append(result)
#     print(finalmatrix)
#     if i == 1:
#         print(finalmatrix)
        
#         break


# In[10]:


708 * 5603


# In[11]:


# finalmatrix


# In[12]:


len(finalmatrix)


# In[13]:


del drug_disease_int


# In[14]:


# pd.DataFrame(finalmatrix).to_csv("saved F_pairs.csv",index=False)
F = finalmatrix.copy()
del finalmatrix


# In[ ]:


# F = pd.read_csv("saved F_pairs.csv")


# In[16]:


# r, c = F.shape
# print(r,c)
# print(F.iloc[0,:])


# In[17]:


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


# In[18]:


r, c = F.shape
print(r,c)
# list(range(0, r, len(DF.iloc[:,0])))
F.iloc[0,:]


# In[19]:


print(Drugsim.shape, Diseasesim.shape)


# In[20]:


Drugsim.head()


# In[21]:


708 * 5603


# In[22]:


# F =F.iloc[:,:3]
# F


# In[23]:


# print(len(Drugsim.iloc[0,:]),len(Diseasesim.iloc[0,:]))


# In[24]:


# for k in range(len(Drugname)):
#     F[k + 3] = F['0'].map(Drugsim.set_index(-1)[k])
#     k += 1

F = F.merge(Drugsim, left_on='0', right_on=-1, how='left' )
F.drop([-1], axis=1, inplace=True)

print(F.shape)


# In[25]:


del Drugsim
del Drugname


# In[26]:


print(F.shape)
F.head()


# In[ ]:


# F.to_csv('../F_temp.csv', index=False)
# del F


# In[ ]:


# F = pd.read_csv('../F_temp.csv')
# F.shape


# In[ ]:


# for k in range(len(Diseasesim.iloc[0,:])-1):
#     F[k - 1 + len(Drugsim.iloc[0,:]) + 3] = F['1'].map(Diseasesim.set_index(-1)[k])
#     k += 1

F = F.merge(Diseasesim, left_on='1', right_on=-1, how='left' )
F.drop([-1], axis=1, inplace=True)

print(F.shape)
# F


# In[ ]:


# print(F.shape)
# F


# In[ ]:


del Diseasename
# del Drugsim
# del Drugname
del Diseasesim


# In[ ]:


F.shape


# In[ ]:


F.to_csv("../saved F(Drug-Disease).csv", index=False, header=None)


# In[ ]:


# del F


# In[ ]:


# F.head()


# In[ ]:


# df = pd.read_csv("../saved F(Drug-Disease).csv", header=None)


# In[ ]:


# df


# In[ ]:


# df[2]


# In[ ]:


# # OverSampling
# from imblearn.over_sampling import RandomOverSampler


# # Dividing the data into X and Y
# X=df.drop(2,axis=1)
# y=df[2]

# oversample = RandomOverSampler(sampling_strategy='minority')
# X_oversample, y_oversample = oversample.fit_resample(X, y)


# In[ ]:


# y_oversample.value_counts()


# In[ ]:


# df=pd.concat([X_oversample, y_oversample ], axis=1)
# df.shape


# In[ ]:


# df.to_csv('../F-Drug-DiseaseRandomOverSampler.csv', index=False)
# del df


# In[ ]:





# In[ ]:


# #smote
# from imblearn.over_sampling import SMOTE
# smote_sampler = SMOTE(random_state=42)
# X_smote , y_smote = smote_sampler.fit_resample(X.iloc[:,2:],y)
# X_smote.shape


# In[ ]:


# y_smote.value_counts()


# In[ ]:


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




