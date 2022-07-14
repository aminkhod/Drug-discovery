#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
import pandas as pd


# In[25]:


# two type of data 'C' and 'F'
# Let's start with smaller one, I mean the 'F' with dimensions of (313 Diseases, 593 Drugs).


# In[2]:


# excel_file='Rivise data/DDI_triple_2.csv'
# drug=pd.read_csv(excel_file, header=None)
# df='data/drug_name_568.txt'
# DF=pd.read_csv(df, header=None)
# d='data/drug-disease-datasets/F-DrugSim.txt'
# Drugsim=pd.read_csv(d, sep='\t', header=None)


Diseasesim = pd.read_csv('data/drug-disease-datasets/F-DiseaseSim.txt', sep='\t', header=None)


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

# drug_disease_int.iloc[0,1:].to_csv("saved Drugname.csv", index=False)
# Drugname = pd.read_csv("saved Drugname.csv", header=0)



# Drugsim.insert(0, "-1", Drugname.values)
Diseasesim.insert(0, "-1", Diseasename.values)


F = pd.read_csv('../F_temp.csv')

F = F.merge(Diseasesim, left_on='1', right_on=-1, how='left' )
F.drop([-1], axis=1, inplace=True)
del Diseasename
# del Drugsim
# del Drugname
del Diseasesim
F.to_csv("../saved F(Drug-Disease).csv", index=False, header=None)
