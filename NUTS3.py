#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Read data
data = pd.read_csv('/filepath',sep=',')

# Choosing only the NUTS3
data = data[data['NUTS_ID'].astype(str).str.len() == 5]

# Sorting
data = data.sort_values(by='NUTS_ID', key=lambda col: col.astype(str))


# In[ ]:




