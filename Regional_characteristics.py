#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[28]:


# Read data
gdp = pd.read_csv('/Users/maijahavusela/Desktop/gradu/data/regional char/gdp.csv',sep=',')
pop = pd.read_csv('/Users/maijahavusela/Desktop/gradu/data/regional char/population.csv',sep=',')
med_age = pd.read_csv('/Users/maijahavusela/Desktop/gradu/data/regional char/median age.csv',sep=',')
emp = pd.read_csv('/Users/maijahavusela/Desktop/gradu/data/regional char/employed.csv',sep=',')

# Choose wanted columns
gdp = gdp[['geo',
           'TIME_PERIOD',
           'OBS_VALUE']]

pop = pop[['geo: Geopolitical entity (reporting)',
           'TIME_PERIOD: Time',
           'OBS_VALUE: Observation value']]

med_age = med_age[['geo: Geopolitical entity (reporting)',
           'TIME_PERIOD: Time',
           'OBS_VALUE: Observation value']]

emp = emp[['geo',
           'TIME_PERIOD',
           'OBS_VALUE']]

# Rename columns
gdp = gdp.rename(
    columns={
        'geo':'NUTS3',
        'TIME_PERIOD':'YEAR',
        'OBS_VALUE':'GDP' # Million euro
    }
)
pop = pop.rename(
    columns={
        'geo: Geopolitical entity (reporting)':'NUTS3',
        'TIME_PERIOD: Time':'YEAR',
        'OBS_VALUE: Observation value':'population'
    }
)
med_age = med_age.rename(
    columns={
        'geo: Geopolitical entity (reporting)':'NUTS3',
        'TIME_PERIOD: Time':'YEAR',
        'OBS_VALUE: Observation value':'median age'
    }
)
emp = emp.rename(
    columns={
        'geo':'NUTS3',
        'TIME_PERIOD':'YEAR',
        'OBS_VALUE':'employed' # Unit of measure: thousand people
    }
)

# Cleaning the NUTS ID column
pop['NUTS3'] = pop['NUTS3'].str.split(':').str[0]
med_age['NUTS3'] = med_age['NUTS3'].str.split(':').str[0]
emp['NUTS3'] = emp['NUTS3'].str.split(':').str[0]

# Merging dataframes
merged_df = pd.merge(gdp, pop, on=['NUTS3', 'YEAR'], how='outer')
merged_df = pd.merge(merged_df, med_age, on=['NUTS3', 'YEAR'], how='outer')
merged_df = pd.merge(merged_df, emp, on=['NUTS3', 'YEAR'], how='outer')

# Grouping byt NUTS ID and calculating average of all years
grouped_df = merged_df.groupby('NUTS3')[['GDP', 'population', 'median age', 'employed']].mean().reset_index()
grouped_df


# In[29]:


# Save data
grouped_df.to_csv('/Users/maijahavusela/Desktop/gradu/data/pythonista/regional char/regional_char.csv')


# In[ ]:




