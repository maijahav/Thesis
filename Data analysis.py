#!/usr/bin/env python
# coding: utf-8

# In[1]:


import skbio
import sklearn
import tslearn
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_samples
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import cdist_dtw
import numpy as np


# In[2]:


# Read data
data = pd.read_csv('/Users/maijahavusela/Desktop/gradu/data/10.4. saadut/Erasmus_2014-2022_aggregated_NUTS3_v2021.csv',sep=',')

# Splitting the origin and destination NUTS 3 codes to separate columns
data['orig'] = data['OD_ID'].apply(lambda x: x.split('_')[0])
data['dest'] = data['OD_ID'].apply(lambda x: x.split('_')[1])

# Getting a list of all origins per destination and renaming the column
opd = data.groupby(['dest', 'year']).agg({'orig':lambda x: list(x)}).reset_index().rename(columns={'dest':'NUTS3'})

# Vice versa
dpo = data.groupby(['orig', 'year']).agg({'dest':lambda x: list(x)}).reset_index().rename(columns={'orig':'NUTS3'})


# ### Calculating diversity indices

# In[3]:


# Creating a column for the number of occurences
opd['input'] = opd['orig'].apply(lambda x: list(Counter(x).values()))
dpo['input'] = dpo['dest'].apply(lambda x: list(Counter(x).values()))


# In[4]:


# Defining a funtion to calculate the Shannon entropy
def shannon_entropy(data):
    """Calculates the Shannon entropy of a sequence."""
    if not data:
        return 0
    counts = Counter(data)
    probabilities = [count / len(data) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def normalized_shannon_entropy(data):
    """Calculates the normalized Shannon entropy of a sequence."""
    entropy = shannon_entropy(data)
    if not data:
        return 0

    max_entropy = math.log2(1514) # Normalizing with the number of NUTS 3 regions
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

# Origins per destination
for i, row in opd.iterrows():
    opd.at[i, 'simpson'] = skbio.diversity.alpha.simpson(row['input']) # Simpson's diversity index
    opd.at[i, 'norm_shannon'] = normalized_shannon_entropy(row['orig']) # Normalized Shannon entropy
    opd.at[i, 'shannon'] = skbio.diversity.alpha.shannon(row['input']) # Shannon entropy/diversity index
    opd.at[i, 'uniques'] = skbio.diversity.alpha.observed_features(row['input']) # Number of unique regions
    
# Destinations per origin
for i, row in dpo.iterrows():
    dpo.at[i, 'simpson'] = skbio.diversity.alpha.simpson(row['input']) # Simpson's diversity index
    dpo.at[i, 'norm_shannon'] = normalized_shannon_entropy(row['dest']) # Normalized Shannon entropy
    dpo.at[i, 'shannon'] = skbio.diversity.alpha.shannon(row['input']) # Shannon entropy/diversity index
    dpo.at[i, 'uniques'] = skbio.diversity.alpha.observed_features(row['input']) # Number of unique regions
    
# Save data
#opd.to_csv('/Users/maijahavusela/Desktop/gradu/data/pythonista/origins_per_destination_nuts3.csv')


# #### Plotting origins per destination regions
# Change variables where needed.

# In[5]:


# All nuts 3, origins per destination
# Set Seaborn style
#sns.set_style('whitegrid')

# Customize grid appearance
#plt.rcParams['grid.color'] = '#eeeeee'     # Very light grey
#plt.rcParams['grid.linewidth'] = 0.5
#plt.rcParams['axes.grid'] = True
#plt.rcParams['grid.alpha'] = 0.8           # Dimmer with transparency

#fig, ax = plt.subplots(figsize=(8, 6))

# Plot line in orange
#sns.lineplot(data=opd, x='year', y='uniques', ax=ax, color='pink')

# Labels and title
#ax.set_title("Unique Origins Per NUTS 3 Destination Regions", fontsize=14, fontname='Arial')
#ax.set_ylabel("Number of Unique Origin Regions", fontname='Arial')
#ax.set_xlabel('Year', fontname='Arial')

# Save to file
#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/UniqueOrigins_nuts3_per_dest.png',
            #dpi=500, bbox_inches='tight')


# #### Plotting destinations per origin regions
# Change variables where needed.

# In[6]:


# All nuts 3, destinations per origin
# Set Seaborn style
#sns.set_style('whitegrid')

# Customize grid appearance
#plt.rcParams['grid.color'] = '#eeeeee'     # Very light grey
#plt.rcParams['grid.linewidth'] = 0.5
#plt.rcParams['axes.grid'] = True
#plt.rcParams['grid.alpha'] = 0.8           # Dimmer with transparency

#fig, ax = plt.subplots(figsize=(8, 6))

# Plot line
#sns.lineplot(data=dpo, x='year', y='uniques', ax=ax, color='pink')

# Labels and title
#ax.set_title("Unique Destinations Per NUTS 3 Origin Regions", fontsize=14, fontname='Arial')
#ax.set_ylabel("Number of Unique Destination Regions", fontname='Arial')
#ax.set_xlabel('Year', fontname='Arial')

# Save to file
#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/UniqueDestinations_nuts3_per_orig.png',
            #dpi=500, bbox_inches='tight')


# #### Plotting both in the same graph
# Change variables where needed.

# In[7]:


# Set Seaborn style and grid appearance
#sns.set_style('whitegrid')
#plt.rcParams['grid.color'] = '#eeeeee'
#plt.rcParams['grid.linewidth'] = 0.5
#plt.rcParams['axes.grid'] = True
#plt.rcParams['grid.alpha'] = 0.8

# Create one figure and axis
#fig, ax = plt.subplots(figsize=(8, 6))

# Define light purple
#light_purple = "#d4b3ff"

# Plot both lines on the same axes
#sns.lineplot(data=opd, x='year', y='uniques', ax=ax, color='pink', label='Origins per Destination')
#sns.lineplot(data=dpo, x='year', y='uniques', ax=ax, color=light_purple, label='Destinations per Origin')

# Add title and labels
#ax.set_title("Unique regions per NUTS 3 (Origins and Destinations)", fontsize=14, fontname='Arial')
#ax.set_ylabel("Number of Unique Regions", fontname='Arial')
#ax.set_xlabel("Year", fontname='Arial')

# Show legend
#ax.legend()

# Save to file
#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Uniques_nuts3_both.png',
            #dpi=500, bbox_inches='tight')

# Show the plot
#plt.show()


# ## Clustering

# ### Years 2014-2022

# In[8]:


# Pivoting the data
#pivot_df = opd.pivot(index='NUTS3', columns='year', values='norm_shannon') # Origins per destinations
    
# Handling missing values (filling with 0 for simplicity)
#pivot_df = pivot_df.fillna(0.0)

# Creating lists for silhouettes scores and inertias
#wcss = []
#silhouette_scores = []

#pivot_df


# In[9]:


# Scaling the data if "pivot_df" doesnt work
#scaler = StandardScaler()
#scaled_data = scaler.fit_transform(pivot_df)

# Initiating the model
#print("Initializing model")
#model = TimeSeriesKMeans(n_clusters=6, metric="dtw", n_init=15, max_iter=42, random_state=42)

# Fitting the model
#print("Fitting model")
#clusters = model.fit_predict(pivot_df)
    
# Getting silhouette scores
#silhs = silhouette_samples(cdist_dtw(pivot_df), clusters, metric='precomputed')

# Creating an empty dataframe for silhouette scores
#sco_df = pd.DataFrame()

# Looping over cluster count
#print("Calculating silhouette scores..")
#for label in range(6):
        
        # Getting silhouette score for current cluser
        #score = silhs[clusters == label].mean()
        
        # Putting the score into dataframe
        #sco_df.at[label, 'cluster'] = label
        #sco_df.at[label, 'score'] = score
    
# Adding cluster labels to the DataFrame
#pivot_df['cluster'] = clusters
    
# Creating a copy of dataframe
#nudf = pivot_df.reset_index()

# Creating a dictionary of NUTS 3 codes and cluster labels
#clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# Assigning cluster labels to NUTS 3 region polygons
#print("Assigning clusters to original data")
#opd['cluster'] = opd['NUTS3'].apply(lambda x: clusterd[x])

# Plotting
#print("Plotting")
#g = sns.lmplot(data=opd, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster", y=1.02)

# Saving the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends_k6.png',
          #dpi=500, bbox_inches='tight')


# In[10]:


#orange_shades = ['#ffe5b4',  # light peach
                 #'#ffcc80',  # light orange
                 #'#ff9900',  # standard orange
                 #'#cc6600']  # darker burnt orange

#plt.figure(figsize=(10, 6))
#print("Plotting with orange shades")

#for i, cluster in enumerate(sorted(opd['cluster'].unique())):
    #subset = opd[opd['cluster'] == cluster]
    #sns.regplot(data=subset, x='year', y='norm_shannon', 
                #scatter=False, order=1, label=f'Cluster {cluster}', color=orange_shades[i])

#plt.title("Normalized Shannon entropy per cluster, origin regions per destination")
#plt.xlabel("Year")
#plt.ylabel("Normalized Shannon Entropy")
#plt.legend(title='Cluster')

#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends_orange_k6.png',
            #dpi=500, bbox_inches='tight')
#plt.show()


# In[11]:


#k_range = range(2, 8) # Test k from 2 to 8
#print('Starting clustering...')
#for k in k_range:
        #kmeans = TimeSeriesKMeans(n_clusters=k, metric='dtw', random_state=42,
                                  #n_init=5, max_iter=42) # n_init suppresses warning
        #kmeans.fit(pivot_df)
        #wcss.append(kmeans.inertia_) # Inertia is the WCSS
        #score = silhouette_score(pivot_df, kmeans.labels_, metric='dtw')
        #print('Adding silhouette scores...')
        #silhouette_scores.append(score)
# plot elbow method inertias
#print('Plotting elbow methods...')
#fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
#axes[0].plot(k_range, wcss, marker='o', linestyle='--')
#axes[0].set_title('Elbow Method for Optimal k ({})'.format(col))
#axes[0].set_xlabel('Number of Clusters (k)')
#axes[0].set_ylabel('WCSS (Inertia)')
#axes[0].grid(True)
    
# plot silhouette scores
#print('Plotting silhouette scores...')
#axes[1].plot(k_range, silhouette_scores, marker='o', linestyle='--')
#axes[1].set_title('Silhouette Score for Optimal k ({})'.format(col))
#axes[1].set_xlabel('Number of Clusters (k)')
#axes[1].set_ylabel('Average Silhouette Score')
#axes[1].grid(True)


# # Pre- and post-Covid-19
# Dividing the dataframe, resulting in one df with the years 2014-2019 (pre-Covid) and one with the years 2020-2022 (post-Covid). Finding out the optimal K for both, using the Normalized Shannon entropy values. After K values, clustering.
# ### Origins per destination

# In[12]:


# Origins per destinations with years 2014–2019
opd_2014_2019 = opd[opd['year'].between(2014, 2019)]

# Origins per destinations with years 2020–2022
opd_2020_2022 = opd[opd['year'].between(2020, 2022)]


# In[13]:


# Pivot the data
pivot_df_pre = opd_2014_2019.pivot(index='NUTS3', columns='year', values='norm_shannon')
pivot_df_post = opd_2020_2022.pivot(index='NUTS3', columns='year', values='norm_shannon')
    
# Handle missing values (fill with 0 for simplicity)
pivot_df_pre = pivot_df_pre.fillna(0.0)
pivot_df_post = pivot_df_post.fillna(0.0)

# get lists for silhouettes scores and inertias
wcss_pre = []
wcss_post = []
silhouette_scores_pre = []
silhouette_scores_post = []


# In[14]:


# Finding out the optimal K, pre-covid origins
k_range = range(2, 8) # Test k from 2 to 8
print('Starting...')
for k in k_range:
        print('Loop begins...')
        kmeans = TimeSeriesKMeans(n_clusters=k, metric='dtw', random_state=42,
                                  n_init=5, max_iter=42) # n_init suppresses warning
        kmeans.fit(pivot_df_pre)
        print('Adding WCSS...')
        wcss_pre.append(kmeans.inertia_) # Inertia is the WCSS
        score = silhouette_score(pivot_df_pre, kmeans.labels_, metric='dtw')
        print('Adding silhouette scores...')
        silhouette_scores_pre.append(score)
        


# In[15]:


# plot elbow method inertias, pre-covid origins
print('Plotting elbow methods...')
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
axes[0].plot(k_range, wcss_pre, marker='o', linestyle='--')
axes[0].set_title('Elbow Method for Optimal k (2014-2019)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].grid(True)
    
# plot silhouette scores
print('Plotting silhouette scores...')
axes[1].plot(k_range, silhouette_scores_pre, marker='o', linestyle='--')
axes[1].set_title('Silhouette Score for Optimal k (2014–2019)')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Average Silhouette Score')
axes[1].grid(True)

plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/ElbowSilhouette_preCov_opd_normsha.png',
            dpi=500, bbox_inches='tight')
    


# In[16]:


# Finding out the optimal K, post-covid origins
k_range = range(2, 8) # Test k from 2 to 8
print('Starting...')
for k in k_range:
        print('Loop begins...')
        kmeans = TimeSeriesKMeans(n_clusters=k, metric='dtw', random_state=42,
                                  n_init=5, max_iter=42) # n_init suppresses warning
        kmeans.fit(pivot_df_post)
        print('Adding WCSS...')
        wcss_post.append(kmeans.inertia_) # Inertia is the WCSS
        score = silhouette_score(pivot_df_post, kmeans.labels_, metric='dtw')
        print('Adding silhouette scores...')
        silhouette_scores_post.append(score)


# In[17]:


# plot elbow method inertias, post-covid origins
print('Plotting elbow methods...')
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
axes[0].plot(k_range, wcss_post, marker='o', linestyle='--')
axes[0].set_title('Elbow Method for Optimal k (2020-2022)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].grid(True)
    
# plot silhouette scores
print('Plotting silhouette scores...')
axes[1].plot(k_range, silhouette_scores_post, marker='o', linestyle='--')
axes[1].set_title('Silhouette Score for Optimal k (2020–2022)')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Average Silhouette Score')
axes[1].grid(True)

plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/ElbowSilhouette_postCov_opd_normsha.png',
            dpi=500, bbox_inches='tight')
    


# In[18]:


"""Clustering pre-Covid, origins"""

# initiate the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# fit the model
print("Fitting model")
clusters = model.fit_predict(pivot_df_pre)
    
# get silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df_pre), clusters, metric='precomputed')

# create empty dataframe for silhouette scores
sco_df_opd_pre = pd.DataFrame()

# loop over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # get silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # put into dataframe
        sco_df_opd_pre.at[label, 'cluster'] = label
        sco_df_opd_pre.at[label, 'score'] = score
    
# Add cluster labels to the DataFrame
pivot_df_pre['cluster'] = clusters
    
# create a copy of dataframe
nudf = pivot_df_pre.reset_index()

# create dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# assign cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
opd_2014_2019['cluster'] = opd_2014_2019['NUTS3'].apply(lambda x: clusterd[x])

# Plot
print("Plotting")
g = sns.lmplot(data=opd_2014_2019, x='year', y='norm_shannon', hue='cluster', 
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, origins per destination regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_opd_normsha.png',
          dpi=500, bbox_inches='tight')


# In[19]:


# Checking the silhouette scores per cluster
sco_df_opd_pre


# In[34]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_opd_pre = opd_2014_2019.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_opd_pre.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_opd_pre


# In[20]:


"""Clustering post-Covid, origins"""

# initiate the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# fit the model
print("Fitting model")
clusters = model.fit_predict(pivot_df_post)
    
# get silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df_post), clusters, metric='precomputed')

# create empty dataframe for silhouette scores
sco_df_opd_post = pd.DataFrame()

# loop over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # get silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # put into dataframe
        sco_df_opd_post.at[label, 'cluster'] = label
        sco_df_opd_post.at[label, 'score'] = score
    
# Add cluster labels to the DataFrame
pivot_df_post['cluster'] = clusters
    
# create a copy of dataframe
nudf = pivot_df_post.reset_index()

# create dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# assign cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
opd_2020_2022['cluster'] = opd_2020_2022['NUTS3'].apply(lambda x: clusterd[x])

# Plot
print("Plotting")
g = sns.lmplot(data=opd_2020_2022, x='year', y='norm_shannon', hue='cluster', 
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, origins per destination regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_normsha_opd.png',
          dpi=500, bbox_inches='tight')


# In[21]:


# Checking the silhouette scores per cluster
sco_df_opd_post


# In[33]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_opd_post = opd_2020_2022.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_opd_post.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_opd_post


# ### Destinations per origins

# In[22]:


# Destinations per origins with years 2014–2019
dpo_2014_2019 = dpo[dpo['year'].between(2014, 2019)]

# Destinations per origins with years 2020–2022
dpo_2020_2022 = dpo[dpo['year'].between(2020, 2022)]


# In[23]:


# Pivot the data
pivot_df_pred = dpo_2014_2019.pivot(index='NUTS3', columns='year', values='norm_shannon')
pivot_df_postd = dpo_2020_2022.pivot(index='NUTS3', columns='year', values='norm_shannon')
    
# Handle missing values (fill with 0 for simplicity)
pivot_df_pred = pivot_df_pred.fillna(0.0)
pivot_df_postd = pivot_df_postd.fillna(0.0)

# get lists for silhouettes scores and inertias
wcss_pred = []
wcss_postd = []
silhouette_scores_pred = []
silhouette_scores_postd = []


# In[24]:


# Finding out the optimal K
k_range = range(2, 8) # Test k from 2 to 8
print('Starting...')
for k in k_range:
        print('Loop begins...')
        kmeans = TimeSeriesKMeans(n_clusters=k, metric='dtw', random_state=42,
                                  n_init=5, max_iter=42) # n_init suppresses warning
        kmeans.fit(pivot_df_pred)
        print('Adding WCSS...')
        wcss_pred.append(kmeans.inertia_) # Inertia is the WCSS
        score = silhouette_score(pivot_df_pred, kmeans.labels_, metric='dtw')
        print('Adding silhouette scores...')
        silhouette_scores_pred.append(score)
        


# In[25]:


# plot elbow method inertias
print('Plotting elbow methods...')
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
axes[0].plot(k_range, wcss_pred, marker='o', linestyle='--')
axes[0].set_title('Elbow Method for Optimal k (2014-2019)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].grid(True)
    
# plot silhouette scores
print('Plotting silhouette scores...')
axes[1].plot(k_range, silhouette_scores_pred, marker='o', linestyle='--')
axes[1].set_title('Silhouette Score for Optimal k (2014–2019)')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Average Silhouette Score')
axes[1].grid(True)

plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/ElbowSilhouette_preCov_dpo_normsha.png',
            dpi=500, bbox_inches='tight')


# In[26]:


"""Clustering pre-Covid, destinations"""

# initiate the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# fit the model
print("Fitting model")
clusters = model.fit_predict(pivot_df_pred)
    
# get silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df_pred), clusters, metric='precomputed')

# create empty dataframe for silhouette scores
sco_df_dpo_pre = pd.DataFrame()

# loop over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # get silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # put into dataframe
        sco_df_dpo_pre.at[label, 'cluster'] = label
        sco_df_dpo_pre.at[label, 'score'] = score
    
# Add cluster labels to the DataFrame
pivot_df_pred['cluster'] = clusters
    
# create a copy of dataframe
nudf = pivot_df_pred.reset_index()

# create dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# assign cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
dpo_2014_2019['cluster'] = dpo_2014_2019['NUTS3'].apply(lambda x: clusterd[x])

# Plot
print("Plotting")
g = sns.lmplot(data=dpo_2014_2019, x='year', y='norm_shannon', hue='cluster', 
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, destinations per origin regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_normsha_dpo.png',
          dpi=500, bbox_inches='tight')


# In[27]:


# Checking the silhouette scores per cluster
sco_df_dpo_pre


# In[32]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_dpo_pre = dpo_2014_2019.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_dpo_pre.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_dpo_pre


# In[28]:


"""Clustering post-Covid, destinations"""

# initiate the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# fit the model
print("Fitting model")
clusters = model.fit_predict(pivot_df_postd)
    
# get silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df_postd), clusters, metric='precomputed')

# create empty dataframe for silhouette scores
sco_df_dpo_post = pd.DataFrame()

# loop over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # get silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # put into dataframe
        sco_df_dpo_post.at[label, 'cluster'] = label
        sco_df_dpo_post.at[label, 'score'] = score
    
# Add cluster labels to the DataFrame
pivot_df_postd['cluster'] = clusters
    
# create a copy of dataframe
nudf = pivot_df_postd.reset_index()

# create dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# assign cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
dpo_2020_2022['cluster'] = dpo_2020_2022['NUTS3'].apply(lambda x: clusterd[x])

# Plot
print("Plotting")
g = sns.lmplot(data=dpo_2020_2022, x='year', y='norm_shannon', hue='cluster', 
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, destinations per origin regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_normsha_dpo.png',
          dpi=500, bbox_inches='tight')


# In[29]:


# Checking the silhouette scores per cluster
sco_df_dpo_post


# In[31]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_dpo_post = dpo_2020_2022.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_dpo_post.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_dpo_post


# In[ ]:




