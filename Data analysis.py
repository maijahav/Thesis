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

# ## Years 2014-2022
# ### Origins per destinations

# In[173]:


# Pivoting the data
pivot_df = opd.pivot(index='NUTS3', columns='year', values='norm_shannon') # Origins per destinations
   
# Handling missing values (filling with 0 for simplicity)
pivot_df = pivot_df.fillna(0.0)

# Creating lists for silhouettes scores and inertias
wcss = []
silhouette_scores = []


# In[174]:


"""Clustering"""

# Initiating the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# Fitting the model
print("Fitting model")
clusters = model.fit_predict(pivot_df)
    
# Getting silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df), clusters, metric='precomputed')

# Creating an empty dataframe for silhouette scores
sco_df = pd.DataFrame()

# Looping over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # Getting silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # Putting the score into dataframe
        sco_df.at[label, 'cluster'] = label
        sco_df.at[label, 'score'] = score
    
# Adding cluster labels to the DataFrame
pivot_df['cluster'] = clusters
    
# Creating a copy of dataframe
nudf = pivot_df.reset_index()

# Creating a dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# Assigning cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
opd['cluster'] = opd['NUTS3'].apply(lambda x: clusterd[x])

# Plotting
#print("Plotting")
#g = sns.lmplot(data=opd, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster", y=1.02)

# Saving the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends_k6.png',
          #dpi=500, bbox_inches='tight')


# In[175]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df['mean_entropy'] = pivot_df[[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df['cluster'] = pivot_df['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
opd['cluster'] = opd['NUTS3'].apply(lambda x: clusterd[x])

# Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df['integration_level'] = pivot_df['cluster'].map(cluster_labels)
opd['integration_level'] = opd['cluster'].map(cluster_labels)


# In[176]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Adding a 'cluster_name' column
opd['Cluster description'] = opd['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(data=opd, x='year', y='norm_shannon', hue='Cluster description', hue_order=cluster_order,
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, origins per destination regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_all_opd_normsha.png',
          dpi=500, bbox_inches='tight')


# In[177]:


# Checking the silhouette scores per cluster
sco_df


# In[178]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_opd = opd.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_opd.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_opd


# ### Destinations per origin regions

# In[179]:


# Pivoting the data
pivot_df_d = dpo.pivot(index='NUTS3', columns='year', values='norm_shannon') # Destinations per origins
   
# Handling missing values (filling with 0 for simplicity)
pivot_df_d = pivot_df_d.fillna(0.0)

# Creating lists for silhouettes scores and inertias
wcss_d = []
silhouette_scores_d = []


# In[180]:


"""Clustering"""

# Initiating the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# Fitting the model
print("Fitting model")
clusters = model.fit_predict(pivot_df_d)
    
# Getting silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df_d), clusters, metric='precomputed')

# Creating an empty dataframe for silhouette scores
sco_df_d = pd.DataFrame()

# Looping over cluster count
print("Calculating silhouette scores..")
for label in range():
        
        # Getting silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # Putting the score into dataframe
        sco_df_d.at[label, 'cluster'] = label
        sco_df_d.at[label, 'score'] = score
    
# Adding cluster labels to the DataFrame
pivot_df_d['cluster'] = clusters
    
# Creating a copy of dataframe
nudf = pivot_df_d.reset_index()

# Creating a dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# Assigning cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
dpo['cluster'] = dpo['NUTS3'].apply(lambda x: clusterd[x])

# Plotting
#print("Plotting")
#g = sns.lmplot(data=opd, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster", y=1.02)

# Saving the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends_k6.png',
          #dpi=500, bbox_inches='tight')


# In[181]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df_d['mean_entropy'] = pivot_df_d[[2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df_d.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df_d['cluster'] = pivot_df_d['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df_d.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
dpo['cluster'] = dpo['NUTS3'].apply(lambda x: clusterd[x])

# Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df_d['integration_level'] = pivot_df_d['cluster'].map(cluster_labels)
dpo['integration_level'] = dpo['cluster'].map(cluster_labels)


# In[182]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Adding a 'cluster_name' column
dpo['Cluster description'] = dpo['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(data=dpo, x='year', y='norm_shannon', hue='Cluster description', hue_order=cluster_order,
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, destinations per origin regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_all_dpo_normsha.png',
          dpi=500, bbox_inches='tight')


# In[183]:


# Checking the silhouette scores per cluster
sco_df_d


# In[184]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_dpo = dpo.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_dpo.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_dpo


# In[ ]:


## k_range = range(2, 8) # Test k from 2 to 8
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

# In[61]:


# Origins per destinations with years 2014–2019
opd_2014_2019 = opd[opd['year'].between(2014, 2019)]

# Origins per destinations with years 2020–2022
opd_2020_2022 = opd[opd['year'].between(2020, 2022)]


# In[62]:


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
    


# In[63]:


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
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_opd_normsha.png',
          #dpi=500, bbox_inches='tight')
    


# In[137]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df_pre['mean_entropy'] = pivot_df_pre[[2014, 2015, 2016, 2017, 2018, 2019]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df_pre.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df_pre['cluster'] = pivot_df_pre['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df_pre.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
opd_2014_2019['cluster'] = opd_2014_2019['NUTS3'].apply(lambda x: clusterd[x])

# Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df_pre['integration_level'] = pivot_df_pre['cluster'].map(cluster_labels)
opd_2014_2019['integration_level'] = opd_2014_2019['cluster'].map(cluster_labels)


# In[138]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Adding a 'cluster_name' column
opd_2014_2019['Cluster description'] = opd_2014_2019['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(data=opd_2014_2019, x='year', y='norm_shannon', hue='Cluster description', hue_order=cluster_order,
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, pre-Covid-19 origins per destination regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_opd_normsha.png',
          dpi=500, bbox_inches='tight')


# In[139]:


# Checking the silhouette scores per cluster
sco_df_opd_pre


# In[140]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_opd_pre = opd_2014_2019.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_opd_pre.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_opd_pre


# In[141]:


pivot_df_post


# In[68]:


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
#print("Plotting")
#g = sns.lmplot(data=opd_2020_2022, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster, origins per destination regions", y=1.02)

# Save the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_normsha_opd.png',
          #dpi=500, bbox_inches='tight')


# In[129]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df_post['mean_entropy'] = pivot_df_post[[2020, 2021, 2022]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df_post.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df_post['cluster'] = pivot_df_post['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df_post.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
opd_2020_2022['cluster'] = opd_2020_2022['NUTS3'].apply(lambda x: clusterd[x])

# Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df_post['integration_level'] = pivot_df_post['cluster'].map(cluster_labels)
opd_2020_2022['integration_level'] = opd_2020_2022['cluster'].map(cluster_labels)


# In[130]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Adding a 'cluster_name' column
opd_2020_2022['Cluster description'] = opd_2020_2022['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(
    data=opd_2020_2022, x='year', y='norm_shannon', hue='Cluster description',hue_order=cluster_order,
    height=6, aspect=1.2, order=1, scatter=False, ci=None
).set(xlim=(2019.5, 2022.5), xticks=[2020, 2021, 2022])

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, post-Covid-19 origins per destination regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_opd_normsha.png',
          dpi=500, bbox_inches='tight')


# In[131]:


# Checking the silhouette scores per cluster
sco_df_opd_post


# In[132]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_opd_post = opd_2020_2022.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_opd_post.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_opd_post


# ### Destinations per origins

# In[73]:


# Destinations per origins with years 2014–2019
dpo_2014_2019 = dpo[dpo['year'].between(2014, 2019)]

# Destinations per origins with years 2020–2022
dpo_2020_2022 = dpo[dpo['year'].between(2020, 2022)]


# In[74]:


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


# In[75]:


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
#print("Plotting")
#g = sns.lmplot(data=dpo_2014_2019, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster, destinations per origin regions", y=1.02)

# Save the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_normsha_dpo.png',
          #dpi=500, bbox_inches='tight')
    


# In[126]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df_pred['mean_entropy'] = pivot_df_pred[[2014, 2015, 2016, 2017, 2018, 2019]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df_pred.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df_pred['cluster'] = pivot_df_pred['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df_pred.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
dpo_2014_2019['cluster'] = dpo_2014_2019['NUTS3'].apply(lambda x: clusterd[x])

# Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df_pred['integration_level'] = pivot_df_pred['cluster'].map(cluster_labels)
dpo_2014_2019['integration_level'] = dpo_2014_2019['cluster'].map(cluster_labels)


# In[127]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Adding a 'cluster_name' column
dpo_2014_2019['Cluster description'] = dpo_2014_2019['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(data=dpo_2014_2019, x='year', y='norm_shannon', hue='Cluster description', hue_order=cluster_order,
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, pre-Covid-19 destinations per origin regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_preCov_dpo_normsha.png',
          dpi=500, bbox_inches='tight')


# In[133]:


# Checking the silhouette scores per cluster
sco_df_dpo_pre


# In[134]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_dpo_pre = dpo_2014_2019.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_dpo_pre.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_dpo_pre


# In[79]:


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
        
        # get silhouette score for current cluster
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
#print("Plotting")
#g = sns.lmplot(data=dpo_2020_2022, x='year', y='norm_shannon', hue='cluster', 
               #height=6, aspect=1.2, order=1, scatter=False)

#g.set_axis_labels("Year", "Normalized Shannon Entropy")
#g.fig.suptitle("Normalized Shannon entropy per cluster, destinations per origin regions", y=1.02)

# Save the plot
#g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_normsha_dpo.png',
          #dpi=500, bbox_inches='tight')


# In[120]:


"""Making sure the cluster labels match"""

# Step 1: Calculate mean entropy per region
pivot_df_postd['mean_entropy'] = pivot_df_postd[[2020, 2021, 2022]].mean(axis=1)

# Step 2: Calculate average entropy per cluster
entropy_by_cluster = pivot_df_postd.groupby('cluster')['mean_entropy'].mean().reset_index()

# Step 3: Sort clusters by average entropy
entropy_by_cluster = entropy_by_cluster.sort_values('mean_entropy').reset_index(drop=True)

# Step 4: Create remapping dictionary (lowest entropy = 0, highest = 3)
label_remap = {row['cluster']: new_label for new_label, row in entropy_by_cluster.iterrows()}

# Step 5: Apply remapping
pivot_df_postd['cluster'] = pivot_df_postd['cluster'].map(label_remap)

# Step 6: Update downstream cluster dict
nudf = pivot_df_postd.reset_index()
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))
dpo_2020_2022['cluster'] = dpo_2020_2022['NUTS3'].apply(lambda x: clusterd[x])

# (Optional) Add descriptive labels
cluster_labels = {
    0: "poorly integrated",
    1: "less integrated",
    2: "moderately integrated",
    3: "highly integrated"
}
pivot_df_postd['integration_level'] = pivot_df_postd['cluster'].map(cluster_labels)
dpo_2020_2022['integration_level'] = dpo_2020_2022['cluster'].map(cluster_labels)


# In[125]:


"""Enhancing the plot"""
# Defining cluster names
cluster_names = {
    0: "0. Poorly integrated",
    1: "1. Less integrated",
    2: "2. Moderately integrated",
    3: "3. Highly integrated"
}
# Get cluster descriptions in order by cluster number
cluster_order = [cluster_names[i] for i in sorted(cluster_names.keys())][::-1]

# Now map it
dpo_2020_2022['Cluster description'] = dpo_2020_2022['cluster'].map(cluster_names)

# Plotting
print("Plotting")
g = sns.lmplot(
    data=dpo_2020_2022, x='year', y='norm_shannon', hue='Cluster description',hue_order=cluster_order,
    height=6, aspect=1.2, order=1, scatter=False, ci=None
).set(xlim=(2019.5, 2022.5), xticks=[2020, 2021, 2022])

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Normalized Shannon entropy per cluster, post-Covid-19 destinations per origin regions", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/clusters/Clusters_postCov_dpo_normsha.png',
          dpi=500, bbox_inches='tight')


# In[135]:


# Checking the silhouette scores per cluster
sco_df_dpo_post


# In[136]:


# Counting the number of unique NUTS3 per cluster
nuts3_per_cluster_dpo_post = dpo_2020_2022.groupby('cluster')['NUTS3'].nunique().reset_index()
nuts3_per_cluster_dpo_post.columns = ['cluster', 'unique_NUTS3_count']
nuts3_per_cluster_dpo_post


# # Typology
# ## Destinations per origin regions

# In[144]:


# Creating a dataframe with a cluster typology for each NUTS 3 region, destinations per origin regions
pred = dpo_2014_2019.sort_values(by=['NUTS3', 'year']).groupby('NUTS3').last().reset_index() # Taking the last one
pred = pred[['NUTS3', 'Cluster description']]
pred = pred.rename(columns={'Cluster description': 'Pre-COVID Cluster'})

postd = dpo_2020_2022.sort_values(by=['NUTS3', 'year']).groupby('NUTS3').last().reset_index() # Taking the last one
postd = postd[['NUTS3', 'Cluster description']]
postd = postd.rename(columns={'Cluster description': 'Post-COVID Cluster'})

# Merge
combined_dest = pd.merge(pred, postd, on='NUTS3', how='outer')


# In[159]:


# Create the 'typology' column by combining the cluster numbers from both columns
combined_dest['typology'] = combined_dest.apply(
    lambda row: f"{str(row['Pre-COVID Cluster'][0])}_{str(row['Post-COVID Cluster'][0])}" 
    if pd.notna(row['Pre-COVID Cluster']) and pd.notna(row['Post-COVID Cluster']) 
    else f"{str(row['Pre-COVID Cluster'][0])}_x" 
    if pd.notna(row['Pre-COVID Cluster']) 
    else f"x_{str(row['Post-COVID Cluster'][0])}" 
    if pd.notna(row['Post-COVID Cluster']) 
    else "x_x", axis=1
)

# Display the updated dataframe
combined_dest[0:20]


# ## Origins per destination regions

# In[146]:


# Creating a dataframe with a cluster typology for each NUTS 3 region, destinations per origin regions
pre = opd_2014_2019.sort_values(by=['NUTS3', 'year']).groupby('NUTS3').last().reset_index() # Taking the last one
pre = pre[['NUTS3', 'Cluster description']]
pre = pre.rename(columns={'Cluster description': 'Pre-COVID Cluster'})

post = opd_2020_2022.sort_values(by=['NUTS3', 'year']).groupby('NUTS3').last().reset_index() # Taking the last one
post = post[['NUTS3', 'Cluster description']]
post = post.rename(columns={'Cluster description': 'Post-COVID Cluster'})

# Merge
combined_orig = pd.merge(pre, post, on='NUTS3', how='outer')


# In[160]:


# Create the 'typology' column by combining the cluster numbers from both columns
combined_orig['typology'] = combined_orig.apply(
    lambda row: f"{str(row['Pre-COVID Cluster'][0])}_{str(row['Post-COVID Cluster'][0])}" 
    if pd.notna(row['Pre-COVID Cluster']) and pd.notna(row['Post-COVID Cluster']) 
    else f"{str(row['Pre-COVID Cluster'][0])}_x" 
    if pd.notna(row['Pre-COVID Cluster']) 
    else f"x_{str(row['Post-COVID Cluster'][0])}" 
    if pd.notna(row['Post-COVID Cluster']) 
    else "x_x", axis=1
)

combined_orig.to_csv('/Users/maijahavusela/Desktop/gradu/data/pythonista/OPD_typology_pre_post.csv')

# Display the updated dataframe
combined_orig[0:20]


# In[ ]:




