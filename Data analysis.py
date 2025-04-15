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


# ### Number of origin region occurrences altogether

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

# Flattening the 'orig' column and counting occurrences of each NUTS 3 region as an origin
flat_list = [item for sublist in dpo['dest'] for item in sublist]
counter = Counter(flat_list)
counter # All years


# In[3]:


opd['input'] = opd['orig'].apply(lambda x: list(Counter(x).values()))
dpo['input'] = dpo['dest'].apply(lambda x: list(Counter(x).values()))
dpo.head()


# In[4]:


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

    max_entropy = math.log2(1514)
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

# Origins per destination
for i, row in opd.iterrows():
    opd.at[i, 'simpson'] = skbio.diversity.alpha.simpson(row['input'])
    opd.at[i, 'norm_shannon'] = normalized_shannon_entropy(row['orig'])
    opd.at[i, 'shannon'] = skbio.diversity.alpha.shannon(row['input'])
    opd.at[i, 'uniques'] = skbio.diversity.alpha.observed_features(row['input'])
    
# Destinations per origin
for i, row in dpo.iterrows():
    dpo.at[i, 'simpson'] = skbio.diversity.alpha.simpson(row['input'])
    dpo.at[i, 'norm_shannon'] = normalized_shannon_entropy(row['dest'])
    dpo.at[i, 'shannon'] = skbio.diversity.alpha.shannon(row['input'])
    dpo.at[i, 'uniques'] = skbio.diversity.alpha.observed_features(row['input'])
    
# Save data
#opd.to_csv('/Users/maijahavusela/Desktop/gradu/data/pythonista/origins_per_destination_nuts3.csv')
    
dpo['uniques'].mean()


# In[5]:


# All nuts 3, origins per destination
# Set Seaborn style
sns.set_style('whitegrid')

# Customize grid appearance
plt.rcParams['grid.color'] = '#eeeeee'     # Very light grey
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8           # Dimmer with transparency

fig, ax = plt.subplots(figsize=(8, 6))

# Plot line in orange
sns.lineplot(data=opd, x='year', y='uniques', ax=ax, color='pink')

# Labels and title
ax.set_title("Unique Origins Per NUTS 3 Destination Regions", fontsize=14, fontname='Arial')
ax.set_ylabel("Number of Unique Origin Regions", fontname='Arial')
ax.set_xlabel('Year', fontname='Arial')

# Save to file
#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/UniqueOrigins_nuts3_per_dest.png',
            #dpi=500, bbox_inches='tight')


# In[ ]:





# In[6]:


# All nuts 3, destinations per origin
# Set Seaborn style
sns.set_style('whitegrid')

# Customize grid appearance
plt.rcParams['grid.color'] = '#eeeeee'     # Very light grey
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.8           # Dimmer with transparency

fig, ax = plt.subplots(figsize=(8, 6))

# Plot line
sns.lineplot(data=dpo, x='year', y='uniques', ax=ax, color='pink')

# Labels and title
ax.set_title("Unique Destinations Per NUTS 3 Origin Regions", fontsize=14, fontname='Arial')
ax.set_ylabel("Number of Unique Destination Regions", fontname='Arial')
ax.set_xlabel('Year', fontname='Arial')

# Save to file
#plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/UniqueDestinations_nuts3_per_orig.png',
            #dpi=500, bbox_inches='tight')


# In[7]:


# Pivot the data
pivot_df = opd.pivot(index='NUTS3', columns='year', values='norm_shannon')
    
# Handle missing values (fill with 0 for simplicity)
pivot_df = pivot_df.fillna(0.0)

# get lists for silhouettes scores and inertias
wcss = []
silhouette_scores = []

pivot_df


# In[8]:


# scale the data if "pivot_df" doesnt work
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_df)

# init the model
print("Initializing model")
model = TimeSeriesKMeans(n_clusters=4, metric="dtw", n_init=15, max_iter=42, random_state=42)

# fit the model
print("Fitting model")
clusters = model.fit_predict(pivot_df)
    
# get silhouette scores
silhs = silhouette_samples(cdist_dtw(pivot_df), clusters, metric='precomputed')

# create empty dataframe for silhouette scores
sco_df = pd.DataFrame()

# loop over cluster count
print("Calculating silhouette scores..")
for label in range(4):
        
        # get silhouette score for current cluser
        score = silhs[clusters == label].mean()
        
        # put into dataframe
        sco_df.at[label, 'cluster'] = label
        sco_df.at[label, 'score'] = score
    
# Add cluster labels to the DataFrame
pivot_df['cluster'] = clusters
    
# create a copy of dataframe
nudf = pivot_df.reset_index()

# create dictionary of NUTS 3 codes and cluster labels
clusterd = dict(zip(nudf['NUTS3'], nudf['cluster']))

# assign cluster labels to NUTS 3 region polygons
print("Assigning clusters to original data")
opd['cluster'] = opd['NUTS3'].apply(lambda x: clusterd[x])

# Plot
print("Plotting")
g = sns.lmplot(data=opd, x='year', y='norm_shannon', hue='cluster', 
               height=6, aspect=1.2, order=1, scatter=False)

g.set_axis_labels("Year", "Normalized Shannon Entropy")
g.fig.suptitle("Shannon entropy per cluster", y=1.02)

# Save the plot
g.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends.png',
          dpi=300, bbox_inches='tight')


# In[11]:


orange_shades = ['#ffe5b4',  # light peach
                 '#ffcc80',  # light orange
                 '#ff9900',  # standard orange
                 '#cc6600']  # darker burnt orange

plt.figure(figsize=(10, 6))
print("Plotting with orange shades")

for i, cluster in enumerate(sorted(opd['cluster'].unique())):
    subset = opd[opd['cluster'] == cluster]
    sns.regplot(data=subset, x='year', y='norm_shannon', 
                scatter=False, order=1, label=f'Cluster {cluster}', color=orange_shades[i])

plt.title("Shannon entropy per cluster")
plt.xlabel("Year")
plt.ylabel("Normalized Shannon Entropy")
plt.legend(title='Cluster')

plt.savefig('/Users/maijahavusela/Desktop/gradu/maps and graphs/graphs/Cluster_trends_orange.png',
            dpi=300, bbox_inches='tight')
plt.show()


# In[10]:


#k_range = range(2, 14) # Test k from 2 to 14
#for k in k_range:
        #kmeans = TimeSeriesKMeans(n_clusters=k, metric='dtw', random_state=42,
                                  #n_init=5, max_iter=42) # n_init suppresses warning
        #kmeans.fit(pivot_df)
        #wcss.append(kmeans.inertia_) # Inertia is the WCSS
        #score = silhouette_score(pivot_df, kmeans.labels_, metric='dtw')
        #silhouette_scores.append(score)
# plot elbow method inertias
#fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
#axes[0].plot(k_range, wcss, marker='o', linestyle='--')
#axes[0].set_title('Elbow Method for Optimal k ({})'.format(col))
#axes[0].set_xlabel('Number of Clusters (k)')
#axes[0].set_ylabel('WCSS (Inertia)')
#axes[0].grid(True)
    
# plot silhouette scores
#axes[1].plot(k_range, silhouette_scores, marker='o', linestyle='--')
#axes[1].set_title('Silhouette Score for Optimal k ({})'.format(col))
#axes[1].set_xlabel('Number of Clusters (k)')
#axes[1].set_ylabel('Average Silhouette Score')
#axes[1].grid(True)


# ### Number of origin region occurrences per destination NUTS 3

# In[ ]:


# Dictionary to store the Counter results for each destination NUTS 3 region
origin_count_per_dest_nuts3 = {}

# Iterating over each unique destination NUTS 3 region
for nuts3 in opd['NUTS3'].unique():
    # Getting all regions in the 'orig' column for this NUTS 3 region
    regions = [region for sublist in opd[opd['NUTS3'] == nuts3]['orig'] for region in sublist]
    # Counting occurrences of each origin NUTS 3 region using Counter
    origin_count_per_dest_nuts3[nuts3] = Counter(regions)
    
origin_count_per_dest_nuts3 # All years


# ### The Shannon-Weiner diversity index (not normalized)
# Shannon equitability index?? Se on normalisoitu 0 ja 1 välille

# In[ ]:


# Defining a function to calculate the Shannon-Weiner index using skbio
def calculate_shannon_weiner_with_skbio(region_counter):
    counts = list(region_counter.values())  # Converting counter to a list of counts
    shannon_index = skbio.diversity.alpha.shannon(counts)
    return shannon_index

# Calculating Shannon-Weiner index for each destination NUTS 3 region
diversity_indexes = {}
for nuts3, counter in origin_count_per_dest_nuts3.items():
    shannon_index = calculate_shannon_weiner_with_skbio(counter)
    diversity_indexes[nuts3] = shannon_index

# Output the results
for nuts3, index in diversity_indexes.items():
    print(f"Shannon-Weiner index for {nuts3}: {index:.4f}")
    
# 0 on ei yhtään diversiteettiä, mitä korkeempi arvo sitä korkeempi diversiteetti


# ### Clustering

# In[ ]:


# Converting the dictionary to dataframe
df = pd.DataFrame(list(diversity_indexes.items()), columns=['nuts3', 'shannon_index'])

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=2)
df['cluster'] = dbscan.fit_predict(df[['shannon_index']])

# Unique cluster labels (including -1 for noise)
unique_clusters = sorted(df['cluster'].unique())

# Choosing a colour map for the clusters
colors = plt.cm.get_cmap('Oranges', len(unique_clusters))

# Plotting each cluster
plt.figure(figsize=(10, 2))
for i, cluster in enumerate(unique_clusters):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(
        cluster_data['shannon_index'],
        [0]*len(cluster_data),
        label=f'Cluster {cluster}' if cluster != -1 else 'Noise',
        color=colors(i),
        edgecolor='black',
        s=50
    )

plt.yticks([])  # 1D, no need for y
plt.xlabel('Shannon-Weiner Index')
plt.title('DBSCAN Clustering of Shannon-Weiner Diversity Index')
plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5), # Outside of graph
    title='Clusters'
)
plt.tight_layout()
plt.show()


# In[ ]:




