#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np
import sklearn as sk
import math
from sklearn.cluster import DBSCAN
from scipy.stats import entropy
from sklearn.cluster import KMeans


# In[194]:


#Define a function to read and preprocess data from a given file
def read_file(data,x):
    data = pd.read_csv(data, low_memory=False)
    data['DT'] = pd.to_datetime(data['Date']) + pd.to_timedelta(data['Time'])
    if x == "c":
        data = data[['DT','Sensor Glucose (mg/dL)']]
    else:
        data = data[data['BWZ Carb Input (grams)'] > 0]
        data = data[['DT','BWZ Carb Input (grams)']]
    data = data.sort_values(by='DT')
    data.reset_index(drop=True, inplace=True)
    return data
        
cgm = read_file('CGMData.csv','c')
insulin = read_file('InsulinData.csv','i')
df = pd.merge(cgm,insulin,on='DT',how='outer').sort_values(by='DT')
df.reset_index(drop=True, inplace=True)


# In[195]:


#Define a function to convert insulin values into corresponding label categories
def insulin_label(insulin):
    # Assign labels based on the insulin value ranges
    if 3 <= insulin < 23:
        return 0
    elif 23 <= insulin < 43:
        return 1
    elif 43 <= insulin < 63:
        return 2
    elif 63 <= insulin < 83:
        return 3
    elif 83 <= insulin < 103:
        return 4
    elif 103 <= insulin < 130:
        return 5
    else:
        return None
    
# Initialize an empty NumPy array for labels and an empty DataFrame for data
label = np.array([])
data_df = pd.DataFrame(columns = [i for i in range(24)])
# Iterate through the rows of the DataFrame 'df'
for idx, dt, glucose, bwz in df.itertuples():
    data_sequence =[]
    positive_values = 0
    total_values = 0
    # If the bwz value is greater than 0
    if bwz > 0:
        meal = sum(df.iloc[idx + 1:idx + 25, 2] > 0)
        if meal == 0 and (idx - 8) > 0:
            data_sequence = df.iloc[idx + 1:idx + 25, 1].tolist()
        for value in data_sequence:
            total_values += 1
            if value > 0:
                positive_values += 1
        if positive_values >= 24 * 0.8 and total_values == 24:
            data_df = data_df.append(pd.DataFrame([data_sequence], columns=data_df.columns), ignore_index=True)
            val = insulin_label(bwz)
            if val is not None:
                label = np.append(label, val)


# In[196]:


#calculates various features for each row in the data_df DataFrame and then appends the resulting feature set to a new DataFrame data1_df. 
data1 = np.empty(shape=[0, 4])
label1 = np.array([])
i=0
for i, p_arr in enumerate(data_df.to_numpy()):
    max_value = None
    meal_value = p_arr[0]
    meal_idx = 0
    for idx, num in enumerate(p_arr):
        if (max_value is None or num > max_value) :
            max_value = num
            max_idx = idx
    diff_max1 = None
    diff_max2 = None
    for num in np.diff(p_arr):
        if (diff_max1 is None or num > diff_max1) :
            diff_max1 = num
    for num in np.diff(p_arr, n=2):
        if (diff_max2 is None or num > diff_max2) :
            diff_max2 = num
    features = [
        abs(max_idx - meal_idx) * 5,
        (max_value - meal_value) / meal_value,
        diff_max1,
        diff_max2,
    ]
    if not np.isnan(features).any():
        data1 = np.append(data1, [np.array(features)], axis=0)
        label1 = np.append(label1, label[i])
data1_df = pd.DataFrame(data=data1)


# In[197]:


#create a contingency table for cluster and label counts.
def matrix(df):
    z_df = pd.DataFrame(np.zeros((6, 7)))
    total_sum = 0
    for i in range(6):
        cluster_df = df[df['cluster'] == i]
        counts = cluster_df['label'].value_counts()
        z_df.loc[i, counts.index] = counts.values
        z_df.loc[i, 6] = counts.sum()
        total_sum += counts.sum()
    return z_df, total_sum

# performs KMeans clustering on the data1_df DataFrame and appends the cluster labels to k_df.
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=1)
kmeans.fit(data1_df)
k_df = data1_df.copy()
k_df['label'] = label1
k_df['cluster'] = kmeans.labels_
k_result = k_df.groupby (['cluster','label'])[0].count()
#calculates the contingency table for KMeans results using the matrix function and computes the inertia (SSE) of the KMeans model, 
#appending it to the result list.
k_matrix,total_k = matrix(k_df)
result=[]
result.append(kmeans.inertia_)


# In[198]:


#performs DBSCAN clustering on the data1_df DataFrame and appends the cluster labels to d_df
#calculates the contingency table for DBSCAN results using the matrix function and computes the total count of data points for the DBSCAN clusters.
dbscan = DBSCAN(eps=5.12, min_samples=7, metric='euclidean').fit(data1_df)
d_df = data1_df.copy()
d_df['label'] = label1
d_df['cluster'] = dbscan.labels_
d_result = d_df.groupby (['cluster','label'])[0].count()
d_matrix,total_d = matrix(d_df)

# calculates the SSE (Sum of Squared Errors) for the DBSCAN clustering results. 
data_arr = data1_df.to_numpy()
arr = []
centroids = []
for i in range(6):
    c = 0
    for j in range(len(data_arr)) :
        if d_df.iloc[j,5] == i : 
            arr.append(data_arr[j])
            c += 1        
    centroid = np.sum(arr, axis=0)/c
    centroids.append(centroid)
sse = 0
for i, centroid in enumerate(centroids):
    cluster_points = data1_df[d_df['cluster'] == i].values
    squared_errors = np.sum((cluster_points - centroid) ** 2)
    sse += squared_errors
result.append(sse)


# In[199]:


# calculates entropy for both kmeans and dbscan
def entro(df,k):
    total = 0
    for i in range(6):
        E = 0
        for j in range(6):
            frac = df.iloc[i, j] / df.iloc[i, 6]
            if frac != 0:
                E += -frac*math.log2(frac)
        total += E * df.iloc[i, 6] / k
    return total
k_entropy = entro(k_matrix,total_k)
d_entropy = entro(d_matrix,total_d)
result.append(k_entropy)
result.append(d_entropy)


# In[200]:


# calculates purity for both kmeans and dbscan
def purity_score(df,total):
    max_total = 0
    for i in range(6):
        P_max = 0
        for j in range(6):
            if P_max < df.iloc[i, j]:
                P_max = df.iloc[i, j]            
        max_total += P_max + P_max
    purity= max_total / total
    return purity
k_purity = purity_score(k_matrix,total_k)
d_purity = purity_score(d_matrix,total_d)
result.append(k_purity)
result.append(d_purity)


# In[201]:


result_df = pd.DataFrame(data=[result])
result_df.to_csv('Result.csv', index = False, header=False)


# In[ ]:




