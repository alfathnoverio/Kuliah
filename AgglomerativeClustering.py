#!/usr/bin/env python
# coding: utf-8

# In[55]:


#library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[56]:


## Data sebelumnya memakai data secara acak

jumlahData = 30

def dataRandom():
    x = []
    y = []
    for i in range(jumlahData):
        x.append(np.random.randint(0, 100))
        y.append(np.random.randint(0, 100))
    return x,y

x,y = dataRandom()

random = pd.DataFrame({
    'x': x,
    'y': y
})

## Data tidak acak

data = pd.DataFrame({
    'x': [91, 68, 86, 33, 58, 17, 70, 5, 40, 73,
         18, 17, 92, 86, 44, 20, 64, 22, 21, 26,
         49, 83, 2, 71, 30, 49, 46, 47, 81, 3],
    'y': [69, 34, 66, 51, 87, 66, 72 ,84, 33, 77,
         93, 74, 62, 25, 20, 31, 17, 9, 71, 38,
         30, 94, 35, 29, 4, 39, 44, 100, 17, 20]
})

df = data #'data' untuk data biasa atau "random" untuk data acak


# In[57]:


# Membuat Dendogram
dendrogram = sch.dendrogram(sch.linkage(df, method = 'ward'))


# In[58]:


# plot dari datasets
plt.scatter(df['x'],df['y'])


# In[59]:


# Pembagian berdasarkan agglomerative clustering

hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(df)
plt.scatter(df['x'],df['y'], c=hc.labels_, cmap='rainbow')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




