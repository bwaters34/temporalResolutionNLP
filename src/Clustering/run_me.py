from random import random
import numpy as np
import matplotlib.pyplot as plt

# sklearn Imports
from sklearn.cluster import KMeans
from cluster_utils import load_data, cluster_means

# Set a seed for reproducibility
np.random.seed(16180)

#Load train and test data
X, Y = load_data('../../ClusterDataset/cluster-GutenbergDataset.txt')

# Create a cluster with K cluster
K = 5
cluster = KMeans(n_clusters=K,random_state=0, n_jobs=-1).fit(X)
Z = cluster.predict(X)
mu = cluster_means(X, Z, K)
#
plt.figure(1, figsize=(12,8))
for i in range(0,K):   
    plt.plot(np.arange(20), mu[i], '-b')
#   plt.title('Blah %d' % i)
#plt.ylim(0, 0.01)
plt.grid()
plt.show()
    

"""
# Create a variable to the store the cluster qualities
qualities = np.zeros(39)

# Determine the cluster qualities for increasing numbers of K
for k in range(1, 40):
    print 'Starting k=%d' % k
    cluster = KMeans(n_clusters=k,random_state=0, n_jobs=-1).fit(X)
    predictions = cluster.predict(X)
    q1_qualities[k-1] = cluster_quality(X, predictions, k)

print 'Determining Cluster Qualities for K = 1 to 40'
# Plot the results
plt.figure(0, figsize=(6,4))
plt.plot(np.arange(39) + 1, q1_qualities, '-r')
plt.xlabel('Number of Clusters, K')
plt.ylabel('Cluster Quality (Within Cluster Distance)')
plt.title('Cluster Quality vs. Number of Clusters')
plt.grid()
plt.show()
"""