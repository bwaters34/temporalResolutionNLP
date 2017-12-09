import numpy as np
import matplotlib.pyplot as plt

# sklearn Imports
from sklearn.cluster import KMeans

# Set a seed for reproducibility
np.random.seed(16180)

#Load train and test data
train = np.load("../../Data/ECG/train.npy")
test = np.load("../../Data/ECG/test.npy")

#Create train and test arrays
Xtr = cluster_utils.transform(train[:,0:-1])
Xte = cluster_utils.transform(test[:,0:-1])
Ytr = np.array(map(int, train[:,-1]))
Yte = np.array( map(int, test[:,-1]))

# === Cluster Quality ===

# Create a variable to the store the cluster qualities
q1_qualities = np.zeros(39)

# Determine the cluster qualities for increasing numbers of K
for k in range(1, 40):
    cluster = KMeans(n_clusters=k,random_state=0).fit(Xtr)
    predictions = cluster.predict(Xtr)
    q1_qualities[k-1] = cluster_utils.cluster_quality(Xtr, predictions, k)

print 'Determining Cluster Qualities for K = 1 to 40'
# Plot the results
plt.figure(0, figsize=(6,4))
plt.plot(np.arange(39) + 1, q1_qualities, '-r')
plt.xlabel('Number of Clusters, K')
plt.ylabel('Cluster Quality (Within Cluster Distance)')
plt.title('Cluster Quality vs. Number of Clusters')
plt.grid()
plt.show()


# === 1.5 Cluster Number Optimization ===
print 'Determining the Optimal K'
K = cluster_utils.determine_K(Xtr, 40)

# Build a KMeans obj. for the given K on the Xtr data
clst = KMeans(n_clusters=K,random_state=0).fit(Xtr)