import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from os import listdir
from math import exp

def load_data(root, ds):
    
    X = np.genfromtxt('%s/%s-probs.csv' % (root, ds), delimiter=',')
    Y = np.genfromtxt('%s/%s-words.txt' % (root, ds), dtype='str')
    
    return X, Y

def cluster_quality(X,Z,K):
    '''
    Compute a cluster quality score given a data matrix X (N,D), a vector of 
    cluster indicators Z (N,), and the number of clusters K.
    '''    
    # Determine the dimensions of the input data
    N, D = X.shape
    
    # Determine the number of instances per cluster
    counts = np.bincount(Z, minlength=K)
        
    # Compute the within-cluster sum of squares
    distances = np.zeros(K)
    for i in range(0, N):
        for j in range(0, i):
            if (Z[i] == Z[j]):
                distances[Z[i]] += np.linalg.norm(X[i] - X[j]) ** 2  
    
    # Return the within cluster sum of squares
    return np.sum(distances / counts)

    
def cluster_proportions(Z,K):
    '''
    Compute the cluster proportions p such that p[k] gives the proportion of
    data cases assigned to cluster k in the vector of cluster indicators Z (N,).
    The proportion p[k]=Nk/N where Nk are the number of cases assigned to
    cluster k. Output shape must be (K,)
    '''
    return np.bincount(Z, minlength=K).astype(float) / Z.shape[0]
        
        
def cluster_means(X,Z,K):
    '''
    Compute the mean or centroid mu[k] of each cluster given a data matrix X (N,D), 
    a vector of cluster indicators Z (N,), and the number of clusters K.
    mu must be an array of shape (K,D) where mu[k,:] is the average of the data vectors
    (rows) that are assigned to cluster k according to the indicator vector Z.
    If no cases are assigned to cluster k, the corresponding mean value should be zero.
    '''
    # Determine the dimensions of the input data
    N, D = X.shape

    # Create an array for the cluster centers
    centers = np.zeros((K, D))
    
    # Determine the number of instances of each cluster
    counts = np.bincount(Z, minlength=K)
    
    # Loop through the data matrix a first time to calculate the cluster centers
    for i in range(0, N):
        centers[Z[i]] += 1.0 * X[i] / counts[Z[i]]
        
    return centers
    
def show_means(mu,p):
    '''
    Plot the cluster means contained in mu sorted by the cluster proportions 
    contained in p.
    '''
    K = p.shape[0]
    f = plt.figure(figsize=(16,16))
    for k in range(K):
        plt.subplot(8,5,k+1)
        plt.plot(mu[k,:])
        plt.title("Cluster %d: %.3f"%(k,p[k]),fontsize=5)
        plt.gca().set_xticklabels([])
        plt.gca().set_xticks([25,50,75,100,125,150,175])
        plt.gca().set_yticklabels([])
        plt.gca().set_yticks([-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
        plt.ylim(-0.2,1.2)
        plt.grid(True)
    plt.tight_layout()
    return f
        
def determine_K(X, K, tau=0.15): 
    '''
    '''
    # Create a copy of the X matrix so it can be shuffled
    X_c = np.copy(X)  
    
    # Shuffle the training data, just in case
    np.random.shuffle(X_c)
    
    # Create variables to store the previous quality
    prev  = 0
    
    # Calculate dQ for each value of K and return the first $K$ s.t.
    #   dQ/Q < tau
    for k in range(1, K+1):
        cluster = KMeans(n_clusters=k,random_state=0).fit(X_c)
        curr = cluster_quality(X_c, cluster.predict(X_c), k)
        # If this is K = 1, then continue
        if (k == 1):
            prev = curr
            continue
        # If the quality has started to decrease by less than tau, return 
        #   the current value of K
        elif ((prev - curr)/prev < tau):
            return k
        # Else, update prev
        else:
            prev = curr
            
    # If we have reached this point, then the quality was still decreasing.
    # Therefore, return the highest allowed value of K
    return K
        
    
    
        
    