# Imports
from cluster_utils import load_data
import math
import numpy as np
import matplotlib.pyplot as plt

bin_size = 20
min_year = 1600

# Dataset location
root = '../../GutenbergDataset'

if False:
    # Load the probabilites for all words
    probs, words = load_data('../../GutenbergDataset', rev=True)
    num_bins = len(probs[0])

    def plot_word(word):
        # Get the probabilities for the word
        ind = np.where(words == word)[0][0]
        # Make the plot
        plt.plot(np.array([min_year + i * bin_size for i in range(num_bins)]), probs[ind], 'g-')
        plt.grid()
        plt.xlabel('Bin Number, B')
        plt.ylabel('Normalized Log Prob')
        plt.title('Prob of %s by Bin' % word.upper())
        plt.show()


    plot_word('and')
    plot_word('machine')
    plot_word('or')


if True:
    # sklearn Imports
    from sklearn.cluster import KMeans
    from cluster_utils import load_data
    from cluster_utils import cluster_means

    # Set a seed for reproducibility
    np.random.seed(16180)

    #Load train and test data
    X, Y = load_data(root, rev=True)
    num_bins = len(X[0])

    # Create a KMeans obj with K=10 clusters
    print 'Creating KMeans cluster model'
    K = 100
    cluster = KMeans(n_clusters=K,random_state=0, n_jobs=-1).fit(X)

    # Predit the clusters of each of the words
    print 'Predicting the clusters for each word'
    Z = cluster.predict(X)

    # Plot the cluster means
    mu = cluster_means(X, Z, K)
    years = [min_year + bin_size*i for i in range(num_bins)]
    for i in [29, 12, 11, 0, 90]:
        plt.figure(i, figsize=(12,8))
        plt.title('Example Cluster %d: Probability of Word by Bin' % i)
        plt.xlabel('Year')
        plt.ylabel('P(Word|Bin)')
        plt.plot(years, mu[i], '-r')
        plt.grid()
        plt.show()


if False:
    # Load the train sets
    train = np.genfromtxt('%s/Clusters/train-rev.csv' % root, delimiter=',')
    N = train.shape[1] - 1

    # Print a few examples
    N = len(train[0]) - 1

    def plot_document(d):
        print 'Year:', train[d][-1]
        plt.plot(np.arange(N), train[d][:-1])
        plt.grid()
        plt.show()

    def plot_bin_mean(t1, t2):
        temp = np.zeros((N, 1))
        count = 0
        for x in train:
            if x[-1] > t1 and x[-1] < t2:
                for i in range(N):
                    temp[i] += x[i]
                count += 1
        print t1, t2
        plt.plot(np.arange(N), temp/count)
        plt.grid()
        plt.show()

    for i in range(10):
        plot_bin_mean(min_year + i * 25, min_year + (i+1) * 25)
        
if False:
    # Load the train sets
    train = np.genfromtxt('%s/Clusters/train-rev.csv' % root, delimiter=',')
    N, M = train.shape
    
    # Determine the top K largest features for each training example
    K = 3
    topK = np.array([np.argsort(train[i][:-1])[-K:][::-1] for i in range(N)])
    
    for l in range(K-1):
        # Histogram it to get scales
        # Get the possible values for this transition
        prev = np.unique(topK[:, l])
        curr = np.unique(topK[:, l+1])
        temp = []
        # Go through and determine what years are in each value
        print '*' * 50
        print "Transition: %s -> %s" % (l,l+1)
        for p in prev:
            for c in curr:
                years = [train[j][-1] for j in range(N) if (topK[j][l] == p) and (topK[j][l+1] == c)]
                if years:
                    print '\t%d - > %d' % (p,c)
                    print '\t%s' % str(years[:10])
                    print '\t\tMean Year: %.2f\n' % np.mean(years)
                    temp.append( (p, c, len(years)) )
        
        # Determine largest transition
        largest = max([float(y) for (_,_,y) in temp])
        # Rescale
        temp = [(p,c,float(y/largest)) for p,c,y in temp]
        # Plot it!
        for p, c, op in temp:
            plt.plot(np.arange(2), np.array([p,c]), 'b-', alpha=op**0.33)
        plt.grid()
        plt.title('Transition %d -> %d' % (l,l+1))
        # Add datalabels
        for p in prev:
            plt.text(0.01, p + (-2) * (p%2), str(p), ha='center', va='bottom')
        for c in curr:
            plt.text(0.99, c + (-2) * (c%2), str(c), ha='center', va='bottom')    
        # Show the plot
        plt.show()
        

# Plots Feature Variance for train set
if False:
    # Load the train sets
    train = np.genfromtxt('%s/Clusters/train-rev.csv' % root, delimiter=',')
    N, M = train.shape
    
    # determine variance
    variance = np.zeros(M-1)
    percent = np.zeros(M-1)
    for i in range(M-1):
        var = np.var(train[:,i]) ** 0.5
        variance[i] = var
        percent[i] = var / np.mean(train[:,i])
        
    plt.bar(np.arange(M-1), variance, 0.75)
    plt.xlabel('Feature Number')
    plt.ylabel('Feature STDEV (Word Counts)')
    plt.title('Word-Cluster Counts: STDEV by Feature')
    plt.grid()
    plt.show()
    print 'Min', min(percent)
    print 'Max', max(percent)
    print 'Avg', np.mean(percent)
            
            

