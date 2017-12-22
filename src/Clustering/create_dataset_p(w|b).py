# -*- coding: utf-8 -*-
from os import makedirs, listdir
from os.path import exists
from math import ceil, log
from collections import defaultdict
import codecs
import cPickle as cpk
import re
import numpy as np

def determine_bin(year, min_year, bin_size):
    return int((year-min_year)/bin_size)

regex = re.compile('[^a-zA-Z]')

def mutate(word):
#    for c in bad_chars:
#        word = word.replace(c, '')
#    return word
    return regex.sub('', word).lower()
    

"""
======================================================================

                            PART 1
                      CSV File Creation

======================================================================
"""
    
# Dataset specific information
# MUST BE SET MANUALLY TO WORK CORRECTLY!  
ds = 'GutenbergDataset'
bin_size = 20.0
min_year = 1600
max_year = 1999

# Some dictionaries
alpha = 1
bin_word_counts = defaultdict(lambda: defaultdict(float))
bin_total_counts = defaultdict(float)
word_doc_counts = defaultdict(int)
vocab = set()

# Some directory names
root = '../../%s' % ds
clst_dir = '%s/Clusters' % root

src_root = '../../%s/%s' % (ds, 'Yearly')
src_train = '%s/Train' % src_root
src_test = '%s/Test' % src_root

# Make the directories, if necessary
if not exists(clst_dir):
	makedirs(clst_dir)

# Go through and build the cluster data
# First the train data
for f in listdir(src_train):
    # Determine the year
    bbin = determine_bin(int(f[:4]), min_year, bin_size)
    # Get the text
    with codecs.open('%s/%s' % (src_train, f), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add the the default dicts
    word_doc_flag = defaultdict(lambda: False) # flag dict
    for t in tokens:
        t = mutate(t)
        if t != '':
            bin_word_counts[bbin][t] += 1
            bin_total_counts[bbin] += 1
            vocab.add(t)
            # If this is first time we have seen the word
            #   for the doc, update word_doc_counts
            if not word_doc_flag[t]:
                word_doc_counts[t] += 1
                word_doc_flag[t] = True
        
# Now the test data
for f in listdir(src_test):
    # Determine the year
    year = determine_bin(int(f[:4]), min_year, bin_size)
    # Get the text
    with codecs.open('%s/%s' % (src_test, f), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add the the default dicts
    word_doc_flag = defaultdict(lambda: False) # flag dict
    for t in tokens:
        t = mutate(t)
        if t != '':
            bin_word_counts[t][bbin] += 1
            bin_total_counts[t] += 1
            vocab.add(t)
            # If this is first time we have seen the word
            #   for the doc, update word_doc_counts
            if not word_doc_flag[t]:
                word_doc_counts[t] += 1
                word_doc_flag[t] = True

print 'Dictionaries built.'
print 'Writing Probabilities to file'

vocab = sorted(list(vocab)) # switch to a list now
vocab_len = len(vocab) # so we dont need to recalc
num_bins = int(ceil((max_year-min_year)/bin_size))
threshold = 3 # Number of documents the word must be in
                # to be included

# Now write the results to file
# pfile - csv file with app the probs
# wfile - the words associated with the probs - for easy loading into memory
with open('%s/probs-rev.csv' % (clst_dir), 'w') as pfile:
    with open('%s/words-rev.txt' % (clst_dir), 'w') as wfile:
        for i in range(len(vocab)):
            # Get the word
            word = vocab[i]
            # If the word appears in enough documents, use it
            if word_doc_counts[word] > threshold:
                # Write the word first
                wfile.write(word + '\n')
                # Now all of the percentages
                probs = np.zeros((num_bins,1))
                for j in range(num_bins):
                    # Number of times the word appeared in the bin
                    wbc = bin_word_counts[j][word] + alpha
                    # determine and write the probability
                    prob = log(wbc/(bin_total_counts[j] + alpha*vocab_len))
                    # write the prob
                    probs[j] = prob
                probs /= -1*np.sum(probs)
                # Now write to file
                buffer = ''
                for j in range(num_bins):
                    buffer += ('%0.6f,' % probs[j])
                # Write to the file
                pfile.write(buffer[:-1] + '\n')
            # Just so we generally know whats going on
            if i % 100000 == 0:
                print 'Tried %d words' % i
# Delete large data structs we dont need..hopefully this is recursive
del vocab
del bin_word_counts
del bin_total_counts
del word_doc_counts
                
print 'Part 1: Done.'
"""
======================================================================

                            PART 2
                          Clustering
                          
Determine the clusters to which each word in the corpus belongs.

======================================================================
"""
    
import numpy as np
import matplotlib.pyplot as plt

# sklearn Imports
from sklearn.cluster import KMeans
from cluster_utils import load_data
from cluster_utils import cluster_means

# Set a seed for reproducibility
np.random.seed(16180)

#Load train and test data
X, Y = load_data(root, rev=True)

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
for i in range(min(K,5)):
    plt.figure(i, figsize=(12,8))
    plt.title('Example Cluster %d: Probability of Word by Bin' % (i+1))
    plt.xlabel('Year')
    plt.ylabel('P(Bin|Word)')
    plt.plot(years, mu[i], '-r')
    plt.grid()
    plt.show()
     
# Write the clusters to file, just so they exist
# Also add them to the lookup dictionary
print 'Creating dictionary to lookup work clusters'
lkup = defaultdict(int)
with open('%s/word-clusters-rev.txt' % clst_dir, 'w') as outfile:
    for i in range(len(Y)):
        outfile.write('%s,%d\n' % (Y[i], Z[i]))
        lkup[Y[i]] = Z[i]

print 'Part 2: Done.'

"""
======================================================================

                            PART 3
                          Train / Test
                          
Count the number of words in each cluster for each document in the
train and test folders.

======================================================================
"""

# Function to period td-idf for a cluster count_vector
def tf_idf(X):
    # Get the row and column sums
    row_sums = np.sum(X[:,:-1], axis=1)
    col_sums = np.sum(X[:,:-1], axis=0)
    total = np.sum(col_sums)
    # Apply my version of tf-idf
    for i in range(X.shape[0]):
        for j in range(X.shape[1]-1):
            tf = X[i][j] / row_sums[i]
            idf = log( total / col_sums[j] )
            X[i][j] = tf * idf
    return X

# First the train_set. 
files = listdir(src_train)
train_size = len(files)

# Note: +1 for the goldlabels
# We will put the year as the LAST element in the array
train = np.zeros((train_size, K+1))

# Get the files
print 'Building the Train Set'
for i in range(train_size):
    # Determine the year
    train[i][-1] = int(files[i][:4])
    # Get the text
    with codecs.open('%s/%s' % (src_train, files[i]), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add train matrix
    count = 0
    for t in tokens:
        t = mutate(t)
        if t != '': 
            z = lkup[t]
            train[i][z] += 1
            count += 1.0

# Apply tf-idf
train = tf_idf(train)
            
# Write this to the train file
N, M = train.shape
with open('%s/train-rev.csv' % clst_dir, 'w') as tfile:
    for i in range(N):
        buffer = ''
        for j in range(M-1):
            buffer += '%0.6f,' % train[i][j]
        # newline instead of comma
        buffer += (str(train[i][j+1]) + '\n')
        # write to file
        tfile.write(buffer)

        
# now the test set
files = listdir(src_test)
test_size = len(files)
test = np.zeros((test_size, K+1))

# Get the files
print 'Building the Test Set'
for i in range(test_size):
    # Determine the year
    test[i][-1] = int(files[i][:4])
    # Get the text
    with codecs.open('%s/%s' % (src_test, files[i]), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add train matrix
    count = 0
    for t in tokens:
        t = mutate(t)
        if t != '': 
            z = lkup[t]
            test[i][z] += 1
            count += 1.0
    # Normalize using term frequency

test = tf_idf(test)

# Write this to the train file
N, M = test.shape
with open('%s/test-rev.csv' % clst_dir, 'w') as tfile:
    for i in range(N):
        buffer = ''
        for j in range(M-1):
            buffer += '%0.6f,' % test[i][j]
        # newline instead of comma
        buffer += (str(train[i][j+1]) + '\n')
        # write to file
        tfile.write(buffer)
        
        
    
            
