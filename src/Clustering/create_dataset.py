# -*- coding: utf-8 -*-
from os import makedirs, listdir
from os.path import exists
from math import ceil, log
from collections import defaultdict
import codecs
import cPickle as cpk
import re

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
train_size = 247
test_size = 88

# Some dictionaries
word_bin_counts = defaultdict(lambda: defaultdict(float))
word_total_counts = defaultdict(float)
vocab = set()

# Some directory names
root = '../../%s' % ds
clst_dir = '%s/Clusters'

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
    # Add the the default dict
    for t in tokens:
        t = mutate(t)
        if t != '':
            word_bin_counts[t][bbin] += 1
            word_total_counts[t] += 1
            vocab.add(t)

# Now the test data
for f in listdir(src_test):
    # Determine the year
    year = determine_bin(int(f[:4]), min_year, bin_size)
    # Get the text
    with codecs.open('%s/%s' % (src_test, f), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add the the default dict
    for t in tokens:
        t = mutate(t)
        if t != '':
            word_bin_counts[t][bbin] += 1
            word_total_counts[t] += 1
            vocab.add(t)

print 'Dictionaries built.'
print 'Writing Probabilities to file'

vocab = sorted(list(vocab)) # switch to a list now
num_bins = int(ceil((max_year-min_year)/bin_size))
threshold = 50

# Now write the results to file
# pfile - csv file with app the probs
# wfile - the words associated with the probs - for easy loading into memory
with open('%s/%probs.csv' % (clst_dir), 'w') as pfile:
    with open('%s/words.txt' % (clst_dir), 'w') as wfile:
        for i in range(len(vocab)):
            # Get the word
            word = vocab[i]
            # Get the total number of counts for the word
            wtc = word_total_counts[word]
            # If the word appeared enough times, write it to file
            if (wtc > threshold):
                # Write the word first
                wfile.write(word + '\n')
                # Now all of the percentages
                buffer = ''
                for j in range(num_bins):
                    # Number of times the word appeared in the bin
                    wbc = word_bin_counts[word][j]
                    # determine and write the probability
                    prob = wbc/wtc
                    # write the prob
                    buffer += ('%0.4f,' % prob)
                # Last Comma becomes newline
                buffer = buffer[:-1] + '\n'
                # Write to the file
                pfile.write(buffer)
                
            # Just so we generally know whats going on
            if i % 100000 == 0:
                print 'Tried %d words' % i
# Delete large data structs we dont need..hopefully this is recursive
del vocab
del word_bin_counts
del word_total_counts
                
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

# Set a seed for reproducibility
np.random.seed(16180)

#Load train and test data
X, Y = load_data(root)

# Create a KMeans obj with K=10 clusters
print 'Creating KMeans cluster model'
K = 10
cluster = KMeans(n_clusters=K,random_state=0, n_jobs=-1).fit(X)

# Predit the clusters of each of the words
print 'Predicting the clusters for each word'
Z = cluster.predict(X)

# Write the clusters to file, just so they exist
# Also add them to the lookup dictionary
print 'Creating dictionary to lookup work clusters'
lkup = defaultdict(int)
with open('%s/word-clusters.txt' % clst_dir, 'w') as outfile:
    if i in range(len(Y)):
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

# First the train_set. Note: +1 for the goldlabels
# We will put the year as the LAST element in the array
train = np.zeros((train_size, K+1))

# Get the files
print 'Building the Train Set'
files = listdir(src_train)
for i in range(files):
    # Determine the year
    train[i][-1] = int(f[:4])
    # Get the text
    with codecs.open('%s/%s' % (src_train, files[i]), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add train matrix
    count = 0
    for t in tokens:
        t = mutate(t)
        z = lkup[t]
        train[i][z] += 1
        count += 1.0
    # Perhaps we should marginalize by the number of words in the book?
    train[i] /= count

# Write this to the train file
N, M = train.shape
with open('%s/train.txt' % clst_dir, 'w') as tfile:
    for in range(N):
        buffer = ''
        for j in range(M):
            buffer += '%0.6f,' % train[i][j]
        # newline instead of comma
        buffer = buffer[:-1] + '\n'
        # write to file
        tfile.write(buffer)

        
# now the test set
test = np.zeros((train_size, K+1))

# Get the files
print 'Building the Test Set'
files = listdir(src_test)
for i in range(files):
    # Determine the year
    test[i][-1] = int(f[:4])
    # Get the text
    with codecs.open('%s/%s' % (src_test, files[i]), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add train matrix
    count = 0
    for t in tokens:
        t = mutate(t)
        z = lkup[t]
        test[i][z] += 1
        count += 1.0
    # Perhaps we should marginalize by the number of words in the book?
    test[i] /= count

# Write this to the train file
N, M = train.shape
with open('%s/test.txt' % clst_dir, 'w') as tfile:
    for in range(N):
        buffer = ''
        for j in range(M):
            buffer += '%0.6f,' % test[i][j]
        # newline instead of comma
        buffer = buffer[:-1] + '\n'
        # write to file
        tfile.write(buffer)
        
        
    
            
