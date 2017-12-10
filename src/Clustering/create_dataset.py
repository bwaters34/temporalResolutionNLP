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

def mean(x):
    return sum([float(t) for t in x])/len(x)

bad_chars = ['/', '+', '*', '(', ')', '[', ']', '_', '!', '?', '\'', '$', '%', '&', '^', '#', '@', '~', '-', '.', '<', '>', ':', ';', '"', ',', '`', u'“']

regex = re.compile('[^a-zA-Z]')

def mutate(word):
#    for c in bad_chars:
#        word = word.replace(c, '')
#    return word
    return regex.sub('', word).lower()
    

# Location for the dataset
ds = 'GutenbergDataset'
bin_size = 20.0
min_year = 1600
max_year = 1899
bin_word_counts = defaultdict(lambda: defaultdict(float))
bin_total_counts = defaultdict(float)
vocab = set()

# Some directory names
root = '../../%s' % ds
clst_dir = '../../Clustering'

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
            bin_word_counts[bbin][t] += 1
            bin_total_counts[bbin] += 1
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
            bin_word_counts[bbin][t] += 1
            bin_word_counts[t][bbin] += 1
            bin_total_counts[bbin] += 1
            vocab.add(t)

print 'Dictionaries built.'
print 'Writing Probabilities to file'

vocab = sorted(list(vocab)) # switch to a list now
threshold = 100

# Now write the results to file
with open('%s/cluster-%s.txt' % (clst_dir, ds), 'w') as outfile:
    for i in range(len(vocab)):
        # if we have enough instances, write to file
        appearances = [bin_word_counts[j][vocab[i]] for j in range(len(bin_total_counts))]
        # Yeah this is a comment
        if mean(appearances) > threshold:
            # Continue writing to the file
            # Write the word first
            outfile.write(vocab[i] + ',')
            # Now all of the percentages
            for j in range(len(bin_total_counts)):
                num = bin_word_counts[j][vocab[i]]
                denom = bin_total_counts[j]
                try:
                    log_prob = log(num/denom)
                except:
                    print i, j
                outfile.write('%0.4f,' % log_prob)
            # Newline!
            outfile.write('\n')

            if i % 100000 == 0:
                print 'Added %d words', i
    
            
    
    
    

