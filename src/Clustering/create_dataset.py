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

# bad_chars = ['/', '+', '*', '(', ')', '[', ']', '_', '!', '?', '\'', '$', '%', '&', '^', '#', '@', '~', '-', '.', '<', '>', ':', ';', '"', ',', '`', u'“']

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
max_year = 1999
word_bin_counts = defaultdict(lambda: defaultdict(float))
word_total_counts = defaultdict(float)
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
with open('%s/%s-probs.csv' % (clst_dir, ds), 'w') as pfile:
    with open('%s/%s-words.txt' % (clst_dir, ds), 'w') as wfile:
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
    
            
    
    
    

