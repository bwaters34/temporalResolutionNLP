"""
Tokenize.py

Contains function for extraction of the features from the Train/Test dirs
"""
# Imports
from cPickle import loads
from collections import defaultdict

# --------------------------Some helper functions--------------------------

def load_dict(filename):
	# read the file
	with open(filename, 'r') as f:
		doc = f.read()
	# tokenize
	return loads(doc)

def tokenize(root_dir, features, filename):
	# initially empty
	bow = defaultdict(float)
	# Unigrams
	if 'unigrams' in features:
		bow.update( load_dict('%s/%s/%s' % (root_dir, 'Unigrams', filename)) )
	# Bigrams
	if 'bigrams' in features:
		bow.update( load_dict('%s/%s/%s' % (root_dir, 'Bigrams', filename)) )
	# POS-Tags
	if 'pos' in features:
		bow.update( load_dict('%s/%s/%s' % (root_dir, 'POS-Tags', filename)) )

	# done!
	return bow