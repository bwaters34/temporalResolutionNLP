
from os import makedirs, listdir
from os.path import exists
from collections import defaultdict
import codecs
import cPickle as cpk
from nltk import pos_tag, word_tokenize, sent_tokenize
from parser import parse


"""
Copied them here just so I didnt need to deal with import statements
"""
# Some characters whose lines we will throw out
delimitable = ['\t', '|', '[', ']', '+']
bad_tags = ['.','``', '\'\'', ':']

def simple_tokenizer(doc): 
	bow = defaultdict(float)
	tokens = [t.lower() for t in doc.split()]
	for t in tokens:
		bow[t] += 1
	return bow

def bigram_tokenizer(doc):
	bow = defaultdict(float)
	tokens = [t.lower() for t in doc.split()]
	for i in range(1,len(tokens)):
		bow[(tokens[i-1], tokens[i])] += 1
	return bow

def pos_tag_tokenizer(doc):
	tags_dict = defaultdict(float)

	# Format the document
	doc = doc.replace('\n', ' ')
	doc = doc.replace('\r', ' ')
	doc = doc.replace('  ', ' ')

	# Get the sentences
	sents = sent_tokenize(doc)

	# For all good sentences, pos-tag them
	for f in sents:
		if not any([(x in f) for x in delimitable]):
			tokens = word_tokenize(f)
			tags = pos_tag(tokens)
			# (word, pos) items
			for tag in tags:
				tags_dict[tag] +=1
			# transition counts
			for i in range(1, len(tags)):
				prev = tags[i-1][1]
				curr = tags[i][1]
				tags_dict[(prev,curr)] += 1

	# done!
	return tags_dict

def tree_tokenizer(doc):
	tree_dict = defaultdict(float)

	# Format the document
	doc = doc.replace('\n', ' ')
	doc = doc.replace('\r', ' ')
	doc = doc.replace('  ', ' ')

	# Get the sentences
	sents = sent_tokenize(doc)

	# For all good sentences, pos-tag them
	for f in sents:
		if not any([(x in f) for x in delimitable]):
			tokens = word_tokenize(f)
			tags = [x[1] for x in pos_tag(tokens) if x[1] not in bad_tags]
			# build the tree
			tree = parse(tags)
			# check if -1
			if (tree == -1):
				continue
			for t1 in tree:
				for t2 in tree[t1]:
					tree_dict[(t1, t2)] += 1

	# done!
	return tree_dict
			

# Location for the dataset
ds = 'GutenbergDataset'
loc = 'SentTrees'
func = tree_tokenizer

# Some directory names
root = '../../%s' % ds
train = '%s/Train/%s' % (root, loc)
test = '%s/Test/%s' % (root, loc)

src_root = '../../%s/%s' % (ds, 'Yearly')
src_train = '%s/Train' % src_root
src_test = '%s/Test' % src_root


# Make the directories, if necessary
if not exists(root):
	makedirs(root)
if not exists(train):
	makedirs(train)
if not exists(test):
	makedirs(test)

# Now actually build the dataset
# First the trainset
for filename in listdir(src_train):
	with codecs.open('%s/%s' % (src_train, filename), 'r', encoding='utf-8') as infile:
		# Read the document
		doc = infile.read()
		# Create the Bigrams
		feats = func(doc)
		# Write it to a file
		with open('%s/%s' % (train, filename), 'w') as outfile:
			outfile.write(cpk.dumps(feats))

# Now the test set
for filename in listdir(src_test):
	with codecs.open('%s/%s' % (src_test, filename), 'r', encoding='utf-8') as infile:
		# Read the document
		doc = infile.read().replace('\n', '')
		# Create the Bigrams
		feats = func(doc)
		# Write it to a file
		with open('%s/%s' % (test, filename), 'w') as outfile:
			outfile.write(cpk.dumps(feats))


