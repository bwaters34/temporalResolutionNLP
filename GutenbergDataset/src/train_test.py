
# Imports
from os import listdir, path
from shutil import move
from collections import defaultdict
from random import random, shuffle
from copy import copy
import sys


# Move everything back into the main dir
for f in listdir('../Yearly/Train'):
	move('../Yearly/Train/'+f, '../Yearly/'+f)
for f in listdir('../Yearly/Test'):
	move('../Yearly/Test/'+f, '../Yearly/'+f)
for f in listdir('../Yearly/Extra'):
	move('../Yearly/Extra/'+f, '../Yearly/'+f)


# Move all the 1500s into their own folder
#	and 2000s to excess (there are only a few)
for doc in listdir('../Yearly'):
	if doc.endswith('.txt') and (doc.startswith('15') or doc.startswith('14')):
		move('../Yearly/' + doc, '../Yearly/1500s/'+ doc)
	elif doc.endswith('.txt') and doc.startswith('20'):
		move('../Yearly/' + doc, '../Yearly/Extra/'+ doc)

# Dictionary to store the file sizes by year
# Will contain (ID, Size) pairs
doc_sizes = defaultdict(list)
bin_size = 100
# Add to the bins
for doc in [x for x in listdir('../Yearly') if x.endswith('.txt')]:
	# Get the file information
	year = int(doc[:4])
	code = doc[5:-4]
	size = path.getsize('../Yearly/' + doc)
	# Determine the bin
	bin = int((year - 1600) / bin_size)
	# Store in the bin
	doc_sizes[bin].append( (size, code, year) )

# Determine the smallest bin
min_bin = 99999999999
for bin in doc_sizes:
	acc = 0.0
	for s, c, y in doc_sizes[bin]:
		acc += s
	min_bin = min(min_bin, acc)

# Now fill in the bins
for bin in doc_sizes:
	# Some variables
	train = []
	test = []
	diff = 9999999

	# Try to build the bins 1000s times
	# Take the one which is closest to min_bin
	for i in range(1000):
		# Some variables
		tr = []
		te = []
		size_tr = 0.0
		size_te = 0.0
		
		# Shuffle the files so its not the same each time
		shuffle(doc_sizes[bin])

		# Randomlly allocate them, until min_bin is reached
		for size, code, year in doc_sizes[bin]:
			# If there is still space in the training set, add it
			if size_tr < (min_bin * 0.75):
				tr.append( (code, year) )
				size_tr += size

			# Else if there is space in the testing set, add it to that
			elif size_te < (min_bin * 0.25):
				te.append( (code, year) )
				size_te += size

			# Else, done with this bin
			else:
				break
				
		# determine if better setting
		if abs((size_tr / (size_tr + size_te)) - 0.75) < diff:
			diff = abs((size_tr / (size_tr + size_te)) - 0.75)
			train = copy(tr)
			test = copy(te)
	
	# print bin*bin_size + 1499, ':', diff
	# Move the files
	for code,year in train:
		move('../Yearly/%s-%s.txt' % (year, code), '../Yearly/Train/%s-%s.txt'% (year, code))

	for code,year in test:
		move('../Yearly/%s-%s.txt' % (year, code), '../Yearly/Test/%s-%s.txt'% (year, code))

# Move the remaining files into extra
for doc in listdir('../Yearly'):
	if doc.endswith('.txt'):
		move('../Yearly/' + doc, '../Yearly/Extra/' + doc)