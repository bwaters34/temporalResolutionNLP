# Imports
from os import listdir, path
from shutil import move
from collections import defaultdict
from random import random, shuffle
from copy import copy
import matplotlib.pyplot as plt

def stdev(lst):
	mean = 1.0*sum(lst)/len(lst)
	return sum([(x-mean) ** 2 for x in lst])/len(lst)

class Organizer:

	def __init__(self, bin_size, start):
		self.bin_size = bin_size
		self.start = start
		self.bins = defaultdict(list)

	def build(self, fpath):
		for doc in [x for x in listdir(fpath) if x.endswith('.txt')]:
			# Get the file information
			year = int(doc[:4])
			code = doc[5:-4]
			size = path.getsize('../Yearly/' + doc)
			# Add to the organizer
			self.add(year,code,size)
		# Set the smallest bin
		self.set_smallest_bin_size()

	def set_smallest_bin_size(self):
		min_bin = 99999999999
		for bin in self.bins:
			acc = self.get_bin_size(bin)
			min_bin = min(min_bin, acc)
		self.min_bin = min_bin

	def determine_bin(self, year):
		return int((year - self.start)/self.bin_size)

	def add(self, year, code, size):
		bin = self.determine_bin(year)
		self.bins[bin].append( (year,code,size) )

	def total_count(self):
		return sum([len(x) for x in self.bins.values()])

	def get_bin_size(self, b):
		acc = 0
		for _,_,s in self.bins[b]:
			acc += s
		return acc

	def bin_stdev(self):
		return stdev([len(x) for x in self.bins.values()])

	def score(self):
		try:
			return float(self.total_count() - self.bin_stdev())
		except ZeroDivisionError:
			return 0.0

	def move(self, tpath):
		for b in self.bins:
			for year,code,size in self.bins[b]:
				move('../Yearly/%s-%s.txt' % (year,code), '../Yearly/%s/%s-%s.txt' % (tpath,year,code))

	def print_stats(self):
		for b in self.bins:
			year = self.start + bin_size * b
			count = len(self.bins[b])
			size = self.get_bin_size(b)
			print 'Bin:', year
			print '\tCount:', count
			print '\tSize:', size

	def plot_stats(self):
		print self.total_count(), self.bin_stdev()
		years = []
		counts = []
		sizes = []
		for b in self.bins:
			years.append ( self.start + bin_size * b )
			counts.append ( len(self.bins[b]) )
			sizes.append ( self.get_bin_size(b) )

		# Plot counts
		plt.figure(1, figsize=(6,4))
		plt.bar(years, counts)
		plt.show()
		plt.figure(2, figsize=(6,4))
		plt.bar(years, sizes)
		plt.show()


# PARAMETERS
bin_size = 50 # Bin size for the organizer
year_i = 1600 # First year to include
year_f = 1900 # First century to exclude


# Move everything back into the main dir
for f in listdir('../Yearly/Train'):
	move('../Yearly/Train/'+f, '../Yearly/'+f)
for f in listdir('../Yearly/Test'):
	move('../Yearly/Test/'+f, '../Yearly/'+f)
for f in listdir('../Yearly/Extra'):
	move('../Yearly/Extra/'+f, '../Yearly/'+f)

# Move all the files we dont want into the Extra bin
for doc in listdir('../Yearly'):
	if doc.endswith('.txt'):
		year = int(doc[:4])
		if year not in range(year_i, year_f):
			move('../Yearly/' + doc, '../Yearly/Extra/' + doc)

# Organizer object to store all the (year,code,size fields)
org = Organizer(50, year_i)
org.build('../Yearly')

# Now randomly build organizers and choose the one which maximizes
#	0.5 * (0.75 * tr + 0.25 * te) - 0.5 * stdev(bin_sizes)
best_tr = Organizer(bin_size, year_i)
best_te = Organizer(bin_size, year_i)

for i in range(1000):
	# Make a new organizer
	new_tr = Organizer(bin_size, year_i)
	new_te = Organizer(bin_size, year_i)

	# Add stuff to each of the bins frm the overall organizer
	for b in org.bins:
		# Shuffle the bin
		shuffle(org.bins[b])

		# Randomlly allocate them, until min_bin is reached
		for year, code, size in org.bins[b]:
			# If there is still space in the training set, add it
			if new_tr.get_bin_size(b) < (org.min_bin * 0.75):
				new_tr.add(year,code,size)

			# Else if there is space in the testing set, add it to that
			elif new_te.get_bin_size(b) < (org.min_bin * 0.25):
				new_te.add(year,code,size)

			# Else, done with this bin
			else:
				break

	# Check to see if these one is better matching
	if (0.75*new_tr.score() + 0.25 * new_te.score()) > (0.75*best_tr.score() + 0.25 * best_te.score()):
		best_tr = new_tr
		best_te = new_te		

# Move the docs to the train and test
best_tr.move('Train')
best_te.move('Test')

# Move the remaining files into extra
for doc in listdir('../Yearly'):
	if doc.endswith('.txt'):
		move('../Yearly/' + doc, '../Yearly/Extra/' + doc)

# Print the stats
best_tr.plot_stats()
best_te.plot_stats()