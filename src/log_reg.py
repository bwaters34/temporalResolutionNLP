
# Imports
import math
from collections import defaultdict
from os import listdir
from numpy import argmax, array
from math import ceil, exp

# --------------------------Some helper functions--------------------------
def sigmoid(x):
	try:
		return 1 / (1 + exp(-1*x))
	except OverflowError:
		return 1

def simple_tokenizer(doc): 
	bow = defaultdict(float)
	tokens = [t.lower() for t in doc.split()]
	for t in tokens:
		bow[t] += 1
	# Add the bias term
	bow['__BIAS'] = 1
	return bow

def dict_dot_prod(d1, d2):
	smaller = d1 if len(d1)<d2 else d2
	acc = 0.0
	for k in smaller:
		acc += d1[k] * d2[k]
	return acc

def dict_add(d1, d2, f=lambda x: x):
	"""
	Add dict 2 to dict 1
	Returns dict 1
	"""
	smaller = d1 if len(d1)<d2 else d2
	for k in smaller:
		d1[k] += f(d2[k])
	return d1


# -------------------------- Logistic Regression --------------------------
class MultiClass:

	# Constructor
	def __init__(self, bin_size=20.0, learn_rate=0.1, tokenizer=simple_tokenizer):
		# Class fields
		self.bin_size = bin_size
		self.learn_rate = learn_rate
		self.tokenize = tokenizer

		# Weight vector for each class
		# Basically a binary log reg for each class
		# "One vs. All" implementation
		self.weights = defaultdict(Worker)

	"""
	------------------------------------------------------------------------

	Train/Test functions

	------------------------------------------------------------------------

	"""
	def fit(self, train_dir, filenames, numpasses=1, verbose=False):
		"""
		Function which builds the class dictionaries given the directory of the
		training set


		Note: Files must be in the form [Year]-[Code].txt for the classifier 
		to work
		"""
		
		# Determine the min year
		min_y = min([ int(f[:4]) for f in filenames])
		max_y = max([ int(f[:4]) for f in filenames])
		bins = int(ceil((max_y - 1.0*min_y)/self.bin_size))

		# Build the workers for each of the bins
		bin_years = [min_y + i * self.bin_size for i in range(bins)]
		for y in bin_years:
			self.weights[y] = Worker(self.bin_size, y, self.learn_rate)

		if verbose:
			print 'Beginning Training.'

		# A counter for verbose
		counter = 0
		
		# Perform a pass through for each numpasses
		for iteration in range(numpasses):

			if verbose:
				print '\tStarting Pass', iteration+1

			# Go through each file and learn predict
			for filename in filenames:
				counter += 1

				# Determine the year
				gold = int(filename[:4])
				
				# Build the classifier
				with open(train_dir + '/' + filename, 'r') as infile:
					# Read the document
					doc = infile.read()

					# Tokenize the doc
					feats = self.tokenize(doc)

					# Update the weights of all the worker classifiers
					for bin in self.weights:
						# get the worker
						worker = self.weights[bin]

						# Determine if the worker likes these feats
						pred = worker.predict(feats)

						# If yes but the worker is incorrect, move down the gradient
						if worker.is_in_bin(gold) and (pred == 0):
							worker.update(feats)

						# If no but the worker is incorrect, move up the gradient
						elif not worker.is_in_bin(gold) and (pred == 1):
							worker.update(feats, mod=-1)

				# If a multiple of 50 and verbose, alert the user
				if verbose and ((counter % 50) == 0):
					print '\tFinished', counter, 'docs.'

			# If verbose, state the we have finished iteration i
			if verbose:
				print '\tFinished Pass', iteration+1
		

		if verbose:
			print 'Finished Training.'

		# As to be the same as sklearn implementation (i.e. more standardized)
		return self


	def predict_helper(self, feats):
		# Get the (year, score) pairs
		scores = [(worker.get_year(), worker.score(feats)) for worker in self.weights.values()]
		# Return the highest score year
		(year, _) = max(scores, key=lambda x: x[1])
		return year


	def predict(self, test_dir, filenames, verbose=False):
		"""
		Function which predicts the yeard of the test set, given the 
		test dir.

		Again, files must be in the form [Year]-[Code].txt for the classifier 
		to work
		"""
		predictions = []

		# A counter for verbose
		counter = 0

		if verbose:
			print 'Beginning Prediction.'

		# Go through the files and predict
		for filename in filenames:
			# update the counter
			counter += 1

			with open(test_dir + '/' + filename, 'r') as infile:
				# Read the doc
				doc = infile.read()

				# Tokenize
				feats = self.tokenize(doc)

				# Predict the year
				pred = self.predict_helper(feats)

				# Append to the prediction list
				predictions.append( pred )

			# If a multiple of 100 and verbose, alert the user
			if verbose and (counter % 100 == 0):
				print '\tFinished', counter, 'docs.'

		# Return the prediction
		return array(predictions)



class Worker:
	"""
	The units of the multi class classifier. The goal of an object of this
	class is to determine if a book belongs to a given time period.

	It is very simple, but was made separate for clarity
	"""

	def __init__(self, bin_size, year, lr):
		# Set the weights
		self.weights = defaultdict(float)
		# Set the start year and bin_size
		self.year = year
		self.bin_size = bin_size
		# Set the learning rate
		self.learn_rate = lr

	def score(self, feats):
		# Calculate the dot product and return 
		return dict_dot_prod(feats, self.weights)

	def predict(self, feats):
		score = self.score(feats)
		return 1 if (score > 0) else 0

	def update(self, feats, mod=1):
		self.weights = dict_add(self.weights, feats, lambda x: mod*(self.learn_rate)*x)

	def get_year(self):
		return self.year + self.bin_size/2

	def is_in_bin(self, year):
		return (year > self.year) and (year < (self.year + self.bin_size))