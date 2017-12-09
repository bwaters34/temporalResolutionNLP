
# Imports
import math
from collections import defaultdict
from os import listdir
from numpy import argmax, array
from nltk import pos_tag, word_tokenize
from tokenize import tokenize
# --------------------------Some helper functions--------------------------

def bin_mean(start, bin, bin_size):
	return int(start + bin*bin_size + bin_size/2)

def count_bow(bow):
    return sum(bow.values())


# -------------------------- Naive Bayes --------------------------
class NaiveBayes:

	# Constructor
	def __init__(self, bin_size=20.0, alpha=1.0, features=['uni'], prediction_func=bin_mean):
		# Class fields
		self.bin_size = bin_size
		self.alpha = alpha
		self.features = features
		self.prediction_func = prediction_func

		# Build the class dictionaries
		# Vocabulary - set of all tokens every observed
		self.vocab = set()
		self.vocab_size = 0.0 # Will be set when trained

		# Dictionary counting the number of documents observed in each bin
		self.bin_total_doc_counts = defaultdict(float)
		self.total_docs = 0.0 # Will be set when trained

		# Dictionary counting the number of tokens observed in each class
		self.bin_total_word_counts = defaultdict(float)

		# Dictionary counting, for each class, the number of times each tokens
		#	occured in the training corpus
		self.bin_word_counts = defaultdict(lambda: defaultdict(float) )

	"""
	------------------------------------------------------------------------

	Train/Test functions

	------------------------------------------------------------------------

	"""

	def fit(self, train_dir, filenames=None, verbose=False):
		"""
		Function which builds the class dictionaries given the directory of the
		training set


		Note: Files must be in the form [Year]-[Code].txt for the classifier 
		to work
		"""
		
		# If no filenames given, us all in Unigrams
		#	which will be the same for all names in the other folders
		if not filenames:
			filenames = listdir('%s/Unigrams' % train_dir)

		# Determine the min year
		self.min_year = min([ int(f[:4]) for f in filenames])
		self.max_year = max([ int(f[:4]) for f in filenames])

		if verbose:
			print 'Beginning Training.'

		# A counter for verbose
		counter = 0
		
		# Start to build the NB dictionaries
		for filename in filenames:
			# Increase the counter 
			counter += 1

			# Determine the year
			year = int(filename[:4])
			
			# Determine the bin
			bin = int((year - self.min_year) / self.bin_size)
			
			# Extract tokens from the document
			bow = tokenize(train_dir, self.features, filename)

			# Update the class dictionaries
			# Update number of docs seen for this bin
			self.bin_total_doc_counts[bin] += 1
			
			# Update the individual token count for this bin
			for token in bow:
				self.vocab.add(token)
				self.bin_word_counts[bin][token] += bow[token]
			
			# Update the total token count for this bin
			self.bin_total_word_counts[bin] += count_bow(bow)

			# If a multiple of 50 and verbose, alert the user
			if verbose and (counter % 50 == 0):
				print '\tFinished', counter, 'docs.'

		# Set the class vocab size
		self.vocab_size = len(self.vocab)
		# Set the overall corpus size
		self.total_docs = sum(self.bin_total_doc_counts.values())

		if verbose:
			print 'Finished Training.'

		# For 
		return self


	def predict(self, test_dir, filenames=None, verbose=False):
		"""
		Function which predicts the yeard of the test set, given the 
		test dir.

		Again, files must be in the form [Year]-[Code].txt for the classifier 
		to work
		"""

		# If no filenames given, use all in Unigrams
		if not filenames:
			filenames = listdir('%s/Unigrams' % test_dir)

		predictions = []
		bins = self.bin_total_doc_counts.keys()

		# A counter for verbose
		counter = 0

		if verbose:
			print 'Beginning Prediction.'

		# Go through the files and predict
		for filename in filenames:
			# update the counter
			counter += 1

			# Tokenize
			bow = tokenize(test_dir, self.features,  filename)

			# Predict the bin
			bin_probs = [self.unnormalized_log_posterior(bow, bin) for bin in bins]

			# Determine the most likely bin
			best_bin = bins[argmax(bin_probs)]

			# Determine the best bin and append to the prediction list
			predictions.append( self.prediction_func(self.min_year, best_bin, self.bin_size) )

			# If a multiple of 50 and verbose, alert the user
			if verbose and (counter % 50 == 0):
				print '\tFinished', counter, 'docs.'

		# Return the prediction
		return array(predictions)

	"""
	Some helper functions for predictions

	Taken from HW1 NB implementation
	"""
	def prob_word_given_bin(self, token, bin):
		"""
		Returns the probability of the token given the bin, using the class'
		pseudocount alpha
		"""
		num = self.bin_word_counts[bin][token] + self.alpha
		denom = self.bin_total_word_counts[bin] + self.alpha*self.vocab_size
		return num / denom

	def log_likelihood(self, bow, bin):
		"""
		Returns the log log likelihood of a bin given a set of tokens
		"""
		acc = 0.0
		# Loop through the tokens and add the log prob
		#	to the accumulator
		for token in bow:
			acc += math.log(self.prob_word_given_bin(token, bin)) * bow[token]
		# Return the log prob
		return acc

	def log_prior(self, bin):
		"""
		Returns the log of the prior probability of a given bin
		"""
		return math.log( self.bin_total_doc_counts[bin] / self.total_docs)

	def unnormalized_log_posterior(self, bow, bin):
		"""
		Computes the log of P(tokens | bin)
		"""
		return self.log_prior(bin) + self.log_likelihood(bow, bin)






