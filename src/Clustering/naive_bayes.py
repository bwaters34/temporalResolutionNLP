
# Imports
import math
import numpy as np
from os import listdir
from numpy import argmax, array
from nltk import pos_tag, word_tokenize
from tokenize import tokenize
# --------------------------Some helper functions--------------------------

def bin_mean(start, bin, bin_size):
	return int(start + bin*bin_size + bin_size/2)


# -------------------------- Naive Bayes --------------------------
class NaiveBayes:

	# Constructor
	def __init__(self, bin_size=20.0, alpha=1.0, prediction_func=bin_mean):
		# Class fields
		self.bin_size = bin_size
		self.alpha = alpha
		self.prediction_func = prediction_func

		# Build the class dictionaries
		# Vocabulary - set of all tokens every observed
		self.vocab_size = 0.0 # Will be set when trained

		# Array counting the number of documents observed in each bin
		self.bin_total_doc_counts = None
		self.total_docs = 0.0 # Will be set when trained

		# Array counting the number of tokens observed in each class
		self.bin_total_word_counts = None

		# Array counting, for each class, the number of times each tokens
		#	occured in the training corpus
		self.bin_word_counts = None

	"""
	------------------------------------------------------------------------

	Train/Test functions

	------------------------------------------------------------------------

	"""

	def fit(self, X, y):
		"""
		Function which builds the class dictionaries given the features X
		and cooresponding labels y
		"""

		# Determine the min year
		self.min_year = min(y)
		self.max_year = max(y)

		# Determine the number of bins
		self.B = int((max(y)-min(y)) / self.bin_size) + 1

		# Create the arrays to store counts now that we know the shape
		#   of the features
		N, M = X.shape
		self.bin_total_doc_counts = np.zeros(self.B)
		self.bin_total_word_counts = np.zeros(self.B)
		self.bin_word_counts = np.zeros((self.B, M))

		# Start to build the NB dictionaries
		for i in range(N):
			# Determine the year
			year = y[i]

			# Determine the bin
			bbin = int((year - self.min_year) / self.bin_size)

			# Update the class dictionaries
			# Update number of docs seen for this bin
			self.bin_total_doc_counts[bbin] += 1

			# Update the individual token count for this bin
			for j in range(M):
				self.bin_word_counts[bbin][j] += X[i][j]

			# Update the total token count for this bin
			self.bin_total_word_counts[bbin] += np.sum(X[i])

		# Set the class vocab size
		self.vocab_size = M
		# Set the overall corpus size
		self.total_docs = N
		return self


	def predict(self, X):
		"""
		Function which predicts the yeard of the test set
		"""
		predictions = []
		bins = np.arange(self.B)

		N, M = X.shape
		
		# Go through the files and predict
		for i in range(N):
			# Predict the bin
			bin_probs = [self.unnormalized_log_posterior(X[i], bbin) for bbin in bins]

			# Determine the most likely bin
			best_bin = bins[argmax(bin_probs)]

			# Determine the best bin and append to the prediction list
			predictions.append( self.prediction_func(self.min_year, best_bin, self.bin_size) )

		# Return the prediction
		return array(predictions)

	"""
	Some helper functions for predictions

	Taken from HW1 NB implementation
	"""
	def prob_word_given_bin(self, i, bbin):
		"""
		Returns the probability of the token given the bin, using the class'
		pseudocount alpha
		"""
		num = self.bin_word_counts[bbin][i] + self.alpha
		denom = self.bin_total_word_counts[bbin] + self.alpha*self.vocab_size
		return num / denom

	def log_likelihood(self, x, bbin):
		"""
		Returns the log log likelihood of a bin given a set of tokens
		"""
		acc = 0.0
		# Loop through the tokens and add the log prob
		#	to the accumulator
		for i in range(len(x)):
			acc += math.log(self.prob_word_given_bin(i, bbin)) * x[i]
		# Return the log prob
		return acc

	def log_prior(self, bbin):
		"""
		Returns the log of the prior probability of a given bin
		"""
		return math.log( (self.bin_total_doc_counts[bbin]+1) / self.total_docs)

	def unnormalized_log_posterior(self, x, bbin):
		"""
		Computes the log of P(tokens | bin)
		"""
		return self.log_prior(bbin) + self.log_likelihood(x, bbin)






