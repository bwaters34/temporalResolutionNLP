"""
Header
"""

# Libraries
from os import listdir
from random import shuffle
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import sys


def calc_MAE(gold, pred):
	return np.mean( np.abs(np.array(gold) - np.array(pred)) )


def create_folds(l, k):
	for i in range(0, len(l), k):
		yield l[i:i+k]


def cross_validation(train_dir, model_func, hparams, K=5, verbose=False):
	"""
	This function performs k-fold crossvalidation to determine
	the best hyperparameters for the given model
	"""
	# Get the filenames
	# These are the same in all feature directories, so it suffices
	#	to just use the ones in Unigrams
	train_set = listdir('%s/Unigrams' % train_dir)

	# Build all possible ordered pairs of hyperparameters
	hparam_pairs = list(product(*hparams))

	# Arrays to store the train and test losses for each fold
	train_losses = np.zeros((len(hparam_pairs), K))
	test_losses  = np.zeros((len(hparam_pairs), K))

	# Shuffle the training set
	shuffle(train_set)

	# Build the folds
	folds = list(create_folds(train_set, K))

	# For each fold, hold it out and train the model for classifier
	#   using each pair of hyperparameters.
	# Then test each classifier on held-out fold.
	# Record the err for each classifier for each held-out fold
	print '-------------------------------------------------------------'
	print 'Beginning Cross-Validation with ' + str(K) + ' folds.'
	print '-------------------------------------------------------------'
	for i in range(0, K):
		# Setup the folds 
		train = [x for j in range(K) for x in folds[j] if j != i]
		test = folds[i]
		# Train and test the different models on the selected folds
		train_losses[:, i], test_losses[:, i] = test_hyperparameters(train_dir, train, test, model_func, hparam_pairs)
		# Repeat for all folds
		print 'Fold ' + str(i+1) + ' completed.'
		print 'Mean Train Loss:', np.mean(train_losses[:, i])
		print 'Mean Test Loss:', np.mean(test_losses[:, i])
		print '-------------------------------------------------------------'

	# Send the errors to a help function to find the best hyperparameter pair
	(best_params, mean_loss, loss) = determine_best_hyperparameter(hparam_pairs, test_losses, K)
	# If verbose is true, print out the errs on all of the hparam pairs
	if verbose:
		plot_hyperparam_loss(best_params, hparams, np.mean(train_losses, axis=1), mean_loss)
	# Give the user some feedback on the process
	print 'Cross-validation completed.'
	print 'Optimal Hyperparameters:', best_params
	print 'Mean loss on the best hyperparameter setting: ' + str(loss/K)
	print 'Now training the model on the entire test set.'
	return model_func(best_params).fit(train_dir, train_set, verbose=True)



def test_hyperparameters(train_dir, train, test, model_func, hparams):
	"""
	Helper function for all validation methods
	For each hyperparameter setting, it trains the model on the 
	training set and then tests the fitted model on the test set.
	Returns an array of each settings error
	"""

	train_losses = np.zeros(len(hparams))
	test_losses = np.zeros(len(hparams))

	for i in range(0, len(hparams)):
		# Build the regressor for the i'th hparam setting
		clf = model_func(hparams[i])
		clf.fit(train_dir, train)
		# Test the regressor on the test set
		train_losses[i] = calculate_loss(clf, train_dir, train)
		test_losses[i] = calculate_loss(clf, train_dir, test)
		print '\tHyperparameter', hparams[i], 'completed.'

	return (train_losses, test_losses)




def determine_best_hyperparameter(params, losses, num_iterations):
	"""
	# Given all hyper parameter settings and an array of their errors
	# on various sets of tests, returns the hyper parameter with the lowest mean err
	"""
	# Calculate the mean % error
	mean_loss = np.sum(losses, axis=1)/num_iterations
	# Find the index which has the lowest mean error
	best_index = np.argmin(mean_loss)
	# Return the respected hyperparameter settings, the mean errs
	# and the mean error of the best parameter
	# These are all returned for analysis purposes
	best_param = params[ best_index ]

	return (best_param, mean_loss, mean_loss[best_index])


  
def calculate_loss(clf, test_dir, datafiles):
	"""
	Inputs:
		clf - classifer object
		datafiles - data points which the classifier will predict the classes of
	The function computes the validation error given a classifer and a data set
	"""  
	# Determine the gold labels
	# Get the gold labels:
	gold = [int(f[:4]) for f in datafiles]
	# Have the classifier predict the classes of the data
	pred = clf.predict(test_dir, datafiles)
	# count the number of predictions which are incorrect
	return calc_MAE(gold, pred)


def plot_hyperparam_loss(best_params, params, train_loss, test_loss):
	# Count the number of settings we tested
	N = len(params[0])

	best_inds = []

	# Determine the indices of the best hyperparameters other than the main one
	for i in range(1, len(params)):
		best_inds.append( np.argwhere(params[i] == best_params[i]) )

	# Now find 
	product = np.product(best_inds)
	y1 = np.zeros(N)
	y2 = np.zeros(N)
	for i in range(0, N):
		y1[i] = train_loss[i*N + product]
		y2[i] = test_loss[i*N + product]

	plot_num = int(np.random.rand()*100)

	# The main parameter is always the first dimension of the array
	x = params[0]

	# Now plot the results
	plt.figure(plot_num, figsize=(6,4))
	plt.plot(x,y1,'or-', linewidth=3, label='Train Loss') 
	plt.plot(x,y2,'ob-', linewidth=3, label='Test Loss') 
	plt.grid(True) #Turn the grid on
	plt.ylabel("Mean Loss") #Y-axis label
	plt.xlabel("Main Hyperparameter Setting") #X-axis label
	plt.title("Mean Loss vs. Main Hyperparameter Setting") #Plot title
	plt.xlim(min(x) - 0.5, max(x)+0.5) #set x axis range
	plt.ylim(0,max(max(y1),max(y2))+1) #Set yaxis range
	plt.legend(loc='best')
	plt.show()
	return 


def evaluate_model(clf, root_dir, title):
	# Set the bin_size
	bin_size = clf.bin_size

	# Get the gold labels
	gold = np.array([int(f[:4]) for f in listdir('%s/Unigrams' % root_dir)])

	# Predict the test data
	pred = clf.predict(root_dir, verbose=True)
	sys.exit(0)
	
	# Create the century vector
	c1 = int(bin_size*min(x/bin_size for x in gold))
	c2 = int(bin_size*max(x/bin_size for x in gold))
	x = range(c1, c2+1, int(bin_size))

	# Determine the mean abs error per century
	y = [0] * len(x)
	for i in range(len(x)):
		acc = 0.0
		for g, p in zip(gold, pred):
			if (g - x[i]) < bin_size and (g - x[i]) >= 0:
				y[i] += abs(g - p)
				acc += 1
		try:
			y[i] /= acc
		except ZeroDivisionError:
			y[i] = 0

	# Now plot it
	plt.figure(1, figsize=(6,4))
	plt.plot(x, y, 'or-', linewidth=3)
	plt.grid(True)
	plt.title('%s: MAE by Bin' % title)
	plt.xlabel('Century')
	plt.ylabel('MAE in Years')
	plt.show()

	# Print the overall MAE
	print 'Overall MAE:', calc_MAE(gold, pred)

