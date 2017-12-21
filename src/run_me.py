"""
Header

"""

# Imports
import numpy as np
from random import seed
from os import listdir
from naive_bayes import NaiveBayes
from logregpipeline import LogReg
from train_test_helper import cross_validation, evaluate_model

# Set the random seed for reproducibility
seed(314159)

### Dataset 1 : Gutenberg

# Set the train and test dir
train_dir = '../GutenbergDataset/Train'
test_dir = '../GutenbergDataset/Test'

"""
BASELINE: NAIVE BAYES MODEL --------------------------------------------
"""
if True:
	# Function to build a NB model
	NB_func = lambda args: NaiveBayes(bin_size=args[0], alpha=args[1], features=['pos'])

	# These are the hyperparameters we are testing over
	# nb_hparams = [ np.array([5, 10, 20, 35, 50]), # Bin Size
	# 			   np.array([1, 10, 25, 50, 100])] # Alpha
	nb_hparams = [np.array([5,10]),  # Bin Size
				  np.array([1, 3])]  # Alpha

	# Create the classifier using 5-fold CV
	nb = cross_validation(train_dir, NB_func, nb_hparams, verbose=True)

	# Evaluate the model
	evaluate_model(nb, test_dir, 'Gutenberg-POS Tags')

	### Dataset 2 : Proquest

	# Set the train and test dir
	train_dir = '../ProquestDataset/Train'
	test_dir  = '../ProquestDataset/Test'

	"""
	BASELINE: NAIVE BAYES MODEL --------------------------------------------
	"""

	# Create the classifier using 5-fold CV
	nb = cross_validation(train_dir, NB_func, nb_hparams, verbose=True)

	# Evaluate the model
	evaluate_model(nb, test_dir, 'Proquest-POS Tags')


# """
# MODEL: LOGISTIC REGRESSION ---------------------------------------------
# """
if False:
	# Function to build the LogReg model
	LR_func = lambda args: LogReg(bin_size=args[0], features=['trees'])

	# Hyperparameters for the log reg classifier
	LR_hparams = [ np.array([5, 10, 20, 35, 50]) ]

	# Create the classifier using 5-fold CV
	LR = cross_validation(train_dir, LR_func, LR_hparams, verbose=False)

	# Evaluate the model
	evaluate_model(LR, test_dir, 'Proquest')

