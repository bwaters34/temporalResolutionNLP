"""
Header

"""

# Imports
import numpy as np
from os import listdir
from naive_bayes import NaiveBayes
from log_reg import MultiClass
from train_test_helper import cross_validation, evaluate_model


### Dataset 1 : Gutenberg

# Set the train and test dir
train_dir = '../Proquest Dataset/Yearly/Train'
test_dir  = '../Proquest Dataset/Yearly/Test'

"""
BASELINE: NAIVE BAYES MODEL --------------------------------------------
"""

# Function to build a NB model
NB_func = lambda args: NaiveBayes(bin_size=args[0], alpha=args[1])

# These are the hyperparameters we are testing over
nb_hparams = [ np.array([1, 10, 20, 30, 50]),   # Bin Size
			   np.array([1, 5, 10, 25, 50])]	# Alpha 

# Create the classifier using 5-fold CV
nb = cross_validation(train_dir, NB_func, nb_hparams, verbose=True)

# Evaluate the model
evaluate_model(nb, test_dir)
