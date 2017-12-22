# run_me.py module

# Library dependencies
import numpy as np
import scipy
from os import listdir

# Local imports
from validation import random_resample_validation
from validation import calculate_loss
from validation import RMSE
from validation import evaluate_model

# Classifier imports
from my_classifier import BinClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Load data function
def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# Set a seed for reproducibility
np.random.seed(271828)

# Dataset location
root = '../../ProquestDataset'
features = 'trees'

# Load the train and test sets
train = load_sparse_csr('%s/Train/Numpy/%s.npz' % (root, features))
test  = load_sparse_csr('%s/Test/Numpy/%s.npz' % (root, features))

# Regressor function
func = lambda args: BinClassifier(bin_size=int(args[0]), model=LogisticRegression)

# Hyperparameters
hparams = np.array([[5, 10, 20, 35, 50]])

# Cross-validation for hyperparameter optimization
# Note: Turned off verbose so plots do not interrupt the running
regr = random_resample_validation(train, func, hparams, RMSE)

# Evaluate the model
evaluate_model(regr, test, 'LogReg, Proquest Unigrams')