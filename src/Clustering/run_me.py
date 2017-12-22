# run_me.py module

# Library dependencies
import numpy as np

# Local imports
from validation import cross_validation
from validation import calculate_loss
from validation import RMSE
from validation import evaluate_model

# Classifier imports
from my_classifier import BinClassifier
from sklearn.linear_model import LogisticRegression
from naive_bayes import NaiveBayes

# Set a seed for reproducibility
np.random.seed(271828)

# Dataset location
root = '../../GutenbergDataset'

# Load the train and test sets
train = np.genfromtxt('%s/Clusters/train-rev.csv' % root, delimiter=',')
test  = np.genfromtxt('%s/Clusters/test-rev.csv' % root, delimiter=',')

if True:
    # Regressor function
    func = lambda args: BinClassifier(bin_size=int(args[0]), model=LogisticRegression)

    # Hyperparameters
    hparams = np.array([[5, 10, 20, 35, 50]])

    # Cross-validation for hyperparameter optimization
    # Note: Turned off verbose so plots do not interrupt the running
    clf = cross_validation(train, func, hparams, RMSE, verbose=False)

    # Evaluate the model
    evaluate_model(clf, test, 'Gutenberg')

if False:
    # Create the classifier using 5-fold CV
    nb = NaiveBayes(bin_size=25, alpha=1).fit(train[:,:-1], train[:,-1])
    
    # Evaluate the model
    evaluate_model(nb, test, 'Gutenberg')