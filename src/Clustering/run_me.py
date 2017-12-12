# run_me.py module

# Library dependencies
import numpy as np

# Local imports
from validation import cross_validation
from validation import calculate_loss
from validation import RMSE
from validation import evaluate_model

# Classifier
from my_classifier import BinDecisionTreeClf

# Set a seed for reproducibility
np.random.seed(271828)

# Dataset location
root = '../../GutenbergDataset'

# Load the train and test sets
train = np.genfromtxt('%s/Clusters/train.csv' % root, delimiter=',')
test  = np.genfromtxt('%s/Clusters/test.csv' % root, delimiter=',')

# Regressor function
func = lambda args: BinDecisionTreeClf(bin_size=35, depth=int(args[0]), crit=args[1])

# Hyperparameters
hparams = np.array([np.array([5,6,7]), # num trees
                        np.array(['gini', 'entropy']) # 
                       ]) 

# Cross-validation for hyperparameter optimization
# Note: Turned off verbose so plots do not interrupt the running
regr = cross_validation(train, func, hparams, RMSE)

# Evaluate the model
evaluate_model(regr, test, 'Gutenberg')