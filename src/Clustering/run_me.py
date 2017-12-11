# run_me.py module

# Library dependencies
import numpy as np

# Local imports
from validation import cross_validation
from validation import calculate_loss
from validation import RMSE
from validation import evaluate_model

# Regressor imports
from sklearn.ensemble import RandomForestRegressor

# Set a seed for reproducibility
np.random.seed(271828)

# Dataset location
root = '../../GutenbergDataset'

# Load the train and test sets
train = np.genfromtxt('%s/Clusters/train.csv' % root, delimiter=',')
test  = np.genfromtxt('%s/Clusters/test.csv' % root, delimiter=',')

# Regressor function
func = lambda args: RandomForestRegressor(n_estimators=int(args[0]), criterion=args[1])

# Hyperparameters
hparams = np.array([np.array([5, 10, 20, 30, 40]), # num trees
                        np.array(['mse', 'mae']) # 
                       ]) 

# Cross-validation for hyperparameter optimization
# Note: Turned off verbose so plots do not interrupt the running
regr = cross_validation(train, func, hparams, RMSE)

# Evaluate the model
evaluate_model(regr, test, 'Gutenberg')