"""


@author: Peter
"""

# 
import kaggle, time
import numpy as np

# Local imports
from feature_selection import random_forest_feature_selection

from validation import cross_validation
from validation import RMSE
from validation import calculate_loss

"""
This pipeline is for datasets in which you have both train and test data.

The pipeline applies Gini importance feature selection, and then trains the model
using 10-fold cross-validation (if not specificed otherwise). Then the model is tested
on the test set, and the RMSE is printed to stdout.
"""
def pipeline_with_test(train, test, regr_func, hparams, num_trees=100, num_folds=20, threshold=0.005):
    print '-------------------------------------------------------------'
    print 'Beginning Pipeline with Test Set'
    print '-------------------------------------------------------------'
    # Split the data into x and y components
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]
    
    # Run the ExtraTreeRegressor feature selection method
    print 'Running ExtraTreeRegressor Feature Extraction'
    print '\tNumber of Trees: ' + str(num_trees)
    print '\tImportance Threshold: ' + str(threshold)
    feats = random_forest_feature_selection(train_x,train_y,num_trees,threshold)
    print 'Feature space reduced from ' + str(len(train_x[0])) + ' features to ' + str(len(feats)) + ' features'
    
    # Update the train and test data
    train_x = train_x[:, feats]
    test_x = test_x[:, feats]

    # Repackage the training data (it is useful to have the training data
    #   together when implementing cross validation)
    train = np.hstack((train_x, train_y.reshape((len(train_y),1))))

    # Cross-validation for hyperparameter optimization
    # Note: Turned off verbose so plots do not interrupt the running
    regr = cross_validation(train, regr_func, hparams, RMSE, num_folds=num_folds, verbose=False)
    
    # Test the regressor on the text data
    pipe_rmse = calculate_loss(RMSE, regr, test_x, test_y)
    
    # Print the results
    print '-------------------------------------------------------------'
    print 'Regressor Training Complete'
    print 'Regressor Hyperparameters: ' + str(regr.get_params())
    print '-------------------------------------------------------------'
    print 'Regressor Evaluation'
    print 'Predicting the Test Data'
    print 'Pipeline RMSE: ' + str(pipe_rmse)
    
    return regr
  
    
"""
This pipeline is for datasets in which you have only the train data.

The pipeline applies Gini importance feature selection, and then trains the model
using 10-fold cross-validation (if not specificed otherwise). Then the model predicts
the test set and outputs a kaggle csv to disk.
"""    
def pipeline_without_test(train, test, regr_func, hparams, path, num_trees=100, num_folds=10, threshold=0.005):
    # Set the start time
    start = time.time()
    print '-------------------------------------------------------------'
    print 'Beginning Pipeline without Test Set'
    print '-------------------------------------------------------------'
    # Split the data into x and y components
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    
    # Run the ExtraTreeRegressor feature selection method
    print 'Running ExtraTreeRegressor Feature Extraction'
    print '\tNumber of Trees: ' + str(num_trees)
    print '\tImportance Threshold: ' + str(threshold)
    feats = random_forest_feature_selection(train_x,train_y,num_trees,threshold)
    print 'Feature space reduced from ' + str(len(train_x[0])) + ' features to ' + str(len(feats)) + ' features'
    # Record the time from the start to get to this point
    fs_done = time.time()
    print 'Approx. Feature Selection Time: ' + str(fs_done - start)

    # Update the train and test data
    train_x = train_x[:, feats]
    test_x = test_x[:, feats]

    # Repackage the training data (it is useful to have the training data
    #   together when implementing cross validation)
    train = np.hstack((train_x, train_y.reshape((len(train_y),1))))

    # Cross-validation for hyperparameter optimization
    # Note: Turned off verbose so plots do not interrupt the running
    regr = cross_validation(train, regr_func, hparams, RMSE, num_folds=num_folds,verbose=False)
    
    # Record the training time
    train_done = time.time()
    
    # Print the results
    print '-------------------------------------------------------------'
    print 'Regressor Training Complete'
    print 'Approx. Train Time: ' + str(train_done - fs_done)
    print 'Regressor Hyperparameters: ' + str(regr.get_params())
    print '-------------------------------------------------------------'
    
    # Predict the values of the test data
    predictions = regr.predict(test_x)
    
    #Save prediction file in Kaggle format for scoring on Kaggle
    kaggle.kaggleize(predictions, path)
    print 'Predicting the Test Data'
    print 'Wrote predictions for the test set to ' + path

"""
This pipeline is for more developed ensemble methods which do not use cross-validation
for hyperparameter optimization. Instead, this pipeline, optionally, applies gini 
importance feature selection and then trains the model on the training set using 
the regressors build in fit / optimize function. It then predicts the test set
and creates a kaggle csv file
""" 
def pipeline_my_ensembles(train, test, ensemble_func, path, fs= True, num_trees=100, threshold=0.005):
    # Set the start time
    start = time.time()
    print '-------------------------------------------------------------'
    print 'Beginning Pipeline without Test Set'
    print '\tfor an ensemble method.'
    print '-------------------------------------------------------------'
    # Split the data into x and y components
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]

    # If we have opted to use the feature selection algorithm, apply it
    if fs:
        # Run the ExtraTreeRegressor feature selection method
        print 'Running ExtraTreeRegressor Feature Extraction'
        print '\tNumber of Trees: ' + str(num_trees)
        print '\tImportance Threshold: ' + str(threshold)
        feats = random_forest_feature_selection(train_x,train_y,num_trees,threshold)
        print 'Feature space reduced from ' + str(len(train_x[0])) + ' features to ' + str(len(feats)) + ' features'
        fs_done = time.time()
        print 'Approx. Feature Selection Time: ' + str(fs_done - start)    

        # Update the train and test data
        train_x = train_x[:, feats]
        test_x = test_x[:, feats]
    # Necessary because variable fs_done used later
    else:
        fs_done = time.time()

    # Now fit the ensemble to the data
    my_ensemble = ensemble_func().fit(train_x, train_y)
    
    # Record the training time
    train_done = time.time()
    
    # Print the results
    print '-------------------------------------------------------------'
    print 'Regressor Training Complete'
    print 'Approx. Train Time: ' + str(train_done - fs_done)
    print 'Regressor Hyperparameters: ' + str(my_ensemble.get_params())
    print '-------------------------------------------------------------'
    
    # Predict the values of the test data
    predictions = my_ensemble.predict(test_x)
    
    #Save prediction file in Kaggle format for scoring on Kaggle
    kaggle.kaggleize(predictions, path)
    print 'Predicting the Test Data'
    print 'Wrote predictions for the test set to ' + path
    