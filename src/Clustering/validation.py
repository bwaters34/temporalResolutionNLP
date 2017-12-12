# -*- coding: utf-8 -*-
"""
@author: Peter

File containing function for data validation

"""

# Imports
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def cross_validation(train, regr_func, hyperparams, loss_func, num_folds=10, verbose=True):
    # First, build all possible ordered pairs of hyperparameters
    hparam_pairs = list(itertools.product(*hyperparams))
    # Shuffle the training data
    np.random.shuffle(train)
    # Build an array to store the training err for each hyperparameter
    # pair for all folds
    train_losses = np.zeros((len(hparam_pairs), num_folds))
    test_losses = np.zeros((len(hparam_pairs), num_folds))
    # Construct the folds
    folds = fold_builder(train, num_folds)
    # For each fold, hold it out and train the model for classifier
    #   using each pair of hyperparameters.
    # Then test each classifier on held-out fold.
    # Record the err for each classifier for each held-out fold
    print '-------------------------------------------------------------'
    print 'Beginning Cross-Validation with ' + str(num_folds) + ' folds.'
    print '-------------------------------------------------------------'
    for i in range(0, num_folds):
        # Setup the folds 
        temp = np.delete(train, folds[i], axis=0) # Temporary array of the train data
        train_x = temp[:,:-1]
        train_y = temp[:,-1]
        test_x = train[folds[i][0]:folds[i][1],:-1]
        test_y = train[folds[i][0]:folds[i][1],-1]
        # Train and test the different models on the selected folds
        (train_losses[:, i], test_losses[:, i]) = test_hyperparameters(train_x, train_y, test_x, test_y, regr_func, hparam_pairs, loss_func)
        # Repeat for all folds
        print 'Fold ' + str(i+1) + ' completed.'
    print '-------------------------------------------------------------'
        
    # Send the errors to a help function to find the best hyperparameter pair
    (best_params, mean_loss, loss) = determine_best_hyperparameter(hparam_pairs, test_losses, num_folds)
    # If verbose is true, print out the errs on all of the hparam pairs
    if (verbose==1):
        plot_hyperparam_loss(best_params, hyperparams, np.mean(train_losses, axis=1), mean_loss)
    # Give the user some feedback on the process
    print 'Cross-validation completed.'
    print 'Mean loss on the best hyperparameter setting: ' + str(loss/num_folds)
    print 'Now training the model on the entire test set.'
    return regr_func(best_params).fit(train[:,:-1],train[:,-1])
    

# Helper function for cross-validation
# Given the training data and the number of folds 
#   generates the indexes of each fold
# I.e. if the training set had 10 points
#   and we wanted 5 folds, the indexes would
#   be 0-2,2-4,4-6,6-8,8-10
# Note: The last index of the i'th fold is the first
#   fold for the (i+1)'th fold due to how numpy indexing works
def fold_builder(train, num_folds):
    # First, build all possible ordered pairs of hyperparameters
    avg = len(train) / (1.0*num_folds)
    # Create an array for the indexes
    folds = np.zeros((num_folds, 2))
    # The first fold begins are index 0,
    #   so start here
    prev = 0.0
    # For each fold, set the first and last
    #   index of the fold
    for i in range(0, num_folds):
        # First index of the i'th fold is the 
        # last index of the (i-1)'th fold
        # due to numpy array indexing
        folds[i][0] = int(prev)
        # Move forward the average number of step
        folds[i][1] = int(prev + avg)
        # Update start index of the fold
        prev += avg
        # Sometimes I needed to reduce the last index by 1?
        if (i == num_folds - 1) and (folds[i][1] == len(train)):
            folds[i][1] = folds[i][1] - 1
    # Return the folds' indexes
    return folds.astype(int)
 

# Helper function for validation
# For each hyperparameter setting, it trains the model on the 
# training set and then tests the fitted model on the test set.
# Returns an array of each settings error
def test_hyperparameters(train_x, train_y, test_x, test_y, regr_func, hparams, loss_func):
    train_losses = np.zeros(len(hparams))
    test_losses = np.zeros(len(hparams))
    
    for i in range(0, len(hparams)):
        # Build the regressor for the i'th hparam setting
        regr = regr_func(hparams[i])
        
        regr.fit(train_x,train_y)
        # Test the regressor on the test set
        train_losses[i] = calculate_loss(loss_func, regr, train_x, train_y)
        test_losses[i] = calculate_loss(loss_func, regr, test_x, test_y)
    
    return (train_losses, test_losses)

    
# Given all hyper parameter settings and an array of their errors
# on various sets of tests, returns the hyper parameter with the lowest mean err
def determine_best_hyperparameter(params, losses, num_iterations):
    # Calculate the mean % error
    mean_loss = 100*np.sum(losses, axis=1)/num_iterations
    # Find the index which has the lowest mean error
    best_index = np.argmin(mean_loss)
    # Return the respected hyperparameter settings, the mean errs
    # and the mean error of the best parameter
    # These are all returned for analysis purposes
    best_param = params[ best_index ]

    return (best_param, mean_loss, mean_loss[best_index])
   
    
def plot_hyperparam_loss(best_params, params, train_loss, test_loss):
    # Count the number of settings we tested
    N = len(params[0])
    
    # remaining dimensions, flattened
    rem = np.product([len(temp) for temp in params[1:]])

    y1 = np.zeros(N)
    y2 = np.zeros(N)
    for i in range(N):
        for j in range(rem):
            y1[i] = train_loss[i*rem + j]
            y2[i] = test_loss[i*rem + j]

    # plot_num = int(np.random.rand()*100)
    plot_num = 1
    
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


"""
Inputs:
    regr - classifer object
    test_x - data points which the classifier will predict the classes of
    test_y - the actual classes of the test data
The function computes the validation error given a classifer and a data set
"""    
def calculate_loss(loss_func, regr, test_x, test_y):
     # Have the classifier predict the classes of the data
     predictions = regr.predict(test_x)
     # count the number of predictions which are incorrect
     return loss_func(predictions, test_y)
    

# LOSS FUNCTIONS
    
def RMSE(y1, y2):
    return mean_squared_error(y1, y2)


"""
Batman!
"""

def evaluate_model(regr, test, title, bin_size=20):
    # split into X, y
    X = test[:,:-1]
    gold = test[:, -1]

    # Predict the test data
    pred = regr.predict(X)

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
    plt.bar(x, y, width =bin_size-1)
    # plt.plot(x, y, 'or-', linewidth=3)
    plt.grid(True)
    plt.title('%s: MAE by Bin' % title)
    plt.xlabel('Century')
    plt.ylabel('MAE in Years')
    plt.show()

    # Print the overall MAE
    print 'Overall MAE:', calc_MAE(gold, pred)
    
    
def calc_MAE(gold, pred):
	return np.mean( np.abs(np.array(gold) - np.array(pred)) ) 
                  
                  
                  
                  
                  
                  
                  
                  
                     
     