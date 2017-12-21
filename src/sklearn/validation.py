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


def random_resample_validation(train, regr_func, hyperparams, loss_func, num_folds=1, verbose=False):
    # First, build all possible ordered pairs of hyperparameters
    hparam_pairs = list(itertools.product(*hyperparams))
    # Build an array to store the training err for each hyperparameter
    # pair for all folds
    train_losses = np.zeros((len(hparam_pairs), num_folds))
    test_losses = np.zeros((len(hparam_pairs), num_folds))
    # For each iteration, take a random sample of the data and train on it.
    # Then test each classifier on held-out fold.
    # Record the err for each classifier for each held-out fold
    print '-------------------------------------------------------------'
    print 'Beginning Cross-Validation with ' + str(num_folds) + ' folds.'
    print '-------------------------------------------------------------'
    for i in range(0, num_folds):
        # Setup the folds 
        tr, te = sample(train) # Temporary array of the train data
        train_x = tr[:,:-1]
        train_y = tr[:,-1]
        test_x = te[:,:-1]
        test_y = te[:,-1]
        # Train and test the different models on the selected folds
        (train_losses[:, i], test_losses[:, i]) = test_hyperparameters(train_x, train_y, test_x, test_y, regr_func, hparam_pairs, loss_func)
        # Repeat for all folds
        print 'Fold ' + str(i+1) + ' completed.'
    print '-------------------------------------------------------------'
        
    # Send the errors to a help function to find the best hyperparameter pair
    (best_params, mean_loss, loss) = determine_best_hyperparameter(hparam_pairs, test_losses, num_folds)
    # If verbose is true, print out the errs on all of the hparam pairs
    if verbose:
        plot_hyperparam_loss(best_params, hyperparams, np.mean(train_losses, axis=1), mean_loss)
    # Give the user some feedback on the process
    print 'Cross-validation completed.'
    print 'Mean loss on the best hyperparameter setting: ' + str(loss/num_folds)
    print 'Now training the model on the entire test set.'
    return regr_func(best_params).fit(train[:,:-1],train[:,-1])
    
    
# Helper function for validation
# splits the data into train and test splits
def sample(M, size=0.75):
    # Determine the train size
    K = int(M.shape[0] * size)
    # get the indices
    inds = np.arange(M.shape[0])
    # shuffle
    np.random.shuffle(inds)
    # return train and test
    return M[inds[:K], :], M[inds[K:], :]
    
 

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
    return mean_squared_error(y1, y2.toarray())


"""
Batman!
"""

def evaluate_model(regr, test, title, bin_size=20):
    # split into X, y
    X = test[:,:-1]
    gold = test[:, -1].toarray()

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
                  
                  
                  
                  
                  
                  
                  
                  
                     
     