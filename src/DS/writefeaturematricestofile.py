import os
from sklearn.feature_extraction import DictVectorizer
from my_tokenize import my_tokenize
import numpy as np
import scipy

def construct_dataset(location, filenames, features):
    """
    The train flags determines if this is the test set
    """
    books = []
    years = []
    # Loop through the files, recording the bodies and labels

    for filename in filenames:
        # Get the year
        year = int(filename[:4])  # first 4 characters
        # Append these to the corresponding lists
        books.append(my_tokenize(location, features, filename))
        years.append(year)
        
    return books, np.array(years)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
    
def save_as_csv(filename, X, y):
    with open(filename, 'w') as outfile:
        for i in range(X.shape[0]):
            buffer = ''
            for j in range(X.shape[1]):
                buffer += '%s,' % X[i,j]
            # Add the label
            buffer += str(y[i]) + '\n'
            outfile.write(buffer)

def write_feature_matrices_to_file(features, train_dir, test_dir):
    # Get the train features
    feature_dicts_train, tr_y = construct_dataset(train_dir, os.listdir(train_dir + '/Unigrams/'), features)
    
    # Get the test features
    feature_dicts_test, te_y = construct_dataset(test_dir, os.listdir(test_dir+ '/Unigrams/'), features)
    
    print '(X, y) pairs loaded'
    
    # Dict to vector
    dv = DictVectorizer()
    sparse_feature_matrix = dv.fit_transform(feature_dicts_train + feature_dicts_test)
    
    # Break into train and test splits again
    tr_X = sparse_feature_matrix[:len(tr_y)].toarray()
    te_X = sparse_feature_matrix[len(tr_y):].toarray()
    
    # Add in the labels now

    # reshape
    tr_y = tr_y.reshape( (len(tr_y), 1) )
    te_y = te_y.reshape( (len(te_y), 1) )

    # Append the labels
    tr = np.hstack((tr_X, tr_y))
    te = np.hstack((te_X, te_y))
    
    # Back to sparse matrix
    tr = scipy.sparse.csr_matrix(tr)
    te = scipy.sparse.csr_matrix(te)

    print 'Vectorized'
    
    # Name of train file
    feature_string = '_'.join(sorted(features))
    train_save_directory = train_dir + "/Numpy/"
    train_save_file_name = train_save_directory + feature_string
    # Write to file
    save_sparse_csr(train_save_file_name, tr)
    
    print 'Train written'
    
    # Name of test
    test_save_directory = test_dir + "/Numpy/"
    test_save_file_name = test_save_directory + feature_string
    # Write to file
    save_sparse_csr(test_save_file_name, te)

    print 'Test written'

if __name__ == '__main__':
    train_dir = '../../GutenbergDataset/Train'
    test_dir = '../../GutenbergDataset/Test'
    features = ['unigrams']

    write_feature_matrices_to_file(features, train_dir, test_dir)
    