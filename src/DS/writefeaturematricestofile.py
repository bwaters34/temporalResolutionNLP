import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from my_tokenize import my_tokenize
import numpy as np
import cPickle
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
        
    return books, years

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def write_feature_matrices_to_file(features, train_dir, test_dir):
    feature_dicts_train, labels_train = construct_dataset(train_dir, os.listdir(train_dir + '/Unigrams/'), features)
    feature_dicts_test, labels_test = construct_dataset(test_dir, os.listdir(test_dir+ '/Unigrams/'), features)
    dv = DictVectorizer(sparse=True)
    train_length = len(feature_dicts_train)
    print(train_length)
    test_length = len(feature_dicts_test)
    print(test_length)
    sparse_feature_matrix = dv.fit_transform(feature_dicts_train + feature_dicts_test)
    print(sparse_feature_matrix.shape)
    assert sparse_feature_matrix.shape[0] == train_length + test_length
    train_matrix = sparse_feature_matrix[:train_length]
    test_matrix = sparse_feature_matrix[train_length:]
    assert test_length == test_matrix.shape[0]
    feature_string = '_'.join(sorted(features))

    train_save_directory = train_dir + "/Numpy/"
    train_save_file_name = train_save_directory + feature_string
    # with open(train_save_file_name, 'wb') as f:
    #     cPickle.dump(train_matrix, f)
    save_sparse_csr(train_save_file_name, train_matrix)

    test_save_directory = test_dir + "/Numpy/"
    test_save_file_name = test_save_directory + feature_string
    # with open(test_save_file_name, 'wb') as f:
    #     cPickle.dump(test_matrix, f)
    save_sparse_csr(test_save_file_name, test_matrix)
    # Unigrams.txt


if __name__ == '__main__':
    train_dir = '../../GutenbergDataset/Train'
    test_dir = '../../GutenbergDataset/Test'
    features = ['bigrams']

    write_feature_matrices_to_file(features, train_dir, test_dir)
