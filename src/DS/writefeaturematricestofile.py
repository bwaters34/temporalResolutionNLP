import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.tokenize import tokenize
import numpy
import cPickle



def determine_bin(self, year):
    return (year - self.min_year) / self.bin_size


def determine_year(self, bin):
    return self.min_year + bin * self.bin_size


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
        books.append(tokenize(location, features, filename))
        years.append(year)

    # If this is the train set, return the bins instead,
    #   and set the min year
    # if train:
    #     # self.min_year = min(years)
    #     return books, [determine_bin(x) for x in years]
    # # Else, just return the books and labels
    # else:
    return books, years



# if filenames is None:
#     filenames = os.listdir('%s/Unigrams' % train_dir)




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
    with open(train_save_file_name, 'wb') as f:
        cPickle.dump(train_matrix, f)

    test_save_directory = test_dir + "/Numpy/"
    test_save_file_name = test_save_directory + feature_string
    with open(test_save_file_name, 'wb') as f:
        cPickle.dump(test_matrix, f)

# Unigrams.txt


if __name__ == '__main__':
    train_dir = '../GutenbergDataset/Train'
    test_dir = '../GutenbergDataset/Test'
    features = ['unigrams', 'bigrams']

    write_feature_matrices_to_file(features, train_dir, test_dir)
