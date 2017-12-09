import os
import numpy as np
from tokenize import tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline


class LogReg:

<<<<<<< HEAD
    def __init__(self, bin_size=20, feat):
=======
    def __init__(self, bin_size=20, features = ['unigrams']):
>>>>>>> 766ce9e83f10939c3f2a882625fd1e4978e7860e
        self.bin_size = bin_size
        self.clf = Pipeline([('logreg', LogisticRegression(solver='saga', n_jobs=-1) ), ])
        self.features = features

    def fit(self, train_dir, filenames = None, verbose=False):
        # Extract the dataset
        if filenames is None:
            filenames = os.listdir('%s/Unigrams' % train_dir)
        feature_dicts, labels = self.construct_dataset(train_dir, filenames, train=True)
        sparse_feature_matrix = DictVectorizer(sparse=True).fit_transform(feature_dicts)
        # Fit the model
        self.clf.fit(sparse_feature_matrix, labels)
        # Return self
        return self

    def predict(self, test_dir, filenames, verbose=False):
        # Extract the data
        test, _ = self.construct_dataset(test_dir, filenames, train=False)
        # predict the bins
        predicted = self.clf.predict(test)
        # Convert to years
        return [self.determine_year(x) for x in predicted]

    """

    Helper functions

    """

    def determine_bin(self, year):
        return (year - self.min_year) / self.bin_size

    def determine_year(self, bin):
        return self.min_year + bin * self.bin_size


    def construct_dataset(self, location, filenames, train=True):
        """
        The train flags determines if this is the test set
        """
        books = []
        years = []
        # Loop through the files, recording the bodies and labels

        for filename in filenames:
            # Get the year
            year = int(filename[:4]) # first 4 characters
            # Append these to the corresponding lists
            books.append(tokenize(location, self.features, filename))
            years.append(year)

        # If this is the train set, return the bins instead,
        #   and set the min year
        if train:
            self.min_year = min(years)
            return books, [self.determine_bin(x) for x in years]
        # Else, just return the books and labels
        else:
            return books, years
