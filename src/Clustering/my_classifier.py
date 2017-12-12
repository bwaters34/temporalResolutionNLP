from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BinDecisionTreeClf:
    def __init__(self, bin_size, depth, crit):
        self.clf = DecisionTreeClassifier(criterion=crit, max_depth=depth)
        self.bin_size = float(bin_size)
        
    def determine_bin(self, year):
        return int((year-self.min_year)/self.bin_size)
    
    def determine_year(self, bbin):
        return (self.min_year + (bbin+0.5)*self.bin_size)
    
    def fit(self, X, y):
        self.min_year = min(y)
        self.max_year = max(y)
        
        # Convert to bins
        y = np.array([self.determine_bin(t) for t in y])
        
        # Train the model
        self.clf.fit(X, y)
        
        return self
    
    def predict(self, X):
        # Predict the data
        pred = self.clf.predict(X)
        # Convert to years
        return np.array([self.determine_year(b) for b in pred])