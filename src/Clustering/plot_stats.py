# Imports
from cluster_utils import load_data
import numpy as np
import matplotlib.pyplot as plt

bin_size = 20
start = 1600

# Load the probabilites for all words
probs, words = load_data('../../GutenbergDataset', rev=True)
num_bins = len(probs[0])

def plot_word(word):
    # Get the probabilities for the word
    ind = np.where(words == word)[0][0]
    # Make the plot
    plt.plot(np.array([start + i * bin_size for i in range(num_bins)]), probs[ind], 'g-')
    plt.grid()
    plt.xlabel('Bin Number, B')
    plt.ylabel('Normalized Log Prob')
    plt.title('Prob of %s by Bin' % word.upper())
    plt.show()


plot_word('and')
plot_word('machine')
plot_word('or')




