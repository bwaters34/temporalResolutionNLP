from os import makedirs, listdir
from os.path import exists
from collections import defaultdict
import codecs
import cPickle as cpk

def determine_bin(year, min_year, bin_size):
    return int((year-min_year)/bin_size)

# Location for the dataset
ds = 'GutenbergDataset'
bin_size = 20.
min_year = 1600
bin_word_counts = defaultdict(lambda: defaultdict(float))
bin_total_counts = defaultdict(float)

# Some directory names
root = '../../%s' % ds
clst_dir = '%s/Clustering' % root

src_root = '../../%s/%s' % (ds, 'Yearly')
src_train = '%s/Train' % src_root
src_test = '%s/Test' % src_root

# Make the directories, if necessary
if not exists(clst_dir):
	makedirs(clst_dir)

# Go through and build the cluster data
# First the train data
for f in listdir('%s/Yearly/Train' % ds):
    # Get the text
    with codecs.open('%s/Yearly/Train/%s' % (ds, f), 'r', 'utf-8') as infile:
        data = infile.read()
    # Tokenize
    tokens = data.split()
    # Add the the default dict
    
    

