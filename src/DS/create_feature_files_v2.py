from os import makedirs, listdir
from os.path import exists
from collections import defaultdict
import codecs
import cPickle as cpk
from nltk import pos_tag, word_tokenize, sent_tokenize



import numpy as np #Import numpy for easier matrix manipulations later

# Translations from Stanford POSTagger to simplified syntactic model
translations = {"M":"M","P":"P","D":"D","A":"A","N":"N","ADV":"ADV","V":"V", "CC":"C","CD":"A","DT": "D","EX":"D","FW":"END","IN":"P","JJ":"A","JJR":"A","JJS":"A","LS":"A","MD":"M","NN":"N","NNS":"N","NNP":"N","NNPS":"N","PDT":"D","POS":"DS","PRP":"N","PRP$":"D","RB":"ADV","RBR":"ADV","RBS":"ADV","RP":"D","SYM":"N","TO":"V","UH":"END","VB":"V","VBD":"V","VBG":"V","VBN":"V","VBP":"V","VBZ":"V","WDT":"D","WP":"N","WP$":"D","WRB":"ADV",",":"C"}

# For phrase decomposition later
phrases = {"A":["P","D","N"], "D":["A", "N"], "N":["P","V"], "P":["A","D","N"], "V":["D","A","P","N"], "M":[], "C":[],"DP":["A", "N"], "ADV":["A","V","ADV"]}

###############################################################################
# Sentence Bifurcation Parsing Engine
#
# A library of functions built around parsing arrays of parts of speech into
# syntactic binary trees. Output of main function is a dictionary with a count
# of the number and type of each syntactic phrase in the given sentence. 
#
# Pipeline:
# 1. Parts of speech are translated using the above dictionary to reduce the 
#    number of parts of speech
# 2. Sentences are stripped of duplicate words, and ADV-V/A-N phrases are 
#    compressed into V and N, respectively
# 3. The full sentence is then split at any conjunctive or modals, adding an
#    assumed subject if needed in the case of the conjunctive
# 4. Sentences are then split into subjects and predicates and parsed 
#    individually
# 5. All phrases are recorded in the final dictionary and returned
#
# No linguists were harmed in the making of this script
###############################################################################

##

###############################################################################
# Alternative flatten function
#
# Flattening list of list of strings into a list of strings as opposed to one
# string, which is the default flatten function
###############################################################################

def flat(s):
    t = []              # Define return val
    for h in s:         # Loop through 0th order list
        if h is not None:
            for i in h:     # Loop through 1st order list
                t.append(i) # Append all strings
    return t            # Return list of strings

##

###############################################################################
# Conjunctive/modal splitting function
#
# This splits the list of parts of speech at all conjunctions and modals and
# adds an additional DP when needed
###############################################################################

def splt(arr):
    sents = []                  # Define return value
    temp = []                   # Sentence building array
    for i in range(len(arr)):   # Loop through all parts of speech
        if arr[i] == "M":       # Check for modal phrases
            temp.append("M")    # Append modal to current sentence
            sents.append(temp)  # Add sentence to the list
            temp = []           # Reset sentence builder
        elif arr[i] == "C":     # Check for conjunctive phrases 
            temp.append("C")    # Append conjunctive to current sentence
            sents.append(temp)  # Append current sentence
            temp = ["DP"]       # Reset sentence builder
        else:                   # For all other cases
            temp.append(arr[i]) # Add the word to the sentence
    sents.append(temp)          # Append the last sentence
    return sents                # Return list of sentences

##

###############################################################################
# Phrase parser
#
# Uses sentence bifurcation to split sentences into binary phrases
###############################################################################

def phrase(wrds):
    if wrds == []:
        return
    checks = [phrases[wrds[0]]]                 # Stack to check legal phrases
    retval = wrds[0]                            # Define return value
    i = 1                                       # Define end of list flag
    while checks != [] and i < len(wrds):       # Loop until EoL or No phrases
        curr = checks.pop()                     # Checking for phrase legality
        if wrds[i] in curr:                     # If the phrase is legal
            retval += wrds[i]                   # Add current word to phrase
            checks.insert(0,phrases[wrds[i]])   # Insert check 1
            checks.insert(0,phrases[wrds[i]])   # For certain multiverb phrase
        else:                                   # If phrase is illegal
            retval += wrds[i]                   # Terminate current, start new
        i = i + 1                               # Iterate index counter
    return retval                               # Return parsed phrase

###############################################################################
# Main parse function 
#
# Phrase reconstruction algorithm:
# For the compressions and cuts performed in the earlier stages of the script
# to work, they need to be reinserted into the phrases they were removed from,
# but keeping the phrase ordering the same. Furthermore, the way that the moves
# are stored is in waves as the program loops to find all available cuts it can
# make. To address this, the movelist is used as a stack, and as every value is
# expended, the index at which the wordlist was edited is added to a sorted 
# list of all past indecies. This array keeps track of the offset produced from
# returning values into the sentence that had been removed, and making sure 
# they are returned to the correct location by seeing how many of the values 
# reinserted into the sentence are before the current word being inserted. The
# decompression is always adding a value to the left of the one being expanded.
# This method allows for complex reconstruction such as VP = ADV ADV V PP that 
# doesn't strictly follow x-bar but allows for feature extraction, which is the
# ultimate objective.
###############################################################################

def parse(s):                         
    moves = []                              # Predefine list of compressions
    wrds = []                               # Predefine sentence words list
    for i in range(len(s)):                 # Loop through all of the words
        if s[i] != "POS":                   # Avoiding the possessive
            wrds.append(translations.get(s[i],"END")) # Returning the rest of the list
    if "END" in wrds:                         
        return -1
    last = 0                                # Last list length
    curr = len(wrds)                        # Define current list length
    while curr != last:                     # While there is no change
        last = curr                         # Save last list size
        try:                                # Try to compress...
            for i in range(len(wrds)-1):    # Loop through every bigram
                if (wrds[i] == wrds[i+1]):
                    moves.append((i,"C"))
                    del wrds[i]
                    i -= 1
            for i in range(len(wrds)-1):
                if wrds[i] == "A" and wrds[i+1] == "N": # Check for A+N
                    del wrds[i]             # Delete the adjective
                    moves.append((i,"A"))   # Save move
                    i -= 1                  # Reiterate
                if wrds[i] == "ADV" and (wrds[i+1] == "V" or wrds[i+1] == "A"): # Check for ADV+V
                    del wrds[i]             # Delete ADV
                    moves.append((i,"D"))   # Save move
                    i -= 1                  # Reiterate
        except IndexError:                  # If the index is broken, that 
            curr = len(wrds)                # Means at least one word removed

    sents = splt(wrds)                      # Split at conjunctives and modals

    data = []                               # Predefine data array

    for sent in sents:                      # Loop through all sentences
        if "V" in sent:
            DP = sent[:sent.index("V")]         # Split into demonstrative phrase
            VP = sent[sent.index("V"):]         # ... and a verb phrase 
            data.append(phrase(DP))             # Parse subject
            data.append(phrase(VP))             # Parse predicate
        else:
            data.append(phrase(sent))

    dictionary = {}                         # Predefine final dictionary
    cp = data                               # Make sentence array copy
    data = flat(data)                       # Flatten the data
    srt = []                                # Predefine sorted array

    for m in reversed(moves):               # Loop through moves backwards
        if m[1] == "C":                     # Check for (C)opy scenario
            srt.append(m[0])                # Add the move's index
            srt = sorted(srt)               # Sort the aggregate index array
            curr = m[0] - srt.index(m[0])   # Redefine to new index
            try:                            # Insert into relevant phrase
                data[curr].insert(0,data[curr][0])
            except AttributeError or IndexError:
                data[curr] = [data[curr], data[curr]]
        elif m[1] == "A":                   # Check for (A)djective scenario
            srt.append(m[0])                # Add index
            srt = sorted(srt)               # Sort
            curr = m[0] - srt.index(m[0])   # Redefine index
            try:                            # Add to phrase
                data[curr].insert(0,"A")
            except AttributeError or IndexError:
                data[curr] = ["A", data[curr]]
        elif m[1] == "D":                   # Check for a(D)verb scenario
            srt.append(m[0])                # Add index
            srt = sorted(srt)               # Sort
            curr = m[0] - srt.index(m[0])   # Redefine Index
            try:                            # Add to phrase
                data[curr].insert(0,"ADV")
            except AttributeError or IndexError:
                data[curr] = ["ADV", data[curr]]

    # Everything below is saving the data into the dictionary

    # for i in range(len(cp)/2):           
    #     try:
    #         dictionary["S"][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] += 1
    #         if i > 0:
    #             p = cp[2*i - 1][-1]
    #             dictionary[p][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] += 1
    #     except KeyError:
    #         try:
    #             dictionary["S"] == False
    #         except KeyError:
    #             dictionary["S"] = {}
    #         dictionary["S"][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] = 1
    #         if i > 0:
    #             p = cp[2*i - 1][-1]
    #             try:
    #                 dictionary[p] == False
    #             except KeyError:
    #                 dictionary[p] = {}
    #             dictionary[p][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] = 1

    cp = [x for x in cp if x != None]

    for i in range(len(data) - 2):
        if data[i] != "M" and data[i] != "C":
            try:
                f = "".join(flat([data[i],data[i+1], "P"]))
                dictionary[data[i][-1] + "P"][f] += 1
            except KeyError:
                try:
                    dictionary[data[i][-1] + "P"] == False
                except KeyError:
                    dictionary[data[i][-1] + "P"] = {}
                f = "".join(flat([data[i],data[i+1],"P"]))
                dictionary[data[i][-1] + "P"][f] = 1
        else:
            try:
                f = "".join(flat([data[i],data[i+1],data[i+2], "P"]))
                dictionary[data[i][-1] + "P"][f] += 1
            except KeyError:
                try:
                    dictionary[data[i][-1] + "P"] == False
                except KeyError:
                    dictionary[data[i][-1] + "P"] = {}
                f = "".join(flat([data[i],data[i+1],data[i+2], "P"]))
                dictionary[data[i][-1] + "P"][f] = 1

    #print dictionary
    return dictionary
    
#print parse(["N","V","V","M","D","N","ADV","ADV","V","P","D","A","N"])


"""
Copied them here just so I didnt need to deal with import statements
"""
# Some characters whose lines we will throw out
delimitable = ['\t', '|', '[', ']', '+']
bad_tags = ['.','``', '\'\'', ':']

def simple_tokenizer(doc): 
    bow = defaultdict(float)
    tokens = [t.lower() for t in doc.split()]
    for t in tokens:
        bow[t] += 1
    return bow

def bigram_tokenizer(doc):
    bow = defaultdict(float)
    tokens = [t.lower() for t in doc.split()]
    for i in range(1,len(tokens)):
        bow[(tokens[i-1], tokens[i])] += 1
    return bow

def pos_tag_tokenizer(doc):
    tags_dict = defaultdict(float)

    # Format the document
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\r', ' ')
    doc = doc.replace('  ', ' ')

    # Get the sentences
    sents = sent_tokenize(doc)

    # For all good sentences, pos-tag them
    for f in sents:
        if not any([(x in f) for x in delimitable]):
            tokens = word_tokenize(f)
            tags = pos_tag(tokens)
            # (word, pos) items
            for tag in tags:
                tags_dict[tag] +=1
            # transition counts
            for i in range(1, len(tags)):
                prev = tags[i-1][1]
                curr = tags[i][1]
                tags_dict[(prev,curr)] += 1

    # done!
    return tags_dict

def tree_tokenizer(doc):
    tree_dict = defaultdict(float)

    # Format the document
    doc = doc.replace('\n', ' ')
    doc = doc.replace('\r', ' ')
    doc = doc.replace('  ', ' ')

    # Get the sentences
    sents = sent_tokenize(doc)

    # For all good sentences, pos-tag them
    broken_trees = 0
    good_trees = 0
    for f in sents:
        if not any([(x in f) for x in delimitable]):
            tokens = word_tokenize(f)
            tags = [x[1] for x in pos_tag(tokens) if x[1] not in bad_tags]
            # build the tree
            try:
                tree = parse(tags)
			# Check for a particular indexing error
            except IndexError:
                broken_trees += 1
                continue
            # check if -1
            if (tree == -1):
                broken_trees += 1
                continue
            good_trees += 1
            for t1 in tree:
                for t2 in tree[t1]:
                    tree_dict[(t1, t2)] += 1

    # done!
    print ("Bad Trees: " + str(broken_trees) + "\tGood Trees: " + str(good_trees))
    return tree_dict
            

# Location for the dataset
ds = 'GutenbergDataset'
loc = 'SentTrees'
func = tree_tokenizer

# Some directory names
root = '../../%s' % ds
train = '%s/Train/%s' % (root, loc)
test = '%s/Test/%s' % (root, loc)

src_root = '../../%s/%s' % (ds, 'Yearly')
src_train = '%s/Train' % src_root
src_test = '%s/Test' % src_root


# Make the directories, if necessary
if not exists(root):
    makedirs(root)
if not exists(train):
    makedirs(train)
if not exists(test):
    makedirs(test)

# Now actually build the dataset
# First the trainset
for filename in listdir(src_train):
    with codecs.open('%s/%s' % (src_train, filename), 'r', encoding='utf-8') as infile:
        # Read the document
        doc = infile.read()
        # Create the Bigrams
        feats = func(doc)
        # Write it to a file
        with open('%s/%s' % (train, filename), 'w') as outfile:
            outfile.write(cpk.dumps(feats))

# Now the test set
for filename in listdir(src_test):
    with codecs.open('%s/%s' % (src_test, filename), 'r', encoding='utf-8') as infile:
        # Read the document
        doc = infile.read().replace('\n', '')
        # Create the Bigrams
        feats = func(doc)
        # Write it to a file
        with open('%s/%s' % (test, filename), 'w') as outfile:
            outfile.write(cpk.dumps(feats))


