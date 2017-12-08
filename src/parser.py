import numpy as np #Import numpy for easier matrix manipulations later

# Translations from Stanford POSTagger to simplified syntactic model
translations = {"M":"M","P":"P","D":"D","A":"A","N":"N","ADV":"ADV","V":"V", "CC":"C","CD":"A","DT": "D","EX":"D","FW":"END","IN":"P","JJ":"A","JJR":"A","JJS":"A","LS":"A","MD":"M","NN":"N","NNS":"N","NNP":"N","NNPS":"N","PDT":"D","POS":"DS","PRP":"N","PRP$":"D","RB":"ADV","RBR":"ADV","RBS":"ADV","RP":"D","SYM":"N","TO":"V","UH":"END","VB":"V","VBD":"V","VBG":"V","VBN":"V","VBP":"V","VBZ":"V","WDT":"D","WP":"N","WP$":"D","WRB":"ADV",",":"C"}

# For phrase decomposition later
phrases = {"A":["P","D","N"], "D":["A", "N"], "N":["P","V"], "P":["A","D","N"], "V":["D","A","P","N"]}

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
    if "END" in s:                          # For one currently unparsable word
        return -1                           # Return nothing of essence
    moves = []                              # Predefine list of compressions
    wrds = []                               # Predefine sentence words list
    for i in range(len(s)):                 # Loop through all of the words
        if s[i] != "POS":                   # Avoiding the possessive
            wrds.append(translations[s[i]]) # Returning the rest of the list
    last = 0                                # Last list length
    curr = len(wrds)                        # Define current list length
    while curr != last:                     # While there is no change
        last = curr                         # Save last list size
        try:                                # Try to compress...
            for i in range(len(wrds)-1):    # Loop through every bigram
                if wrds[i] == wrds[i+1]:    # Check duplicates
                    moves.append((i,"C"))   # Save move
                    del wrds[i]             # Delete the duplicate
                    i -= 1                  # Reiterate needed in some cases
                if wrds[i] == "A" and wrds[i+1] == "N": # Check for A+N
                    del wrds[i]             # Delete the adjective
                    moves.append((i,"A"))   # Save move
                    i -= 1                  # Reiterate
                if wrds[i] == "ADV" and wrds[i+1] == "V": # Check for ADV+V
                    del wrds[i]             # Delete ADV
                    moves.append((i,"D"))   # Save move
                    i -= 1                  # Reiterate
        except IndexError:                  # If the index is broken, that 
            curr = len(wrds)                # Means at least one word removed

    sents = splt(wrds)                      # Split at conjunctives and modals

    data = []                               # Predefine data array

    for sent in sents:                      # Loop through all sentences
        DP = sent[:sent.index("V")]         # Split into demonstrative phrase
        VP = sent[sent.index("V"):]         # ... and a verb phrase 
        data.append(phrase(DP))             # Parse subject
        data.append(phrase(VP))             # Parse predicate

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

    for i in range(len(cp)/2):           
        try:
            dictionary["S"][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] += 1
            if i > 0:
                p = cp[2*i - 1][-1]
                dictionary[p][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] += 1
        except KeyError:
            try:
                dictionary["S"] == False
            except KeyError:
                dictionary["S"] = {}
            dictionary["S"][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] = 1
            if i > 0:
                p = cp[2*i - 1][-1]
                try:
                    dictionary[p] == False
                except KeyError:
                    dictionary[p] = {}
                dictionary[p][cp[2*i][0] + "P" + cp[2*i + 1][0] + "P"] = 1
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
    return dictionary
    
# print parse(["N","V","V","M","D","N","ADV","ADV","V","P","D","A","N"])