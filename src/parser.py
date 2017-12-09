import numpy as np

translations = {"D":"D","A":"A","N":"N","ADV":"ADV","V":"V", "CC":"+","CD":"A","DT": "D","EX":"D","FW":"END","IN":"P","JJ":"A","JJR":"A","JJS":"A","LS":"A","MD":"-","NN":"N","NNS":"N","NNP":"N","NNPS":"N","PDT":"D","POS":"DS","PRP":"N","PRP$":"D","RB":"ADV","RBR":"ADV","RBS":"ADV","RP":"D","SYM":"N","TO":"V","UH":"END","VB":"V","VBD":"V","VBG":"V","VBN":"V","VBP":"V","VBZ":"V","WDT":"D","WP":"N","WP$":"D","WRB":"ADV",",":"+",':':'+'}
phrases = {"A":["P","D"], "D":["A", "N"], "N":["P","V"], "P":["A","D","N"], "V":["D","A","P","N"]}

def splt(arr):
    sents = []
    temp = []
    for i in range(len(arr)):
        if arr[i] == "-":
            sents.append(temp)
            temp = []
        elif arr[i] == "+":
            sents.append(temp)
            temp = ["DP"]
        else:
            temp.append(arr[i])
    sents.append(temp)
    return sents

def phrase(words):
    checks = [phrases[words[0]]]
    print("(" + words[0])
    i = 1
    while checks != [] and i < len(words):
        curr = checks.pop()
        if words[i] in curr:
            print("(" + words[i])
            checks.insert(0,phrases[words[i]])
            checks.insert(0,phrases[words[i]])
        else:
            print(words[i] + "]")
        i = i + 1
    print("------")


def parse(s, verbose=False):
    """
    The actual parser
    """
    # Now sure what this does but okay
    if "END" in s:
        return -1

    moves = []
    words = []
    
    # Translate the words to our selected POS tags
    for i in range(len(s)):
        if s[i] != "POS":
            words.append(translations[s[i]])

    # Remove consecutive duplicates
    words = [words[0]] + [words[i] for i in range(1, len(words)) if words[i-1] != words[i]]


    # If verbose, print the words
    if verbose:
        print words

    # 
    sents = splt(words)
    if verbose:
        print sents

    for sent in sents:
        try:
            for i in range(len(sent)):
                if sent[i] == "A" and sent[i+1] == "N":
                    del sent[i]
                    print sent
                if sent[i] == "ADV" and sent[i+1] == "V":
                    del sent[i]
                    print sent
        except IndexError:
            print("A+N and ADV+V compressed")
        DP = sent[:sent.index("V")]
        VP = sent[sent.index("V"):]
        print(DP)
        print(VP)
        phrase(DP)
        phrase(VP)