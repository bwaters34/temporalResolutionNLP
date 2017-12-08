import numpy as np

translations = {"M":"M","P":"P","D":"D","A":"A","N":"N","ADV":"ADV","V":"V", "CC":"C","CD":"A","DT": "D","EX":"D","FW":"END","IN":"P","JJ":"A","JJR":"A","JJS":"A","LS":"A","MD":"M","NN":"N","NNS":"N","NNP":"N","NNPS":"N","PDT":"D","POS":"DS","PRP":"N","PRP$":"D","RB":"ADV","RBR":"ADV","RBS":"ADV","RP":"D","SYM":"N","TO":"V","UH":"END","VB":"V","VBD":"V","VBG":"V","VBN":"V","VBP":"V","VBZ":"V","WDT":"D","WP":"N","WP$":"D","WRB":"ADV",",":"C"}

phrases = {"A":["P","D","N"], "D":["A", "N"], "N":["P","V"], "P":["A","D","N"], "V":["D","A","P","N"]}

def flat(s):
    t = []
    for h in s:
        for i in h:
            t.append(i)
    return t

def splt(arr):
    sents = []
    temp = []
    for i in range(len(arr)):
        if arr[i] == "M":
            temp.append("M")
            sents.append(temp)
            temp = []
        elif arr[i] == "C":
            temp.append("C")
            sents.append(temp)
            temp = ["DP"]
        else:
            temp.append(arr[i])
    sents.append(temp)
    return sents

def dupe(wrds):
    for i in range(len(wrds)-1):
        if wrds[i] == wrds[i+1]:
            return True
    return False

def phrase(wrds):
    retval = ""
    checks = [phrases[wrds[0]]]
    retval += wrds[0]
    i = 1
    while checks != [] and i < len(wrds):
        curr = checks.pop()
        if wrds[i] in curr:
            retval += wrds[i]
            checks.insert(0,phrases[wrds[i]])
            checks.insert(0,phrases[wrds[i]])
        else:
            retval += wrds[i]
        i = i + 1
    return retval

def parse(s):
    if "END" in s:
        return -1
    moves = []
    wrds = []
    for i in range(len(s)):
        if s[i] != "POS":
            wrds.append(translations[s[i]])
    print(wrds)
    try:
        while dupe(wrds):
            for i in range(len(wrds)-1):
                if wrds[i] == wrds[i+1]:
                    moves.append((i,"C"))
                    del wrds[i]
                    print(wrds)
                    i -= 1
                if wrds[i] == "A" and wrds[i+1] == "N":
                    del wrds[i]
                    moves.append((i,"A"))
                    print(wrds)
                    i -= 1
                if wrds[i] == "ADV" and wrds[i+1] == "V":
                    del wrds[i]
                    moves.append((i,"D"))
                    print(wrds)
                    i -= 1

    except IndexError:
        print("Duplicates removed")

    sents = splt(wrds)
    print(sents)

    data = []

    for sent in sents:
        DP = sent[:sent.index("V")]
        VP = sent[sent.index("V"):]
        data.append(phrase(DP))
        data.append(phrase(VP))

    dictionary = {}

    cp = data
    data = flat(data)

    markers = data

    for m in moves:
        if m[1] == "C":
            try:
                data[m[0]].insert(0,data[m[0][0]])
            except AttributeError or IndexError:
                data[m[0]] = [data[m[0]], data[m[0]]]
        elif m[1] == "A":
            try:
                data[m[0]].insert(0,"A")
            except AttributeError or IndexError:
                data[m[0]] = ["A", data[m[0]]]
        elif m[1] == "D":
            try:
                data[m[0]].insert(0,"ADV")
            except AttributeError or IndexError:
                data[m[0]] = ["ADV", data[m[0]]]
    print data

    print cp

parse(["N","V","V","M","D","N","ADV","V","P","D","N"])