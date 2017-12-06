import numpy as np

translations = {"D":"D","A":"A","N":"N","ADV":"ADV","V":"V", "CC":"+","CD":"A","DT": "D","EX":"D","FW":"END","IN":"P","JJ":"A","JJR":"A","JJS":"A","LS":"A","MD":"-","NN":"N","NNS":"N","NNP":"N","NNPS":"N","PDT":"D","POS":"DS","PRP":"N","PRP$":"D","RB":"ADV","RBR":"ADV","RBS":"ADV","RP":"D","SYM":"N","TO":"V","UH":"END","VB":"V","VBD":"V","VBG":"V","VBN":"V","VBP":"V","VBZ":"V","WDT":"D","WP":"N","WP$":"D","WRB":"ADV",",":"+"}

# phrases = {"TP":{"DPVP","VP"},"DP":{"DNP","DDSNP"},"DDS":{"NDS"},"NP":{"N","NPP","APNP","NPPP"},"VP":{{"DP","AP","PP","TP","CP"},"VVP","VPPP"},"PP":{"PDP"},"CP":{"CTP"},"AA":{"A"},"AP":{{"PP","CP"},"A"},}

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

def dupe(wrds):
    for i in range(len(wrds)-1):
        if wrds[i] == wrds[i+1]:
            return True
    return False

def phrase(wrds):
    checks = [phrases[wrds[0]]]
    print("(" + wrds[0])
    i = 1
    while checks != [] and i < len(wrds):
        curr = checks.pop()
        if wrds[i] in curr:
            print("(" + wrds[i])
            checks.insert(0,phrases[wrds[i]])
            checks.insert(0,phrases[wrds[i]])
        else:
            print(wrds[i] + "]")
        i = i + 1
    print("------")

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
                    del wrds[i]
                    print(wrds)

    except IndexError:
        print("Duplicates removed")

    sents = splt(wrds)
    print(sents)

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