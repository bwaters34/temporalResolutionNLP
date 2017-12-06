# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:19:01 2017

@author: Peter
"""

from nltk import sent_tokenize
import re, string

punc_regex = re.compile('[%s]' % re.escape(string.punctuation))


# Given a paragraph, extracts the sentences from it
def add_sentences(sent, temp, N):
    # Replace newlines with spaces
    temp = re.sub('\n', ' ', temp)
    # Remove all formatting
    temp = re.sub('[####.*\n|[*][*].{0,50}[*][*].*\n|*|_]', '', temp)
    # Tokenize and add to the list of sentences
    for string in sent_tokenize(temp):
        if ']' in string or '/images/' in string or (len(string) < 20):
            continue
        else:
            sent.append( punc_regex.sub('', string) )
    # Return the sentences
    return sent


# Given the raw html text, returns the sentences
def extractor(instring):
    # List for sentences
    sent = []
    # Skip first three pages, because they are likely random stuff
    start = 0
    for i in range(3):
        start = instring.find('[Page', start + 1)
    # Now start the main loop
    while True:
        # Find the end of this page
        end = instring.find('[Page', start+1)
        # If we found it, parse the paragraph
        if end != -1:
            # Get the new sentences
            sent = add_sentences(sent, instring[start:end], end-start)
            # Update the start "pointer"
            start = end
                    
        # End no page end was found, see if we can find "FINIS"
        else:
            # Find the FINIS    
            end = instring.find('FINIS', start+1)
            # Check to see if we can find it 
            if end != -1:
                sent = add_sentences(sent, instring[start:end], end-start)
            # Always break
            break
    # return the sentences
    return sent



import html2text
from urllib import urlopen

url = "https://literature.proquest.com/searchFulltext.do?id=Z000050052&divLevel=&area=prose&forward=textsFT&pageSize=&print=No&size=683Kb&queryType=findWork&fromToc=true&warn=Yes"    
html = urlopen(url).read() 
raw = html2text.html2text(html)
snippets = extractor(raw)
for s in snippets[:50]:
    print s
    print ''
