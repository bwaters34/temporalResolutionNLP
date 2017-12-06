# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:19:01 2017

@author: Peter
"""

from urllib import urlopen
from nltk import sent_tokenize # perhaps latter
import re, html2text, time, codecs

def extractor(raw):
    # List for snippets
    sents = []
    # Skip first three pages, because they are likely random stuff
    start = 0
    for i in range(3):
        start = raw.find('[Page', start + 1)
    # Find the end of the html
    end = raw.find('_FINIS._')
    # If no finis found, find the last page
    if end == -1:
        # Else try to find 'NOTES':
        end = raw.rfind('NOTES')
        if end == -1:
            # Else just skip the last page
            end = raw.rfind('[Page')
            # If this is again -1, skip this text by returning -1
            if end == -1:
                return -1
    # Go through line by line
    buf = '' # Buffer to store lines which will need to be combined
    for line in raw[start:end].split('\n'):
        if not re.match('[ ]*$', line) and not line.startswith('[Page') and not line.startswith('####') and '[![View' not in line:
            # Check for images
            if re.search('[/](image|thumbnails)[/]', line) or re.search('[(][.].*[.](gif|png|jpeg)[)]', line):
                continue
            # Check for ** .* **\n instances
            if re.match('[*][*].*[*][*][ ].*', line):    
                continue
            # Else, okay to go
            else:
                # Remove formatting
                rem = re.sub('[_|*]', '', line)
                # Add it to the buffer
                buf += ' ' + rem.strip()
    # return the book
    # for now, all the text from a book will be kept together
    return buf


"""
-----------------------------------------------------------------------------------
The actual function now
-----------------------------------------------------------------------------------
"""

if __name__ == "__main__":

    # Some fields
    year = ''
    missed = []
    processed = []

    # manual
    for i in range(1500, 1825):
        processed.append(str(i))

    
    # ^ logfile was corrupt somehow, need to manually do this now

    # Get all the necessary fields from the logfile
    with open('log.txt') as logfile:
        for line in logfile:
            processed.append(line[:-1])
    

    # Open the code file for reading
    with open('proquest-codes.txt', 'r') as infile:
        with open('log.txt', 'a+') as logfile:
            # Loop through the lines of the infile
            for line in infile:
                # Check if year is finished
                if line.startswith('***'):
                    # If we just finished a year, write that we finished
                    if year != '':
                        logfile.write(year + '\n')
                        print 'Finished year: ' + year
                    
                    # Else, determine if we have already processed this year
                    # If so, keep year as empty string
                    year = line[3:7]
                    if year in processed:
                        year = ''
                    else:
                        print 'Started year: ', year

                elif (year != ''):
                    [title, code] = line[:-1].split(',')
                    print '\tCode: ', code
                    # Get the text from the website
                    # Request the page
                    url = "https://literature.proquest.com/searchFulltext.do?id=%s&divLevel=&area=prose&forward=textsFT&pageSize=&print=No&size=683Kb&queryType=findWork&fromToc=true&warn=Yes" % code
                    
                    # Read the page
                    html = urlopen(url).read()
                    
                    # Get the raw text
                    raw = html2text.html2text(html.decode('utf-8'))
                    
                    # Extract the sentences
                    body = extractor(raw)
                    
                    # Write the sentences to the file if successful read
                    # Note: Using utf-8 encoding
                    if body != -1:
                        with codecs.open('../Yearly/%s-%s-%s.txt' % (year, code, title[:50].replace('/', '')), 'w', 'utf-8') as outfile:
                            outfile.write('%s\n' % body)
                        # Print success
                        print '\t\tParse Successful'

                    else:
                        print 'Missed text: ', year, title, code
                        missed.append( (year, title, code) )
                    
                    # sleep 5 seconds for politeness
                    time.sleep(5)