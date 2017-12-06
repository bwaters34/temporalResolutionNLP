# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 22:19:01 2017

@author: Peter
"""

import html2text, re, time
from urllib import urlopen
from collections import defaultdict

# Urls to get the numbers and years of texts
urls = ['https://literature.proquest.com/contents/volumes-uk/Ren_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/Neo_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/Rom_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/BrVic_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/RevEAm_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/AmRomRel_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/Ir_pr.jsp',
        'https://literature.proquest.com/contents/volumes-uk/Sc_pr.jsp']

# list of missed urls
found = []
record = defaultdict(list)
count = 0

for url in urls:
    print 'Starting HTML: ', url
    html = urlopen(url).read() 
    raw = html2text.html2text(html.decode('utf-8'))
    # some flags
    start = False
    author_found = False
    for line in raw.split('\n'):
        # Remove start and end whitespace
        line = line.strip()
        # Check to see if this a ... idk how to explain
        if re.match('[*] [A-Z]', line) and not start:
            start = True
        elif not re.match('[*] [A-Z]', line) and start:
            # If empty line, continue
            if len(line) == 0:
                author_found = False
            elif line.startswith('*') and '__' in line:
                author_found = True
                years = re.findall('[0-9]{4}', line)
            # Termination condition
            elif 'javascript:newToc' in line:
                title = re.split('(\[|\])', line)[2]
                # Edit this?
                year = re.findall('[0-9]{4}', line)[0]
                # If we didnt find a year, then get the year from the author
                if year.startswith('00'):
                    if years:
                        year = years[0]
                    # If no such year, skip it
                    else:
                        continue
                code = re.findall('Z[0-9]+', line)[0]
                if code not in found:
                    record[year].append( (title, code) )
                    found.append(code)
            # Just to get an idea of how many were missed
            elif 'javascript:penguinToc' in line:
                count += 1
    time.sleep(5)

# Write to file
with open('proquest-codes.txt', 'w') as outfile:
    for key in sorted(record):
        outfile.write('***' + key + '***\n')
        texts = record[key]
        for title, code in texts:
            outfile.write(title.replace(',', '').encode('utf-8') + ',' + code.encode('utf-8') + '\n')

"""
# Plotting time
x = []
y = []
for key in sorted(record):
    x.append(int(key))
    y.append(len(record[key]))

import matplotlib.pyplot as plt
import numpy as np
m = np.mean(y)
plt.bar(x, y)
plt.plot(x, [m] * len(x))
"""