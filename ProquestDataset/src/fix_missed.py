
from os import listdir, path

from urllib import urlopen
import re, html2text, time, codecs

# Does not skip the first few pages.
def extractor(raw):
    # List for snippets
    sents = []
    
    # Find the first page
    start = raw.find('[Page')
    
    # Find the end of the html
    end = raw.find('You have access to:')
    
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


for f in listdir('../Yearly'):
	if path.getsize('../Yearly/' + f) == 1:
		# Split to get the code part
		spl = f.split('-')
		code = spl[1]

		# Re-download the file
		print 'Code: ', code

		# Get the text from the website
		# Request the page
		url = "https://literature.proquest.com/searchFulltext.do?id=%s&divLevel=&area=prose&forward=textsFT&pageSize=&print=No&size=683Kb&queryType=findWork&fromToc=true&warn=Yes" % code

		# Read the page
		html = urlopen(url).read()

		# Get the raw text
		raw = html2text.html2text(html.decode('utf-8'))

		# Extract the text
		body = extractor(raw)
		print body[:1000]
		print
		time.sleep(5) 