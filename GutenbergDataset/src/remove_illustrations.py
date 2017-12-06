from os import listdir
import re, codecs


for f in [x for x in listdir('../Yearly') if x.endswith('.txt')]:
	with open('../Yearly/%s' % f, 'r') as infile:
		# Read the raw text
		raw = infile.read().decode('utf-8')
		# Go through and remove illustrations
		re.sub('[[]illustration.*[]]', '', raw, re.IGNORECASE)

	with codecs.open('../Yearly/%s' % f, 'w', 'utf-8') as outfile:
		outfile.write(raw)
