from os import listdir, remove

count = 0
for f in [x for x in listdir('../Yearly') if x.endswith('.txt')]:
	with open('../Yearly/%s' % f, 'r') as infile:
		# Read the raw text
		raw = infile.read().decode('utf-8')
		# Check to make sure its not an audio recording
		if 'librivox' in raw.lower() or 'librivox' in raw.lower():
			remove('../Yearly/%s' % f)
			count += 1
print count, ' removed'