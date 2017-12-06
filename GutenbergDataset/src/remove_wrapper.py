

# Imports
import codecs, re
from os import listdir
from gutenberg.cleanup import strip_headers


def extractor(raw):
	# Remove the header
	cleaned = strip_headers(raw)

	# Replace '\r' with '\n'
	cleaned.replace('\r', '\n')

	# Find the first chapter mark
	search1 = re.search('\n[ ]*(Chapter|chapter|CHAPTER)[ ]*(1|I|i)([.])?[ ]*\n', raw)

	# Now instances of (I|1|i)[ ]*\n
	search2 = re.search('\n[ ]*(1|I|i|One|one|One)', raw)

	# Check if both searches worked
	if search1 and search2:
		start = max( search1.start() , search2.start() ) 
	# Check if the first search worked
	elif search1:
		start = search1.start()
	# Check if the second search worked
	elif search2:
		start = search2.start()
	# Else we need to start from the beginning
	else:
		start = 0 

	# Now we have the start, so begin parsing
	buf = ''
	for line in raw[start:].split('\n'):
		# If it is a blank line, continue
		if re.match('[ ]*\n', line):
			continue
		# Check to see if END OF THE PROJECT GUTENBERG in line
		elif 'END OF THE PROJECT GUTENBERG' in line:
			break
		# Else add to the buff
		else:
			buf += line + ' '

	# Return the buffer
	return buf

def search_for_newline(raw, start):
	return 0

def simple_extractor(raw):
	# Length of doc
	N = len(raw)
	
	# ----------------------- Initial header removing -----------------------
	# Find a possible start
	starts = [raw.find('***START'),
		raw.find('*** START'), 
		raw.find('***Start'),
		raw.find('*** Start'),
		]
	start = max([x for x in starts if x < N/4])
	
	# Find a possible end
	ends = [raw.find('***END'),
		raw.find('*** END'),
		raw.find('***End'),
	    raw.find('*** End'),
		raw.find('End of the Project Gutenberg'),
		N]
	end = min([x for x in ends if x != -1])

	# Check to see if this worked
	if start == -1: start = 0
	if end == -1: start = N

	# ----------------------- Fine tuning -----------------------
	# print 'Temp', start, end, N

	# Update the body
	raw = raw[start:end]
	N = end - start

	# reset start and end
	start = 0
	end = N

	# Try looking for any mention of ebook or etext or gutenberg
	regex_res = [x.start() for x in re.finditer('(\n| )e(book|text)|email|(@.{1,20}[.]|gutenberg|www[.]|http)', raw, re.IGNORECASE)]
	# Update start or end based on this result
	if regex_res:
		# print regex_res
		# Filter based on which region of document to look in
		start_res = [x for x in regex_res if x < N/10]
		end_res = [x for x in regex_res if x > N*(1 - 1/10)]
		# Adjust accordingly
		if start_res:
			start = max(start_res)
		if end_res:
			end = min(end_res)

	# Find the next newline after start
	start = raw.find('\n', start)
	
	# Find the previous newline before end
	end = raw.rfind('\n', 0, end)

	#print start, end, N

	# Return up to this point
	return raw[start:end].strip().replace('[Illustration]', '')

i = 0
# Cleanup the files
for f in listdir('../Raw Text'):
	with open('../Raw Text/%s' % f, 'r') as infile:
		# Read the raw text
		raw = infile.read().decode('utf-8')
		
		# Check to make sure its not an audio recording
		if not 'librivox' in raw.lower():
			# print '\n', f
			# Call the extractor function
			body = simple_extractor(raw)

			# Write to the new file
			with codecs.open('../Yearly/%s' % f, 'w', 'utf-8') as outfile:
				outfile.write(body)
			
			# Just a counter to know where we are at
			i += 1
	
	#if i > 10: break
	if (i % 100) == 0:
		print 'Completed', i
		