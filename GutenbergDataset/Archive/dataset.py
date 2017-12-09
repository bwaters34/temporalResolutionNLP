import urllib3
import time
import json
from os import listdir

urllib3.disable_warnings()
http = urllib3.PoolManager()

urlbegin = 'https://gutenbergapi.org/texts/'
urlend = '/body'

filename = 'google.txt'
missed = []

def is_good(book):
	# Get the metadata
	url = 'https://gutenbergapi.org/texts/' + book
	r = http.request('GET', url)
	# Load into a json file
	try:
		mdata = json.loads(r.data.decode('utf-8'))['metadata']
		# Determine if we want to parse
		if u'en' in mdata['language']:
			return mdata['formaturi']
		else:
			print 'Book', book, 'is wrong language.'
			new_whitelist.append(book)
			return 0			
	except (KeyError, ValueError):
		return -1

def download_book(book, year):
	full_url = 'https://gutenbergapi.org/texts/' + book + '/body'
	r = http.request('GET', full_url)
	# Load into a json file
	try:
		text = json.loads(r.data.decode('utf-8'))['body']
		with open('Final/' + year + '-' + book + '.txt', 'w') as outfile:
			outfile.write(text)
			return 1
	# If kweyerror, we need to do a fancy download
	except KeyError:
		return -1
	# If unicode encode error then the text contains strange characters, so pass
	except UnicodeEncodeError:
		return -2

def fancy_download(urls, book, year):
	for url in urls:
		if url.endswith('.txt') and 'readme' not in url:
			text = http.request('GET', url).data.decode('utf-8')
			with open('Final/' + year + '-' + book + '.txt', 'w') as outfile:
				outfile.write(text)
			print("Found book: " + book)
				

# Get a list of already existing books in the collection
cached = []
for fname in listdir('/home/ssbm/Documents/NLP/Data/Final'):
	cached.append( fname[5:-4] )

# whitelist
whitelist = []
new_whitelist = []
with open('whitelist.txt', 'r') as infile:
	for line in infile:
		whitelist.append( line[:-1] )

# Now add all the new books
with open(filename, 'r') as infile:
	for line in infile:
		year = line[:4]
		books = line[5:-1].split(' ')
		for book in books:
			if book != '' and book not in cached and book not in whitelist:
				urls = is_good(book)
				time.sleep(5)
				# Missed the book
				if urls == -1:
					print "Missed Book: ", book
					new_whitelist.append(book)
				# Only if correct language
				elif urls:
					print "Book ", book, " is good!"
					res = download_book(book, year)
					time.sleep(5)
					# Successful download?
					# If no, fancy download is necessary
					if (res == 1):
						print '\tBook added to collection'
					elif (res == -1):
						print '\tNeed to lookup this book'
						fancy_download(urls, book, year)
						time.sleep(5)
					else:
						print "\tMissed Book: ", book
						new_whitelist.append(book)
		# Year done
		print "Completed year:", year


with open('missed.txt', 'w') as miss:
	for year, book in missed:
		miss.write(year + ', ' + book + '\n')

with open('whitelist.txt', 'a') as wfile:
	for book in new_whitelist:
		wfile.write(book + '\n')