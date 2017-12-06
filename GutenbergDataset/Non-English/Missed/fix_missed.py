import urllib3
import time
import json
import sys

urllib3.disable_warnings()
http = urllib3.PoolManager()

urlbegin = 'https://gutenbergapi.org/texts/'
urlend = '/body'

lookup = {'772': "12"}

def download_book(book):
	# Get the metadata
	url = 'https://gutenbergapi.org/texts/' + book
	r = http.request('GET', url)
	# Load into a json file
	mdata = json.loads(r.data.decode('utf-8'))['metadata']
	# Determine if we want to parse
	if u'en' in mdata['language']:
		for url in mdata['formaturi']:
			if url.endswith('txt') and 'readme' not in url:
				text= http.request('GET', url).data
				with open(lookup[book] + '-' + book + '.txt', 'w') as outfile:
					outfile.write(text)
				print("Found book: " + book)
				time.sleep(5)

# create a lookup dict for the missed book years
with open('../ebook.txt', 'r') as infile:
	for line in infile:
		year = line[:4]
		books = line[5:-1].split(' ')
		for book in books:
			if book != '':
				lookup[book] = year
# Retry the books
with open('missed.txt', 'r') as infile:
	for line in infile:
		book = line.split(': ')[1][:-1]
		try:
			download_book(book)
		except KeyError:
			print('Missed Book: ' + str(book))
			time.sleep(5)