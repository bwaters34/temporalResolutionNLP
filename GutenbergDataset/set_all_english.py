import urllib3
import time
import json
import sys
from os import listdir
from shutil import move

urllib3.disable_warnings()
http = urllib3.PoolManager()

for filename in listdir('/home/ssbm/Documents/NLP/Data'):
	if not filename.startswith('ebook') and filename.endswith('.txt'):
		[_, book] = filename[:-4].split('-')
		# Get the metadata
		url = 'https://gutenbergapi.org/texts/' + book
		r = http.request('GET', url)
		# Load into a json file
		try:
			mdata = json.loads(r.data.decode('utf-8'))['metadata']
			# Determine if we want to parse
			if u'en' in mdata['language']:
				print('Moving File: ' + filename)
				move(filename, 'Final/' + filename)
			else:
				print('Non-English book: ' + filename)
				
		except KeyError:
			print('Problem with: ' + book)
		time.sleep(5)
