import matplotlib.pyplot as plt
from os import listdir
from collections import defaultdict

years = {}

for filename in listdir('/home/ssbm/Documents/NLP/Data/Final'):
	y = int(filename[:4])
	if y in years:
		years[y] += 1
	else:
		years[y] = 1

counts = defaultdict(float)

for y in years:
	if y > 2000:
		counts['2000s'] += years[y]
	elif y > 1900:
		counts['1900s'] += years[y]
	elif y > 1800:
		counts['1800s'] += years[y]
	elif y > 1700:
		counts['1700s'] += years[y]
	elif y > 1600:
		counts['1600s'] += years[y]
	elif y > 1500:
		counts['1500s'] += years[y]

for x in sorted(counts):
	print x, counts[x]