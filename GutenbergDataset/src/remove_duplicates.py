from os import listdir, remove
from shutil import move
from collections import defaultdict
from numpy import mean, std

codes = defaultdict(list)
for f in [x for x in listdir('../Yearly') if x.endswith('.txt')]:
	codes[f[5:-4]].append( int(f[:4]) )

acc = 0
for k in codes:
	if len(codes[k]) > 1:
		print codes[k]
		year = int(mean(codes[k]))
		# rename one file
		move('../Yearly/%s-%s.txt' % (codes[k][0], k), '../Yearly/%s-%s.txt' % (year, k))
		# delete the other
		remove('../Yearly/%s-%s.txt' % (codes[k][1], k))
		acc +=1
print acc, 'duplicates'
		