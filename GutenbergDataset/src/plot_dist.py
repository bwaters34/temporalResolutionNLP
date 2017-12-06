from os import listdir
from os.path import getsize
import matplotlib.pyplot as plt

x = [1500, 1600, 1700, 1800, 1900, 2000]
y = [0, 0, 0, 0, 0, 0]
dirr = '../Yearly/Test'
for f in listdir('../Yearly/Test'):
	year = int(f[:4])
	index = max((year - 1500)/100, 0)
	y[index] += getsize(dirr + '/' + f)
for xx, yy in zip(x,y):
	print xx, yy

plt.plot(x,y)
plt.show()
